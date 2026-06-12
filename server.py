from dotenv import load_dotenv
from pathlib import Path

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

import os
import re
import uuid
import logging
from datetime import datetime, timezone, timedelta, date as date_cls
from typing import List, Optional, Annotated

import bcrypt
import jwt
from fastapi import FastAPI, APIRouter, HTTPException, Depends, Request, Query, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.cors import CORSMiddleware
import asyncio
import resend
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError
from pydantic import BaseModel, Field, EmailStr, ConfigDict, field_validator

# -------------------- MongoDB --------------------
mongo_url = os.environ["MONGO_URL"]
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ["DB_NAME"]]

# -------------------- Config --------------------
JWT_SECRET = os.environ["JWT_SECRET"]
JWT_ALGORITHM = "HS256"
MAX_BOOKINGS_PER_SLOT = int(os.environ.get("MAX_BOOKINGS_PER_SLOT", "3"))
ADMIN_EMAIL = os.environ["ADMIN_EMAIL"]
ADMIN_PASSWORD = os.environ["ADMIN_PASSWORD"]
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
resend.api_key = RESEND_API_KEY

CONCERN_OPTIONS = {
    "stress",
    "poor_sleep",
    "anxiety",
    "mental_fatigue",
    "lack_of_focus",
    "screen_fatigue",
    "other",
}

APPOINTMENT_STATUSES = {"booked", "completed", "cancelled"}

# -------------------- App --------------------
app = FastAPI(title="Goroga Appointment API", version="1.0.0")
api = APIRouter(prefix="/api")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# -------------------- Auth helpers --------------------
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=12),
        "type": "access",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_admin(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> dict:
    token = None
    if creds and creds.scheme.lower() == "bearer":
        token = creds.credentials
    if not token:
        token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = await db.users.find_one({"id": payload.get("sub")}, {"_id": 0, "password_hash": 0})
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=401, detail="Not authorized")
    return user


# -------------------- Models --------------------
PHONE_RE = re.compile(r"^[+\d][\d\s\-()]{6,}$")
SLOT_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class LoginInput(BaseModel):
    email: EmailStr
    password: str


class AppointmentCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    first_name: str = Field(min_length=1, max_length=50)
    last_name: str = Field(min_length=1, max_length=50)
    email: EmailStr
    phone: str = Field(min_length=7, max_length=20)
    company: str = Field(min_length=1, max_length=100)
    designation: str = Field(min_length=1, max_length=100)
    concerns: List[str] = Field(min_length=1)
    other_concern: Optional[str] = Field(default=None, max_length=200)
    date: str  # YYYY-MM-DD
    slot: str  # HH:MM (24h)

    @field_validator("phone")
    @classmethod
    def _phone(cls, v: str) -> str:
        if not PHONE_RE.match(v):
            raise ValueError("Invalid phone number")
        return v

    @field_validator("concerns")
    @classmethod
    def _concerns(cls, v: List[str]) -> List[str]:
        bad = [c for c in v if c not in CONCERN_OPTIONS]
        if bad:
            raise ValueError(f"Invalid concerns: {bad}")
        return v

    @field_validator("date")
    @classmethod
    def _date(cls, v: str) -> str:
        if not DATE_RE.match(v):
            raise ValueError("Date must be YYYY-MM-DD")
        d = datetime.strptime(v, "%Y-%m-%d").date()
        if d.weekday() == 6:  # Sunday
            raise ValueError("Sundays are not available")
        if d < datetime.now(timezone.utc).date():
            raise ValueError("Date must be today or in the future")
        return v

    @field_validator("slot")
    @classmethod
    def _slot(cls, v: str) -> str:
        if not SLOT_RE.match(v):
            raise ValueError("Slot must be HH:MM (24h)")
        h, m = map(int, v.split(":"))
        # 10:00 to 19:45 inclusive, every 15 min
        if h < 10 or h > 19 or (h == 19 and m > 45):
            raise ValueError("Slot must be between 10:00 and 19:45")
        if m not in (0, 15, 30, 45):
            raise ValueError("Slot must be on a 15-minute interval")
        return v


class StatusUpdate(BaseModel):
    status: str

    @field_validator("status")
    @classmethod
    def _status(cls, v: str) -> str:
        if v not in {"completed", "cancelled", "booked"}:
            raise ValueError("Invalid status")
        return v


# -------------------- Helpers --------------------
def generate_slots() -> List[str]:
    """All 15-min slots from 10:00 to 19:45."""
    slots = []
    for h in range(10, 20):
        for m in (0, 15, 30, 45):
            if h == 20:
                break
            slots.append(f"{h:02d}:{m:02d}")
    return slots


ALL_SLOTS = generate_slots()


def serialize_appt(doc: dict) -> dict:
    doc = {k: v for k, v in doc.items() if k != "_id"}
    return doc


# -------------------- Email --------------------
CONCERN_LABELS = {
    "stress": "Stress",
    "poor_sleep": "Poor Sleep",
    "anxiety": "Anxiety",
    "mental_fatigue": "Mental Fatigue",
    "lack_of_focus": "Lack of Focus",
    "screen_fatigue": "Screen Fatigue",
    "other": "Other",
}


async def send_confirmation_email(doc: dict) -> None:
    """Send appointment confirmation email via Resend. Runs as a background task."""
    if not RESEND_API_KEY:
        logger.warning("RESEND_API_KEY not set — skipping confirmation email")
        return

    try:
        name = f"{doc['first_name']} {doc['last_name']}"
        concerns_str = ", ".join(CONCERN_LABELS.get(c, c) for c in doc.get("concerns", []))
        other = doc.get("other_concern") or ""
        other_row = f"""
            <tr>
              <td style="padding:6px 0;color:#6b7280;font-size:14px;">Other concern</td>
              <td style="padding:6px 0;font-size:14px;font-weight:600;">{other}</td>
            </tr>""" if other else ""

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8" /><meta name="viewport" content="width=device-width,initial-scale=1" /></head>
<body style="margin:0;padding:0;background:#f9f8f6;font-family:'Inter',Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#f9f8f6;padding:40px 0;">
    <tr><td align="center">
      <table width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 2px 16px rgba(0,0,0,0.07);">

        <!-- Header -->
        <tr>
          <td style="background:#111827;padding:32px 40px;text-align:center;">
            <h1 style="margin:0;color:#ffffff;font-size:24px;font-weight:700;letter-spacing:-0.5px;">Goroga</h1>
            <p style="margin:8px 0 0;color:#9ca3af;font-size:13px;letter-spacing:0.1em;">ANTI STRESS WEARABLE DEVICE</p>
          </td>
        </tr>

        <!-- Greeting -->
        <tr>
          <td style="padding:36px 40px 0;">
            <h2 style="margin:0 0 8px;font-size:20px;color:#111827;">Appointment Confirmed ✅</h2>
            <p style="margin:0;color:#4b5563;font-size:15px;line-height:1.6;">Hi {name}, your appointment has been successfully booked. Here are your details:</p>
          </td>
        </tr>

        <!-- Details card -->
        <tr>
          <td style="padding:24px 40px;">
            <table width="100%" cellpadding="0" cellspacing="0" style="background:#f3f4f6;border-radius:8px;padding:20px;">
              <tr>
                <td style="padding:6px 0;color:#6b7280;font-size:14px;">Date</td>
                <td style="padding:6px 0;font-size:14px;font-weight:600;">{doc['date']}</td>
              </tr>
              <tr>
                <td style="padding:6px 0;color:#6b7280;font-size:14px;">Time</td>
                <td style="padding:6px 0;font-size:14px;font-weight:600;">{doc['slot']}</td>
              </tr>
              <tr>
                <td style="padding:6px 0;color:#6b7280;font-size:14px;">Name</td>
                <td style="padding:6px 0;font-size:14px;font-weight:600;">{name}</td>
              </tr>
              <tr>
                <td style="padding:6px 0;color:#6b7280;font-size:14px;">Email</td>
                <td style="padding:6px 0;font-size:14px;font-weight:600;">{doc['email']}</td>
              </tr>
              <tr>
                <td style="padding:6px 0;color:#6b7280;font-size:14px;">Phone</td>
                <td style="padding:6px 0;font-size:14px;font-weight:600;">{doc['phone']}</td>
              </tr>
              <tr>
                <td style="padding:6px 0;color:#6b7280;font-size:14px;">Company</td>
                <td style="padding:6px 0;font-size:14px;font-weight:600;">{doc['company']}</td>
              </tr>
              <tr>
                <td style="padding:6px 0;color:#6b7280;font-size:14px;">Designation</td>
                <td style="padding:6px 0;font-size:14px;font-weight:600;">{doc['designation']}</td>
              </tr>
              <tr>
                <td style="padding:6px 0;color:#6b7280;font-size:14px;">Concerns</td>
                <td style="padding:6px 0;font-size:14px;font-weight:600;">{concerns_str}</td>
              </tr>
              {other_row}
              <tr>
                <td style="padding:6px 0;color:#6b7280;font-size:14px;">Booking ID</td>
                <td style="padding:6px 0;font-size:12px;color:#9ca3af;font-family:monospace;">{doc['id']}</td>
              </tr>
            </table>
          </td>
        </tr>

        <!-- Footer -->
        <tr>
          <td style="padding:24px 40px 36px;text-align:center;">
            <p style="margin:0;color:#9ca3af;font-size:12px;">If you need to reschedule or cancel, please contact us.<br/>© 2025 Goroga · Anti Stress Wearable Device</p>
          </td>
        </tr>

      </table>
    </td></tr>
  </table>
</body>
</html>
"""

        await asyncio.to_thread(
            resend.Emails.send,
            {
                "from": "Goroga <appointment@goroga.in>",
                "to": [doc["email"]],
                "subject": f"Your Goroga Appointment — {doc['date']} at {doc['slot']}",
                "html": html,
            },
        )
        logger.info("Confirmation email sent to %s", doc["email"])
    except Exception as exc:
        logger.error("Failed to send confirmation email: %s", exc)


# -------------------- Routes --------------------
@api.get("/")
async def root():
    return {"service": "Goroga Appointment API", "status": "ok"}


# ---- Auth ----
@api.post("/auth/login")
async def login(payload: LoginInput):
    email = payload.email.lower()
    user = await db.users.find_one({"email": email})
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token(user["id"], user["email"])
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"id": user["id"], "email": user["email"], "name": user.get("name", "Admin"), "role": user["role"]},
    }


@api.get("/auth/me")
async def me(admin: dict = Depends(get_current_admin)):
    return admin


class AllowedDateInput(BaseModel):
    date: str

    @field_validator("date")
    @classmethod
    def _date(cls, v: str) -> str:
        if not DATE_RE.match(v):
            raise ValueError("Date must be YYYY-MM-DD")
        return v


# ---- Allowed Dates ----
@api.get("/allowed-dates")
async def get_allowed_dates():
    cursor = db.allowed_dates.find({}, {"_id": 0})
    dates = [doc["date"] async for doc in cursor]
    return sorted(dates)


@api.post("/admin/allowed-dates", status_code=201)
async def add_allowed_date(payload: AllowedDateInput, admin: dict = Depends(get_current_admin)):
    await db.allowed_dates.update_one(
        {"date": payload.date},
        {"$set": {"date": payload.date}},
        upsert=True
    )
    return {"success": True, "date": payload.date}


@api.delete("/admin/allowed-dates/{date}")
async def delete_allowed_date(date: str, admin: dict = Depends(get_current_admin)):
    if not DATE_RE.match(date):
        raise HTTPException(status_code=400, detail="Invalid date format")
    res = await db.allowed_dates.delete_one({"date": date})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Date not found")
    return {"deleted": True, "date": date}


# ---- Availability ----
@api.get("/availability")
async def availability(date: str = Query(..., description="YYYY-MM-DD")):
    if not DATE_RE.match(date):
        raise HTTPException(status_code=400, detail="Date must be YYYY-MM-DD")
    d = datetime.strptime(date, "%Y-%m-%d").date()
    if d.weekday() == 6:
        return {"date": date, "slots": [], "closed": True, "reason": "Sundays are closed"}

    # Count active bookings per slot (exclude cancelled)
    pipeline = [
        {"$match": {"date": date, "status": {"$ne": "cancelled"}}},
        {"$group": {"_id": "$slot", "count": {"$sum": 1}}},
    ]
    counts = {row["_id"]: row["count"] async for row in db.appointments.aggregate(pipeline)}

    now = datetime.now(timezone.utc)
    today = now.date()
    cur_minutes = now.hour * 60 + now.minute

    result = []
    for s in ALL_SLOTS:
        h, m = map(int, s.split(":"))
        is_past = d == today and (h * 60 + m) <= cur_minutes
        used = counts.get(s, 0)
        result.append(
            {
                "slot": s,
                "booked": used,
                "capacity": MAX_BOOKINGS_PER_SLOT,
                "available": (not is_past) and used < MAX_BOOKINGS_PER_SLOT,
                "is_past": is_past,
            }
        )
    return {"date": date, "slots": result, "closed": False, "max_per_slot": MAX_BOOKINGS_PER_SLOT}


# ---- Public booking ----
@api.post("/appointments", status_code=201)
async def create_appointment(payload: AppointmentCreate):
    """
    Atomically reserve a seat in slot_counts before inserting the appointment.

    Race-condition fix
    ──────────────────
    Old approach:  count_documents → insert   (two round-trips; two concurrent
                   requests both pass the count check → overbooking)

    New approach:  find_one_and_update with filter {count: {$lt: MAX}} + upsert
                   ┌ document doesn't exist yet  → upsert creates {count:1}  ✅
                   ├ document exists, count < MAX → increments count          ✅
                   └ document exists, count ≥ MAX → filter misses, upsert
                     tries to insert a second doc for the same (date,slot)
                     → DuplicateKeyError (unique index)  → slot full          ✅
    All three branches are a single atomic MongoDB operation.
    """

    # ── Step 1: atomically claim one seat ──────────────────────────────────────
    try:
        await db.slot_counts.find_one_and_update(
            {
                "date": payload.date,
                "slot": payload.slot,
                "count": {"$lt": MAX_BOOKINGS_PER_SLOT},
            },
            {
                "$inc": {"count": 1},
                "$setOnInsert": {"date": payload.date, "slot": payload.slot},
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
    except DuplicateKeyError:
        # A doc for this (date, slot) already exists with count >= MAX
        raise HTTPException(
            status_code=409,
            detail="This slot is fully booked. Please choose another.",
        )

    # ── Step 2: prevent the same email booking the same slot twice ─────────────
    dup = await db.appointments.find_one(
        {
            "email": payload.email.lower(),
            "date": payload.date,
            "slot": payload.slot,
            "status": {"$ne": "cancelled"},
        }
    )
    if dup:
        # Release the seat we just claimed
        await db.slot_counts.update_one(
            {"date": payload.date, "slot": payload.slot},
            {"$inc": {"count": -1}},
        )
        raise HTTPException(
            status_code=409,
            detail="You already have an appointment for this slot.",
        )

    # ── Step 3: persist the appointment ───────────────────────────────────────
    appt_id = str(uuid.uuid4())
    doc = {
        "id": appt_id,
        "first_name": payload.first_name,
        "last_name": payload.last_name,
        "email": payload.email.lower(),
        "phone": payload.phone,
        "company": payload.company,
        "designation": payload.designation,
        "concerns": payload.concerns,
        "other_concern": payload.other_concern,
        "date": payload.date,
        "slot": payload.slot,
        "status": "booked",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        await db.appointments.insert_one(doc)
    except Exception:
        # Any unexpected insert failure → release the claimed seat
        await db.slot_counts.update_one(
            {"date": payload.date, "slot": payload.slot},
            {"$inc": {"count": -1}},
        )
        raise

    asyncio.create_task(send_confirmation_email(doc))
    return serialize_appt(doc)


# ---- Admin: list appointments ----
@api.get("/admin/appointments")
async def list_appointments(
    admin: dict = Depends(get_current_admin),
    status_filter: Optional[str] = Query(None, alias="status"),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    q: Optional[str] = Query(None),
    page: int = Query(1, ge=1, description="1-indexed page number"),
    page_size: int = Query(10, ge=1, le=100, description="Records per page"),
):
    # ── Build filter query ─────────────────────────────────────────────────────
    query: dict = {}
    if status_filter and status_filter in APPOINTMENT_STATUSES:
        query["status"] = status_filter
    if date_from or date_to:
        rng: dict = {}
        if date_from and DATE_RE.match(date_from):
            rng["$gte"] = date_from
        if date_to and DATE_RE.match(date_to):
            rng["$lte"] = date_to
        if rng:
            query["date"] = rng
    if q:
        regex = {"$regex": re.escape(q), "$options": "i"}
        query["$or"] = [
            {"first_name": regex},
            {"last_name": regex},
            {"email": regex},
            {"phone": regex},
            {"company": regex},
        ]

    # ── Count filtered total (for pagination math) and fetch one page ──────────
    skip = (page - 1) * page_size

    async def get_concerns_analytics():
        pipeline = [
            {"$match": query},
            {"$unwind": "$concerns"},
            {"$group": {"_id": "$concerns", "count": {"$sum": 1}}}
        ]
        analytics = {}
        async for row in db.appointments.aggregate(pipeline):
            if row["_id"]:
                analytics[row["_id"]] = row["count"]
        return analytics

    # Run count + 4 stat queries + concerns analytics concurrently; cursor is built separately (Motor
    # cursors are async iterables, not awaitables, so can't go into gather).
    filtered_total, total_all, booked, completed, cancelled, concerns_analytics = await asyncio.gather(
        db.appointments.count_documents(query),
        db.appointments.count_documents({}),
        db.appointments.count_documents({"status": "booked"}),
        db.appointments.count_documents({"status": "completed"}),
        db.appointments.count_documents({"status": "cancelled"}),
        get_concerns_analytics(),
    )

    cursor = (
        db.appointments.find(query, {"_id": 0})
        .sort([("date", -1), ("slot", -1)])
        .skip(skip)
        .limit(page_size)
    )
    items = [doc async for doc in cursor]

    return {
        "items": items,
        "total": filtered_total,          # filtered count → drives pagination UI
        "page": page,
        "page_size": page_size,
        "stats": {
            "total": total_all,
            "booked": booked,
            "completed": completed,
            "cancelled": cancelled,
        },
        "concerns_analytics": concerns_analytics,
    }


@api.patch("/admin/appointments/{appt_id}/status")
async def update_appointment_status(
    appt_id: str, payload: StatusUpdate, admin: dict = Depends(get_current_admin)
):
    # Read the appointment first so we know the old status
    existing = await db.appointments.find_one({"id": appt_id}, {"_id": 0})
    if not existing:
        raise HTTPException(status_code=404, detail="Appointment not found")

    old_status = existing["status"]
    new_status = payload.status

    await db.appointments.update_one(
        {"id": appt_id},
        {"$set": {"status": new_status, "updated_at": datetime.now(timezone.utc).isoformat()}},
    )

    # ── Keep slot_counts in sync ───────────────────────────────────────────────
    # Transitioning TO cancelled → free up a seat
    if new_status == "cancelled" and old_status != "cancelled":
        await db.slot_counts.update_one(
            {"date": existing["date"], "slot": existing["slot"]},
            {"$inc": {"count": -1}},
        )
    # Transitioning FROM cancelled → claim a seat back
    elif old_status == "cancelled" and new_status != "cancelled":
        await db.slot_counts.update_one(
            {"date": existing["date"], "slot": existing["slot"]},
            {"$inc": {"count": 1}},
        )

    doc = await db.appointments.find_one({"id": appt_id}, {"_id": 0})
    return doc


@api.delete("/admin/appointments/{appt_id}")
async def delete_appointment(appt_id: str, admin: dict = Depends(get_current_admin)):
    existing = await db.appointments.find_one({"id": appt_id}, {"_id": 0})
    if not existing:
        raise HTTPException(status_code=404, detail="Appointment not found")

    await db.appointments.delete_one({"id": appt_id})

    # Only release the seat if the appointment was NOT already cancelled
    # (cancelled appointments already freed their seat when they were cancelled)
    if existing["status"] != "cancelled":
        await db.slot_counts.update_one(
            {"date": existing["date"], "slot": existing["slot"]},
            {"$inc": {"count": -1}},
        )

    return {"deleted": True, "id": appt_id}


# -------------------- Startup --------------------
@app.on_event("startup")
async def startup_event():
    # ── Core indexes ───────────────────────────────────────────────────────────
    await db.users.create_index("email", unique=True)
    await db.appointments.create_index([("date", 1), ("slot", 1)])
    await db.appointments.create_index("status")
    await db.appointments.create_index("email")

    # ── slot_counts unique index (enforces atomicity in the booking check) ─────
    await db.slot_counts.create_index(
        [("date", 1), ("slot", 1)], unique=True
    )

    # ── Rebuild slot_counts from existing appointments (migration-safe) ─────────
    # This runs on every startup and is idempotent. It ensures slot_counts
    # accurately reflects the live DB even if the server was deployed without
    # the new code, or after a crash mid-operation.
    pipeline = [
        {"$match": {"status": {"$ne": "cancelled"}}},
        {"$group": {"_id": {"date": "$date", "slot": "$slot"}, "count": {"$sum": 1}}},
    ]
    async for row in db.appointments.aggregate(pipeline):
        await db.slot_counts.update_one(
            {"date": row["_id"]["date"], "slot": row["_id"]["slot"]},
            {"$set": {"count": row["count"]}},
            upsert=True,
        )
    logger.info("slot_counts rebuilt from existing appointments")

    # ── Seed admin ─────────────────────────────────────────────────────────────
    existing = await db.users.find_one({"email": ADMIN_EMAIL.lower()})
    if not existing:
        await db.users.insert_one(
            {
                "id": str(uuid.uuid4()),
                "email": ADMIN_EMAIL.lower(),
                "password_hash": hash_password(ADMIN_PASSWORD),
                "name": "Admin",
                "role": "admin",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        logger.info("Seeded admin user: %s", ADMIN_EMAIL)
    elif not verify_password(ADMIN_PASSWORD, existing["password_hash"]):
        await db.users.update_one(
            {"email": ADMIN_EMAIL.lower()},
            {"$set": {"password_hash": hash_password(ADMIN_PASSWORD)}},
        )
        logger.info("Updated admin password for: %s", ADMIN_EMAIL)

    # ── Seed allowed dates ─────────────────────────────────────────────────────
    existing_date = await db.allowed_dates.find_one({"date": "2026-06-11"})
    if not existing_date:
        count = await db.allowed_dates.count_documents({})
        if count == 0:
            await db.allowed_dates.insert_one({"date": "2026-06-11"})
            logger.info("Seeded default allowed date: 2026-06-11")


@app.on_event("shutdown")
async def shutdown_event():
    client.close()


app.include_router(api)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)
