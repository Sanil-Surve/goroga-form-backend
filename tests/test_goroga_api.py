"""Comprehensive backend tests for Goroga Appointment API."""
import os
import uuid
from datetime import date, timedelta

import pytest
import requests

BASE_URL = os.environ["REACT_APP_BACKEND_URL"].rstrip("/") if os.environ.get("REACT_APP_BACKEND_URL") else None
if not BASE_URL:
    # fallback: read from frontend/.env
    with open("./frontend/.env") as f:
        for ln in f:
            if ln.startswith("REACT_APP_BACKEND_URL="):
                BASE_URL = ln.strip().split("=", 1)[1].rstrip("/")
                break

ADMIN_EMAIL = "admin@goroga.com"
ADMIN_PASSWORD = "goroga@2026"


def _next_weekday(target_weekday: int) -> str:
    """Return YYYY-MM-DD of the next future date with given weekday (0=Mon, 6=Sun)."""
    today = date.today()
    for i in range(1, 30):
        d = today + timedelta(days=i)
        if d.weekday() == target_weekday:
            return d.isoformat()
    return (today + timedelta(days=7)).isoformat()


MONDAY = _next_weekday(0)
TUESDAY = _next_weekday(1)
SUNDAY = _next_weekday(6)


@pytest.fixture(scope="session")
def admin_token():
    r = requests.post(f"{BASE_URL}/api/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
    assert r.status_code == 200, f"Login failed: {r.text}"
    data = r.json()
    assert "access_token" in data
    assert data["user"]["email"] == ADMIN_EMAIL
    assert data["user"]["role"] == "admin"
    return data["access_token"]


@pytest.fixture(scope="session")
def admin_headers(admin_token):
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture(scope="session")
def created_appt_ids():
    return []


# ----- Health -----
def test_root():
    r = requests.get(f"{BASE_URL}/api/")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


# ----- Availability -----
def test_availability_weekday_returns_40_slots():
    r = requests.get(f"{BASE_URL}/api/availability", params={"date": MONDAY})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["closed"] is False
    assert len(data["slots"]) == 40
    for s in data["slots"][:3]:
        assert "slot" in s and "booked" in s and "capacity" in s and "available" in s
        assert s["capacity"] == 3


def test_availability_sunday_closed():
    r = requests.get(f"{BASE_URL}/api/availability", params={"date": SUNDAY})
    assert r.status_code == 200
    data = r.json()
    assert data["closed"] is True
    assert data["slots"] == []


def test_availability_bad_date():
    r = requests.get(f"{BASE_URL}/api/availability", params={"date": "bad"})
    assert r.status_code == 400


# ----- Appointment creation -----
def _payload(email=None, slot="10:00", d=None):
    return {
        "first_name": "TEST",
        "last_name": "User",
        "email": email or f"test_{uuid.uuid4().hex[:8]}@example.com",
        "phone": "+1-555-1234567",
        "company": "TestCo",
        "designation": "Engineer",
        "concerns": ["stress", "anxiety"],
        "date": d or MONDAY,
        "slot": slot,
    }


def test_create_appointment_success(created_appt_ids):
    p = _payload(slot="10:00")
    r = requests.post(f"{BASE_URL}/api/appointments", json=p)
    assert r.status_code == 201, r.text
    data = r.json()
    assert data["status"] == "booked"
    assert data["email"] == p["email"]
    assert "id" in data
    assert "_id" not in data
    created_appt_ids.append(data["id"])


def test_create_invalid_phone():
    p = _payload(slot="10:15")
    p["phone"] = "abc"
    r = requests.post(f"{BASE_URL}/api/appointments", json=p)
    assert r.status_code == 422


def test_create_invalid_concern():
    p = _payload(slot="10:30")
    p["concerns"] = ["not_a_concern"]
    r = requests.post(f"{BASE_URL}/api/appointments", json=p)
    assert r.status_code == 422


def test_create_sunday_rejected():
    p = _payload(slot="10:00", d=SUNDAY)
    r = requests.post(f"{BASE_URL}/api/appointments", json=p)
    assert r.status_code == 422


def test_create_past_date_rejected():
    p = _payload(slot="10:00", d=(date.today() - timedelta(days=2)).isoformat())
    r = requests.post(f"{BASE_URL}/api/appointments", json=p)
    assert r.status_code == 422


def test_create_invalid_slot_off_grid():
    p = _payload(slot="10:10")
    r = requests.post(f"{BASE_URL}/api/appointments", json=p)
    assert r.status_code == 422


def test_create_slot_out_of_range():
    p = _payload(slot="07:00")
    r = requests.post(f"{BASE_URL}/api/appointments", json=p)
    assert r.status_code == 422
    p2 = _payload(slot="20:00")
    r2 = requests.post(f"{BASE_URL}/api/appointments", json=p2)
    assert r2.status_code == 422


def test_duplicate_email_same_slot(created_appt_ids):
    email = f"dup_{uuid.uuid4().hex[:8]}@example.com"
    p = _payload(email=email, slot="11:00", d=TUESDAY)
    r1 = requests.post(f"{BASE_URL}/api/appointments", json=p)
    assert r1.status_code == 201, r1.text
    created_appt_ids.append(r1.json()["id"])
    r2 = requests.post(f"{BASE_URL}/api/appointments", json=p)
    assert r2.status_code == 409
    assert "already" in r2.json()["detail"].lower()


def test_capacity_limit_409(created_appt_ids):
    slot = "15:30"
    d = TUESDAY
    for i in range(3):
        p = _payload(email=f"cap_{i}_{uuid.uuid4().hex[:6]}@example.com", slot=slot, d=d)
        r = requests.post(f"{BASE_URL}/api/appointments", json=p)
        assert r.status_code == 201, f"booking #{i+1}: {r.text}"
        created_appt_ids.append(r.json()["id"])
    p4 = _payload(email=f"cap_4_{uuid.uuid4().hex[:6]}@example.com", slot=slot, d=d)
    r4 = requests.post(f"{BASE_URL}/api/appointments", json=p4)
    assert r4.status_code == 409
    assert "fully booked" in r4.json()["detail"].lower()


def test_cancelled_does_not_count(admin_headers, created_appt_ids):
    """Cancel one of the booked appointments and verify a new booking succeeds in same slot."""
    slot = "16:00"
    d = TUESDAY
    ids = []
    for i in range(3):
        p = _payload(email=f"cnc_{i}_{uuid.uuid4().hex[:6]}@example.com", slot=slot, d=d)
        r = requests.post(f"{BASE_URL}/api/appointments", json=p)
        assert r.status_code == 201
        ids.append(r.json()["id"])
        created_appt_ids.append(r.json()["id"])
    # 4th should fail
    p4 = _payload(email=f"cnc_4_{uuid.uuid4().hex[:6]}@example.com", slot=slot, d=d)
    assert requests.post(f"{BASE_URL}/api/appointments", json=p4).status_code == 409
    # Cancel one
    rc = requests.patch(f"{BASE_URL}/api/admin/appointments/{ids[0]}/status",
                        json={"status": "cancelled"}, headers=admin_headers)
    assert rc.status_code == 200
    # Now retry
    r5 = requests.post(f"{BASE_URL}/api/appointments", json=p4)
    assert r5.status_code == 201, r5.text
    created_appt_ids.append(r5.json()["id"])


# ----- Auth -----
def test_auth_me_requires_token():
    r = requests.get(f"{BASE_URL}/api/auth/me")
    assert r.status_code == 401


def test_auth_me_with_token(admin_headers):
    r = requests.get(f"{BASE_URL}/api/auth/me", headers=admin_headers)
    assert r.status_code == 200
    assert r.json()["email"] == ADMIN_EMAIL


def test_auth_login_invalid():
    r = requests.post(f"{BASE_URL}/api/auth/login", json={"email": ADMIN_EMAIL, "password": "wrong"})
    assert r.status_code == 401


# ----- Admin list and filters -----
def test_admin_list_unauth():
    r = requests.get(f"{BASE_URL}/api/admin/appointments")
    assert r.status_code == 401


def test_admin_list_with_stats(admin_headers):
    r = requests.get(f"{BASE_URL}/api/admin/appointments", headers=admin_headers)
    assert r.status_code == 200
    data = r.json()
    assert "items" in data and "stats" in data and "concerns_analytics" in data
    s = data["stats"]
    for k in ("total", "booked", "completed", "cancelled"):
        assert k in s
    assert isinstance(data["concerns_analytics"], dict)


def test_admin_filter_status(admin_headers):
    r = requests.get(f"{BASE_URL}/api/admin/appointments", params={"status": "booked"}, headers=admin_headers)
    assert r.status_code == 200
    for it in r.json()["items"]:
        assert it["status"] == "booked"


def test_admin_filter_date_range(admin_headers):
    r = requests.get(f"{BASE_URL}/api/admin/appointments",
                     params={"date_from": MONDAY, "date_to": TUESDAY}, headers=admin_headers)
    assert r.status_code == 200
    for it in r.json()["items"]:
        assert MONDAY <= it["date"] <= TUESDAY


def test_admin_search_q(admin_headers):
    r = requests.get(f"{BASE_URL}/api/admin/appointments", params={"q": "TestCo"}, headers=admin_headers)
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) >= 1
    assert any("TestCo" in i.get("company", "") for i in items)


# ----- Admin update/delete -----
def test_admin_patch_status_and_delete(admin_headers, created_appt_ids):
    p = _payload(slot="10:00", d=TUESDAY, email=f"upd_{uuid.uuid4().hex[:6]}@example.com")
    r = requests.post(f"{BASE_URL}/api/appointments", json=p)
    assert r.status_code == 201
    aid = r.json()["id"]

    rp = requests.patch(f"{BASE_URL}/api/admin/appointments/{aid}/status",
                        json={"status": "completed"}, headers=admin_headers)
    assert rp.status_code == 200
    assert rp.json()["status"] == "completed"

    rd = requests.delete(f"{BASE_URL}/api/admin/appointments/{aid}", headers=admin_headers)
    assert rd.status_code == 200
    assert rd.json()["deleted"] is True

    rg = requests.get(f"{BASE_URL}/api/admin/appointments", params={"q": p["email"]}, headers=admin_headers)
    assert rg.status_code == 200
    assert not any(i["id"] == aid for i in rg.json()["items"])


def test_admin_delete_404(admin_headers):
    r = requests.delete(f"{BASE_URL}/api/admin/appointments/non-existent", headers=admin_headers)
    assert r.status_code == 404


# ----- Allowed Dates CRUD -----
def test_get_allowed_dates_public():
    r = requests.get(f"{BASE_URL}/api/allowed-dates")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert "2026-06-11" in data


def test_add_allowed_date_unauth():
    r = requests.post(f"{BASE_URL}/api/admin/allowed-dates", json={"date": "2026-06-12"})
    assert r.status_code == 401


def test_add_delete_allowed_date_auth(admin_headers):
    r_add = requests.post(
        f"{BASE_URL}/api/admin/allowed-dates",
        json={"date": "2026-06-12"},
        headers=admin_headers
    )
    assert r_add.status_code == 201
    assert r_add.json()["success"] is True

    r_get = requests.get(f"{BASE_URL}/api/allowed-dates")
    assert "2026-06-12" in r_get.json()

    r_del = requests.delete(f"{BASE_URL}/api/admin/allowed-dates/2026-06-12", headers=admin_headers)
    assert r_del.status_code == 200
    assert r_del.json()["deleted"] is True

    r_get2 = requests.get(f"{BASE_URL}/api/allowed-dates")
    assert "2026-06-12" not in r_get2.json()


# ----- Cleanup -----
def test_zzz_cleanup_test_data(admin_headers, created_appt_ids):
    for aid in created_appt_ids:
        requests.delete(f"{BASE_URL}/api/admin/appointments/{aid}", headers=admin_headers)
