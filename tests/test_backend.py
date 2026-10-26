"""
Comprehensive test suite for EmbedEye backend API, security, staging, and jobs.
Tests every route, auth failure, origin filtering, path validation, 409 busy, and cancellation.
"""

import os
import shutil
import tempfile
import time
from pathlib import Path
from fastapi.testclient import TestClient

# Set test token before imports
TEST_TOKEN = "test-secret-token-12345"
os.environ["EMBEDEYE_TOKEN"] = TEST_TOKEN

from backend.api import app
from backend.jobs import job_manager
from backend.staging import StagingStore

client = TestClient(app)


def test_health_unauthenticated():
    """Verify /health returns startup status without requiring authorization."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "state" in data
    assert "detail" in data


def test_auth_rejection():
    """Verify endpoints reject requests with missing or invalid tokens."""
    # Missing auth
    res1 = client.get("/status")
    assert res1.status_code == 401

    # Bad scheme
    res2 = client.get("/status", headers={"Authorization": f"Basic {TEST_TOKEN}"})
    assert res2.status_code == 401

    # Wrong token
    res3 = client.get("/status", headers={"Authorization": "Bearer wrong-token"})
    assert res3.status_code == 403


def test_auth_success():
    """Verify endpoint succeeds with correct Bearer token."""
    res = client.get("/status", headers={"Authorization": f"Bearer {TEST_TOKEN}"})
    assert res.status_code == 200
    data = res.json()
    assert "device" in data
    assert "vram" in data


def test_foreign_origin_rejected():
    """Verify foreign Origins are rejected with 403."""
    res = client.get(
        "/status",
        headers={
            "Authorization": f"Bearer {TEST_TOKEN}",
            "Origin": "http://malicious-website.com",
        },
    )
    assert res.status_code == 403


def test_path_validation():
    """Verify path validation handles bad paths, non-existent directories, and files."""
    headers = {"Authorization": f"Bearer {TEST_TOKEN}"}

    # Empty path
    res1 = client.get("/folder/summary?path=", headers=headers)
    assert res1.status_code in (400, 422)

    # Non-existent path
    res2 = client.get("/folder/summary?path=Z:/NonExistentFolder12345", headers=headers)
    assert res2.status_code == 404

    # File instead of directory
    test_file = Path(__file__).resolve()
    res3 = client.get(f"/folder/summary?path={test_file}", headers=headers)
    assert res3.status_code == 400


def test_staging_store():
    """Verify SQLite staging store CRUD methods."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_db = Path(tmp_dir) / "test_staging.db"
        store = StagingStore(tmp_db)

        # Upsert
        file1 = str(Path(tmp_dir) / "doc1.png")
        store.upsert(file1, mtime=100.0, size=1024, mode="ocr", ocr_text="Invoice #1")

        # Unchanged check
        assert store.is_staged_and_unchanged(file1, 100.0) is True
        assert store.is_staged_and_unchanged(file1, 101.0) is False

        # In folder summary
        summary = store.count_in_folder(tmp_dir)
        assert summary["total"] == 1
        assert summary["ocr"] == 1
        assert summary["ocr_caption"] == 0

        # Delete
        deleted = store.delete_in_folder(tmp_dir)
        assert deleted == 1
        assert store.count_in_folder(tmp_dir)["total"] == 0


def test_folder_summary_valid():
    """Verify /folder/summary on a real directory."""
    headers = {"Authorization": f"Bearer {TEST_TOKEN}"}
    sample_dir = Path(__file__).resolve().parent.parent / "src"
    res = client.get(f"/folder/summary?path={sample_dir}", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert "files_found" in data
    assert "staged" in data
    assert "stored" in data


def test_job_busy_and_cancellation():
    """Verify that launching a second job returns HTTP 409 and cancel works."""
    headers = {"Authorization": f"Bearer {TEST_TOKEN}"}
    sample_dir = str(Path(__file__).resolve().parent.parent / "src")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a dummy folder with 10 files
        for i in range(10):
            p = Path(tmp_dir) / f"dummy_{i}.png"
            p.write_bytes(b"\x89PNG\r\n\x1a\n")

        # Start first job
        res1 = client.post("/jobs/ocr", json={"folder": tmp_dir}, headers=headers)
        assert res1.status_code == 200
        job_id = res1.json()["job_id"]

        # Attempt to start second job while first is running -> 409
        res2 = client.post("/jobs/ocr", json={"folder": tmp_dir}, headers=headers)
        assert res2.status_code == 409
        assert res2.json()["detail"]["error"] == "busy"

        # Cancel active job
        res_cancel = client.post("/jobs/cancel", headers=headers)
        assert res_cancel.status_code == 200
        assert res_cancel.json()["cancelled"] is True

        # Wait for worker to stop
        for _ in range(20):
            if not job_manager.get_active_job():
                break
            time.sleep(0.1)

        # Should now be idle again
        active = job_manager.get_active_job()
        assert active is None or not active.get("job_id")


def test_search_endpoint():
    """Verify /search endpoint returns search results structure."""
    headers = {"Authorization": f"Bearer {TEST_TOKEN}"}
    res = client.post(
        "/search",
        json={"query": "test query", "mode": "visual", "top_k": 3},
        headers=headers,
    )
    assert res.status_code == 200
    data = res.json()
    assert "query" in data
    assert "results" in data
    assert isinstance(data["results"], list)
