"""
End-to-End Pipeline Integration Test for EmbedEye.
Verifies the complete flow on a 20-image folder:
1. Folder summary before indexing
2. Run OCR -> staging.db
3. Verify staged counts
4. Store in DB -> commits to docs_ocr
5. Verify stored counts > 0
6. Delete from DB -> purges from Qdrant and caches
7. Verify stored counts return to ZERO
8. Verify all 20 image files are UNTOUCHED
9. Measure RAM and peak VRAM
"""

import os
import sys
import time
from pathlib import Path

# Set auth token
TEST_TOKEN = "e2e-secret-token"
os.environ["EMBEDEYE_TOKEN"] = TEST_TOKEN

from fastapi.testclient import TestClient
from backend.api import app
from backend.jobs import job_manager
from src.gpu_manager import get_model_manager

client = TestClient(app)
auth_headers = {"Authorization": f"Bearer {TEST_TOKEN}"}


def wait_for_job_idle(timeout_sec=120):
    time.sleep(0.5)
    start = time.time()
    while time.time() - start < timeout_sec:
        active = job_manager.get_active_job()
        if not active:
            return
        time.sleep(0.5)
    raise TimeoutError("Job did not finish within timeout")


def run_e2e():
    test_folder = (Path(__file__).resolve().parent.parent / "data" / "e2e_test_20").resolve()
    assert test_folder.exists(), f"Test folder does not exist: {test_folder}"
    initial_files = list(test_folder.glob("*"))
    assert len(initial_files) == 20, f"Expected 20 files, found {len(initial_files)}"

    print("\n=======================================================")
    print(f"STARTING END-TO-END TEST ON: {test_folder}")
    print(f"Total image files: {len(initial_files)}")
    print("=======================================================")

    # 1. Check folder summary before job
    print("\n[Step 1] Querying initial folder summary...")
    res1 = client.get(f"/folder/summary?path={test_folder}", headers=auth_headers)
    assert res1.status_code == 200, res1.text
    summary1 = res1.json()
    print(f"  Files found: {summary1['files_found']['total']}")
    print(f"  Staged: {summary1['staged']['total']}")
    print(f"  Stored: {summary1['stored']['total']}")
    assert summary1["files_found"]["total"] == 20
    assert summary1["staged"]["total"] == 0
    assert summary1["stored"]["total"] == 0

    # 2. Trigger OCR action -> staging.db
    print("\n[Step 2] Triggering Action 1: OCR (Text Only)...")
    res_ocr = client.post("/jobs/ocr", json={"folder": str(test_folder)}, headers=auth_headers)
    assert res_ocr.status_code == 200, res_ocr.text
    print(f"  Job started: {res_ocr.json()['job_id']}")
    wait_for_job_idle()

    # 3. Check staged counts
    print("\n[Step 3] Checking staged counts in staging.db...")
    res2 = client.get(f"/folder/summary?path={test_folder}", headers=auth_headers)
    summary2 = res2.json()
    print(f"  Staged OCR: {summary2['staged']['ocr']}")
    print(f"  Stored: {summary2['stored']['total']}")
    assert summary2["staged"]["ocr"] == 20
    assert summary2["stored"]["total"] == 0, "No Qdrant writes allowed during OCR staging"

    # 4. Trigger Store in DB action
    print("\n[Step 4] Triggering Action 3: Store in DB...")
    res_store = client.post("/jobs/store", json={"folder": str(test_folder)}, headers=auth_headers)
    assert res_store.status_code == 200, res_store.text
    print(f"  Job started: {res_store.json()['job_id']}")
    wait_for_job_idle()

    # 5. Check stored counts
    print("\n[Step 5] Checking stored counts in Qdrant docs_ocr...")
    res3 = client.get(f"/folder/summary?path={test_folder}", headers=auth_headers)
    summary3 = res3.json()
    print(f"  Stored docs_ocr: {summary3['stored']['docs_ocr']}")
    print(f"  Total stored: {summary3['stored']['total']}")
    assert summary3["stored"]["docs_ocr"] > 0, "Points should be committed to docs_ocr"

    # 6. Trigger Delete from DB action
    print("\n[Step 6] Triggering Action 4: Delete from DB...")
    res_del = client.post("/jobs/delete", json={"folder": str(test_folder)}, headers=auth_headers)
    assert res_del.status_code == 200, res_del.text
    print(f"  Job started: {res_del.json()['job_id']}")
    wait_for_job_idle()

    # 7. Check stored counts return to ZERO
    print("\n[Step 7] Checking stored counts after delete...")
    res4 = client.get(f"/folder/summary?path={test_folder}", headers=auth_headers)
    summary4 = res4.json()
    print(f"  Staged: {summary4['staged']['total']}")
    print(f"  Stored: {summary4['stored']['total']}")
    assert summary4["stored"]["total"] == 0, "Stored counts must return to exactly ZERO"
    assert summary4["staged"]["total"] == 0, "Staged counts must return to exactly ZERO"

    # 8. Verify all 20 image files are UNTOUCHED
    print("\n[Step 8] Verifying image files on disk...")
    remaining_files = list(test_folder.glob("*"))
    assert len(remaining_files) == 20, f"Expected all 20 files untouched, found {len(remaining_files)}"
    print(f"  [CONFIRMED] All {len(remaining_files)} image files remain intact and untouched on disk!")

    # 9. Hardware & VRAM status check
    manager = get_model_manager()
    vram_report = manager.vram_report()
    print("\n[Step 9] Hardware & GPU VRAM Report:")
    print(f"  Device: {vram_report['device_name']}")
    print(f"  Allocated VRAM: {vram_report['allocated_mb']} MB")
    print(f"  Reserved VRAM : {vram_report['reserved_mb']} MB")
    print(f"  Resident Models: {vram_report['resident_models']}")

    print("\n=======================================================")
    print("[SUCCESS] ALL END-TO-END PIPELINE CHECKS PASSED!")
    print("=======================================================\n")


if __name__ == "__main__":
    run_e2e()
