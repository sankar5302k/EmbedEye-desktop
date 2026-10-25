"""Unit test for strict GPU memory manager (ModelManager).

Tests:
1. Records initial VRAM baseline.
2. Loads and unloads 'clip' twice.
3. Loads and unloads 'florence' twice.
4. Loads and unloads 'reranker' twice.
5. Loads and unloads coexisting ('clip', 'reranker') twice.
6. Asserts allocated VRAM returns to within 50MB of baseline after each unload.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Set cache environment before deep learning libraries
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.config as cfg
from src.gpu_manager import get_model_manager

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_gpu_manager")


def run_vram_test() -> bool:
    print("=" * 70)
    print("UNIT TEST: GPU ModelManager Load/Unload & VRAM Recovery")
    print(f"Device: {cfg.get_device()}")
    print("=" * 70)

    manager = get_model_manager()
    initial_report = manager.vram_report()
    baseline_allocated = initial_report["allocated_mb"]
    print(f"Initial Status: {initial_report['status_text']}")
    print(f"Baseline Allocated VRAM: {baseline_allocated:.2f} MB\n")

    models_to_test = ["clip", "florence", "reranker"]
    all_passed = True
    margin_mb = 50.0

    for model_name in models_to_test:
        for iteration in (1, 2):
            print(f"--> [{model_name.upper()}] Iteration {iteration}/2: Loading...")
            with manager.use(model_name) as bundle:
                active_report = manager.vram_report()
                print(
                    f"    Active: {model_name} | Allocated: {active_report['allocated_mb']:.1f} MB | "
                    f"Reserved: {active_report['reserved_mb']:.1f} MB | Resident: {active_report['resident_models']}"
                )
                assert bundle is not None, f"Bundle for {model_name} should not be None"

            after_report = manager.vram_report()
            freed_allocated = after_report["allocated_mb"]
            diff = freed_allocated - baseline_allocated
            print(
                f"    After Unload: Allocated: {freed_allocated:.1f} MB (Diff from baseline: {diff:+.1f} MB)"
            )

            if diff > margin_mb:
                print(f"    [FAIL] VRAM diff ({diff:.1f} MB) exceeds allowed margin ({margin_mb} MB)!")
                all_passed = False
            else:
                print(f"    [PASS] Within {margin_mb} MB of baseline.")
            print()


    # Test Search Coexistence (CLIP + Reranker)
    for iteration in (1, 2):
        print(f"--> [SEARCH COEXISTENCE (clip + reranker)] Iteration {iteration}/2: Loading...")
        with manager.use_search() as (clip_bundle, reranker_bundle):
            active_report = manager.vram_report()
            print(
                f"    Active Search: Allocated: {active_report['allocated_mb']:.1f} MB | "
                f"Reserved: {active_report['reserved_mb']:.1f} MB | Resident: {active_report['resident_models']}"
            )
            assert "clip" in active_report["resident_models"]
            assert "reranker" in active_report["resident_models"]

        after_report = manager.vram_report()
        freed_allocated = after_report["allocated_mb"]
        diff = freed_allocated - baseline_allocated
        print(
            f"    After Search Unload: Allocated: {freed_allocated:.1f} MB (Diff from baseline: {diff:+.1f} MB)"
        )
        if diff > margin_mb:
            print(f"    [FAIL] Search VRAM diff ({diff:.1f} MB) exceeds allowed margin ({margin_mb} MB)!")
            all_passed = False
        else:
            print(f"    [PASS] Search coexisting models cleanly unloaded within {margin_mb} MB.")
        print()

    # Test Florence isolation: Florence MUST unload any resident model
    print("--> [ISOLATION CHECK] Florence-2 must unload coexisting models...")
    manager.load("clip", allow_coexist=True)
    manager.load("reranker", allow_coexist=True)
    report_before_florence = manager.vram_report()
    print(f"    Resident before Florence: {report_before_florence['resident_models']}")

    with manager.use("florence"):
        report_florence = manager.vram_report()
        print(f"    Resident during Florence: {report_florence['resident_models']}")
        if "clip" in report_florence["resident_models"] or "reranker" in report_florence["resident_models"]:
            print("    [FAIL] Florence failed to evict resident models!")
            all_passed = False
        else:
            print("    [PASS] Florence is the ONLY resident model on GPU.")

    after_all = manager.vram_report()
    diff = after_all["allocated_mb"] - baseline_allocated
    print(f"\nFinal State: Allocated: {after_all['allocated_mb']:.1f} MB (Diff: {diff:+.1f} MB)")
    if diff <= margin_mb:
        print("[SUCCESS] ALL VRAM TESTS PASSED SUCCESSFULLY!")
    else:
        print("[FAILURE] VRAM TESTS FAILED.")
        all_passed = False

    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_vram_test()
    sys.exit(0 if success else 1)
