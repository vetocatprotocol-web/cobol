import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.benchmark_sweep import run_sweep


def test_500x_target_on_synthetic():
    results = run_sweep()
    # results is list of tuples; ensure best > 500x
    best = max(results, key=lambda r: r[3])
    assert best[3] >= 500, f"Best ratio {best[3]:.2f}x < 500x"
