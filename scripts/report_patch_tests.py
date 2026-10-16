"""Write quantitative P1/P2 assembly patch-test metrics."""

import json
from pathlib import Path

import numpy as np

from tests.test_assembly_patch import METHODS, patch_metrics


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "case" / "bbf" / "1" / "results" / "assembly_comparison"


def main():
    p1 = np.array([
        [0.3, 0.02, -0.015, 0.0, 0.0, 0.0],
        [-0.4, 0.01, 0.025, 0.0, 0.0, 0.0],
    ])
    p2 = np.array([
        [0.2, 0.01, -0.008, 2e-4, -1.5e-4, 1e-4],
        [-0.1, 0.006, 0.012, -1e-4, 2.5e-4, -1.2e-4],
    ])
    report = {
        "P1_affine": {name: patch_metrics(p1, assembly)
                      for name, assembly in METHODS.items()},
        "P2_curved_quadratic": {name: patch_metrics(p2, assembly)
                                for name, assembly in METHODS.items()},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "patch_test_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
