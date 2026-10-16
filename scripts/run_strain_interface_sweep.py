"""Run the five Phase-3 methods on one fair 3000-step Adam trajectory each.

Budget results at 500/1000/2000/3000 are read from checkpoints on that same
trajectory.  Every method runs in a fresh Python process and also traverses
the identical baseline path: SIFT-based scaling, no seed-training epochs, and
no final L-BFGS epochs.  Seed-training and L-BFGS are exercised separately by
the regression/path-smoke checks so they do not contaminate this Adam budget.
"""

import argparse
from pathlib import Path
import subprocess
import sys


CONFIGS = {
    "cosine_pou": "bbf_4x1_cosine_5000.txt",
    "quintic_pou": "bbf_4x1_quintic_5000.txt",
    "hermite_1d_c1": "bbf_4x1_hermite1d_5000.txt",
    "c1_minimal_correction_pou": "bbf_4x1_minimal_correction_5000.txt",
    "global_backbone_local_residual": "bbf_4x1_global_residual_5000.txt",
}


def run(command, root):
    print("+", " ".join(str(item) for item in command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "methods", nargs="*", metavar="METHOD",
        default=None, help="Methods to run; default: all five.")
    parser.add_argument(
        "--skip-analysis", action="store_true",
        help="Skip checkpoint evaluation and comparison figures.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    seed_config = root / "config" / "Seed_Configuration.txt"
    config_dir = root / "config" / "strain_interface_phase"
    methods = args.methods or list(CONFIGS)
    unknown = [method for method in methods if method not in CONFIGS]
    if unknown:
        parser.error(f"unknown methods: {', '.join(unknown)}")
    for method in methods:
        dic_config = config_dir / CONFIGS[method]
        code = (
            "from segpinndic.DIC_analysis import main; "
            f"main(r'{seed_config}', r'{dic_config}')"
        )
        run([sys.executable, "-c", code], root)

    if not args.skip_analysis:
        run([
            sys.executable,
            str(root / "scripts" / "analyze_strain_interface_sweep.py"),
        ], root)


if __name__ == "__main__":
    main()
