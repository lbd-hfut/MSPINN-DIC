"""Run the fixed-budget P2/B-spline scale study reproducibly.

Each formal method runs in a fresh Python process with the configuration
locked to 500 Adam steps, zero seed-training steps and zero L-BFGS steps.
"""

import argparse
from pathlib import Path
import subprocess
import sys


CONFIGS = {
    "p2_fullfield": "bbf_4x1_p2_fullfield_500.txt",
    "bspline4_p2": "bbf_4x1_bspline4_p2_500.txt",
    "bspline6_p2": "bbf_4x1_bspline6_p2_500.txt",
    "bspline8_p2": "bbf_4x1_bspline8_p2_500.txt",
    "bspline6_cosine": "bbf_4x1_bspline6_cosine_500.txt",
    "bspline6_p2_frozen": "bbf_4x1_bspline6_p2_frozen_500.txt",
}


def run(command, root):
    print("+", " ".join(str(item) for item in command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "methods", nargs="*", metavar="METHOD", default=None,
        help="Methods to run; default: all six formal configurations.")
    parser.add_argument(
        "--skip-analysis", action="store_true",
        help="Skip unified checkpoint analysis and figure generation.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    methods = args.methods or list(CONFIGS)
    unknown = [method for method in methods if method not in CONFIGS]
    if unknown:
        parser.error(f"unknown methods: {', '.join(unknown)}")
    seed_config = root / "config" / "Seed_Configuration.txt"
    config_dir = root / "config" / "p2_bspline_scale"
    for method in methods:
        dic_config = config_dir / CONFIGS[method]
        code = (
            "from segpinndic.DIC_analysis import main; "
            f"main(r'{seed_config}', r'{dic_config}')"
        )
        run([sys.executable, "-c", code], root)

    if not args.skip_analysis:
        run([sys.executable,
             str(root / "scripts" / "analyze_p2_bspline_scale_phase.py")],
            root)


if __name__ == "__main__":
    main()
