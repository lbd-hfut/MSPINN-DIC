"""Run the fixed-budget 8x8 projected-residual experiments."""

import argparse
from pathlib import Path
import subprocess
import sys


CONFIGS = {
    "none": "bbf_4x1_bspline8_cosine_none_500.txt",
    "p1": "bbf_4x1_bspline8_cosine_p1proj_500.txt",
    "p2": "bbf_4x1_bspline8_cosine_p2proj_500.txt",
}


def run(command, root):
    print("+", " ".join(str(item) for item in command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("methods", nargs="*", metavar="METHOD", default=None)
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args()
    methods = args.methods or list(CONFIGS)
    unknown = [method for method in methods if method not in CONFIGS]
    if unknown:
        parser.error(f"unknown methods: {', '.join(unknown)}")

    root = Path(__file__).resolve().parents[1]
    seed = root/"config"/"Seed_Configuration.txt"
    config_dir = root/"config"/"projected_residual"
    for method in methods:
        config = config_dir/CONFIGS[method]
        code = (
            "from segpinndic.DIC_analysis import main; "
            f"main(r'{seed}', r'{config}')")
        run([sys.executable, "-c", code], root)

    analysis = root/"scripts"/"analyze_projected_residual_phase.py"
    if not args.skip_analysis and analysis.exists():
        run([sys.executable, str(analysis)], root)


if __name__ == "__main__":
    main()
