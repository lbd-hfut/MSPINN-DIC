"""Run the four isolated bbf/1 [4,1] assembly experiments reproducibly.

Each method runs in a fresh Python process so JAX compilation caches and
allocator state from one assembly do not contaminate the next measurement.
"""

import argparse
from pathlib import Path
import subprocess
import sys


CONFIGS = {
    "cosine_pou": "bbf_4x1_cosine.txt",
    "quintic_pou": "bbf_4x1_quintic.txt",
    "hermite_1d_c1": "bbf_4x1_hermite1d.txt",
    "bicubic_hermite_c1": "bbf_4x1_bicubic.txt",
}


def run(command, root):
    print("+", " ".join(str(item) for item in command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "methods", nargs="*", choices=tuple(CONFIGS),
        default=list(CONFIGS), help="Methods to run; default: all four.")
    parser.add_argument(
        "--skip-analysis", action="store_true",
        help="Do not regenerate patch and real-case comparison metrics.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    seed_config = root / "config" / "Seed_Configuration.txt"
    for method in args.methods:
        dic_config = root / "config" / "assembly" / CONFIGS[method]
        code = (
            "from segpinndic.DIC_analysis import main; "
            f"main(r'{seed_config}', r'{dic_config}')"
        )
        run([sys.executable, "-c", code], root)

    if not args.skip_analysis:
        run([sys.executable, str(root / "scripts" / "report_patch_tests.py")], root)
        run([sys.executable, str(root / "scripts" / "analyze_assembly_results.py")], root)


if __name__ == "__main__":
    main()
