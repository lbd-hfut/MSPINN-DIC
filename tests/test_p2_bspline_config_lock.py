"""Lock every formal scale-study configuration to the fair 500-step route."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = tuple(sorted((ROOT / "config" / "p2_bspline_scale").glob("*.txt")))


def parse(path):
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return {
        line[1:-1].strip(): lines[index+1]
        for index, line in enumerate(lines[:-1])
        if line.startswith("#") and line.endswith(":")
    }


class P2BSplineConfigLockTests(unittest.TestCase):
    def test_six_formal_configs_share_budget_geometry_and_optimizer(self):
        self.assertEqual(len(CONFIGS), 6)
        for path in CONFIGS:
            with self.subTest(config=path.name):
                values = parse(path)
                self.assertEqual(values["n_subdomains"], "[4,1]")
                self.assertEqual(values["adam_epochs"], "500")
                self.assertEqual(values["seed_train_epochs"], "0")
                self.assertEqual(values["lbfgs_epochs"], "0")
                self.assertEqual(values["adam_lr"], "0.005")
                self.assertEqual(values["dic_lr"], "0.001")
                self.assertEqual(values["hidden_units"], "[32,32,32,32]")


if __name__ == "__main__":
    unittest.main()
