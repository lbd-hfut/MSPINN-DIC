"""Formal backbone/P2 comparisons are deliberately locked to Adam 500."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = tuple((ROOT / "config" / "backbone_p2").glob("*.txt"))


def parse(path):
    values = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("#") or ":" not in line:
            continue
        key = line[1:].strip().rstrip(":")
        # The project's legacy configuration format stores a key on one
        # comment line and the value on the following line.
        if key and not key.startswith("#"):
            values["_pending"] = key
        continue
    # Values are parsed in a second pass to keep the production format intact.
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    for index, line in enumerate(lines[:-1]):
        if line.startswith("#") and line.endswith(":"):
            values[line[1:-1].strip()] = lines[index + 1]
    return values


class BackboneP2ConfigLockTests(unittest.TestCase):
    def test_all_formal_configs_use_the_same_500_step_budget(self):
        self.assertEqual(len(CONFIGS), 4)
        for path in CONFIGS:
            with self.subTest(config=path.name):
                values = parse(path)
                self.assertEqual(values["adam_epochs"], "500")
                self.assertEqual(values["lbfgs_epochs"], "0")
                self.assertEqual(values["seed_train_epochs"], "0")
                self.assertEqual(values["train_schedulers"], "AllActiveSchedulerND")


if __name__ == "__main__":
    unittest.main()
