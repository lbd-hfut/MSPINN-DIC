import unittest
from pathlib import Path

from segpinndic.DIC_config import DIC_config_txt


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "none": "bbf_4x1_bspline8_cosine_none_500.txt",
    "p1": "bbf_4x1_bspline8_cosine_p1proj_500.txt",
    "p2": "bbf_4x1_bspline8_cosine_p2proj_500.txt",
}


class ProjectedResidualConfigLockTests(unittest.TestCase):
    def test_three_configs_share_formal_budget_geometry_and_optimizer(self):
        parsed = {
            name: DIC_config_txt(
                ROOT/"config"/"projected_residual"/filename, verbose=False)
            for name, filename in CONFIGS.items()}
        for name, config in parsed.items():
            with self.subTest(name=name):
                self.assertEqual(config.input_dir, "case/bbf/1")
                self.assertEqual(config.n_subdomains, [4, 1])
                self.assertEqual(config.hidden_units, [32, 32, 32, 32])
                self.assertEqual(config.network, "AdaptiveFCN")
                self.assertEqual(config.adam_epochs, 500)
                self.assertEqual(config.seed_train_epochs, 0)
                self.assertEqual(config.lbfgs_epochs, 0)
                self.assertEqual(config.adam_lr, .005)
                self.assertEqual(config.dic_lr, .001)
                self.assertEqual(config.bspline_grid, [8, 8])
                self.assertEqual(config.bspline_degree, 3)
                self.assertEqual(config.backbone_stage1_steps, 200)
                self.assertEqual(config.backbone_residual_schedule, "direct")
                self.assertEqual(config.projection_weighting, "uniform")
                self.assertEqual(config.projection_point_count, 256)
                self.assertEqual(config.stage_snapshot_steps, [0, 200, 500])
        self.assertEqual(parsed["none"].local_projection_order, "none")
        self.assertEqual(parsed["p1"].local_projection_order, 1)
        self.assertEqual(parsed["p2"].local_projection_order, 2)


if __name__ == "__main__":
    unittest.main()
