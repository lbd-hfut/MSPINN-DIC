import unittest

import jax
import jax.numpy as jnp
import numpy as np

from segpinndic.DIC_assemblies import BackboneFirstCosineResidualAssembly
from segpinndic.DIC_constants import Constants


class BackboneFirstScheduleTests(unittest.TestCase):
    def _constants(self, mode):
        item = object.__new__(Constants)
        item.assembly = BackboneFirstCosineResidualAssembly
        item.n_steps = 500
        item.backbone_stage1_steps = 150
        item.backbone_stage2_steps = 150
        item.backbone_stage2_gate = 0.25
        item.backbone_residual_schedule = mode
        return item

    def test_progressive_gate_has_three_requested_stages(self):
        c = self._constants("progressive")
        self.assertEqual(c.residual_gate_at_step(0), 0.0)
        self.assertEqual(c.residual_gate_at_step(150), 0.0)
        self.assertAlmostEqual(c.residual_gate_at_step(225), 0.125)
        self.assertAlmostEqual(c.residual_gate_at_step(300), 0.25)
        self.assertAlmostEqual(c.residual_gate_at_step(500), 1.0)

    def test_direct_backbone_first_releases_after_stage_one(self):
        c = self._constants("direct")
        self.assertEqual(c.residual_gate_at_step(150), 0.0)
        self.assertEqual(c.residual_gate_at_step(151), 1.0)

    def test_residual_initializer_zeros_output_map(self):
        layers = [
            (jnp.ones((3, 2)), jnp.ones(3), jnp.ones(3)),
            (jnp.ones((2, 3)), jnp.ones(2), jnp.ones(2)),
        ]
        output = BackboneFirstCosineResidualAssembly.initialize_local_residual_params(
            {"subdomain": {"layers": layers}})
        final = output["subdomain"]["layers"][-1]
        np.testing.assert_allclose(final[0], 0.0)
        np.testing.assert_allclose(final[1], 0.0)
        np.testing.assert_allclose(final[2], 1.0)


if __name__ == "__main__":
    unittest.main()
