import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax

from segpinndic.DIC_assemblies import BSplineP2ResidualAssembly
from segpinndic.DIC_constants import Constants
from segpinndic.DIC_readImg import BufferManager
from segpinndic.DIC_trainers import _restore_frozen_global_field
from tests.bspline_residual_fixture import fixture


def config(**overrides):
    values = dict(
        output_dir="unused", loss_fun="DIC_ZNSSD", znssd_kernel_size=7,
        n_subdomains=[4, 1], network="AdaptiveFCN", hidden_units=[8, 8],
        adam_epochs=500, train_schedulers="AllActiveSchedulerND",
        adam_lr=1e-3, dic_lr=1e-3,
        assembly_method="bspline_p2_residual", field_mode="bspline_residual",
        global_field_type="bspline", bspline_grid=[6, 6], bspline_degree=3,
        backbone_stage1_steps=200, backbone_stage2_steps=0,
        backbone_residual_schedule="direct", freeze_backbone_stage2=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class BSplineBackboneScheduleTests(unittest.TestCase):
    def tearDown(self):
        BufferManager.mask = None

    def test_residual_activates_exactly_at_step_200(self):
        BufferManager.mask = np.ones((20, 40), dtype=bool)[None, ...]
        constants = Constants(config(), roi_id=0)
        self.assertEqual(constants.residual_gate_at_step(0), 0.0)
        self.assertEqual(constants.residual_gate_at_step(199), 0.0)
        self.assertEqual(constants.residual_gate_at_step(200), 1.0)
        self.assertEqual(constants.residual_gate_at_step(500), 1.0)

    def test_frozen_stage_two_has_zero_control_gradient(self):
        params, points, plan, fns = fixture(
            BSplineP2ResidualAssembly, freeze=True, gate=1.0)

        def loss(trainable):
            return jnp.sum(BSplineP2ResidualAssembly.assemble(
                {"static": params["static"], "trainable": trainable},
                points, plan, fns)[0]**2)

        gradient = jax.grad(loss)(params["trainable"])
        self.assertEqual(float(jnp.linalg.norm(
            gradient["global_field"]["controls"])), 0.0)
        self.assertGreater(float(jnp.linalg.norm(
            gradient["network"]["subdomain"]["coefficients"])), 0.0)

    def test_frozen_stage_two_cancels_adam_momentum_drift(self):
        params = {
            "global_field": {"controls": jnp.ones((2, 2, 2))},
            "network": {"weights": jnp.ones((2,))},
        }
        optimiser = optax.adam(1e-2)
        state = optimiser.init(params)
        stage1_grad = jax.tree_util.tree_map(jnp.ones_like, params)
        updates, state = optimiser.update(stage1_grad, state, params)
        params = optax.apply_updates(params, updates)

        stage2_grad = {
            "global_field": {"controls": jnp.zeros((2, 2, 2))},
            "network": {"weights": jnp.ones((2,))},
        }
        previous = params
        updates, _ = optimiser.update(stage2_grad, state, params)
        moved = optax.apply_updates(params, updates)
        self.assertGreater(float(jnp.linalg.norm(
            moved["global_field"]["controls"]-
            previous["global_field"]["controls"])), 0.0)

        restored = _restore_frozen_global_field(previous, moved)
        np.testing.assert_array_equal(
            restored["global_field"]["controls"],
            previous["global_field"]["controls"])
        self.assertGreater(float(jnp.linalg.norm(
            restored["network"]["weights"]-
            previous["network"]["weights"])), 0.0)


if __name__ == "__main__":
    unittest.main()
