import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from segpinndic.DIC_assemblies import (
    BicubicHermiteC1Assembly,
    C1MinimalCorrectionPOUAssembly,
    CosinePOUAssembly,
    Hermite1DC1Assembly,
    QuinticPOUAssembly,
)
from segpinndic.DIC_constants import Constants
from segpinndic.DIC_decompositions import RectangularDecompositionND
from segpinndic.DIC_domains import RectangularDomainND
from segpinndic.DIC_readImg import BufferManager
from segpinndic.DIC_seed_trainer import FBPINN_seed_loss
from segpinndic.DIC_trainers import PINN_model, get_inputs


jax.config.update("jax_enable_x64", True)


ASSEMBLIES = (
    CosinePOUAssembly,
    QuinticPOUAssembly,
    Hermite1DC1Assembly,
    BicubicHermiteC1Assembly,
    C1MinimalCorrectionPOUAssembly,
)


def _config(**overrides):
    values = dict(
        output_dir="unused",
        loss_fun="DIC_ZNSSD",
        znssd_kernel_size=7,
        n_subdomains=[4, 1],
        network="AdaptiveFCN",
        hidden_units=[8, 8],
        adam_epochs=1,
        train_schedulers="AllActiveSchedulerND",
        adam_lr=1e-3,
        dic_lr=1e-3,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class CompatibilityRegressionTests(unittest.TestCase):
    def tearDown(self):
        BufferManager.mask = None

    def test_missing_assembly_config_preserves_cosine_and_run_name(self):
        BufferManager.mask = np.ones((9, 13), dtype=bool)[None, ...]
        constants = Constants(_config(), roi_id=0)
        self.assertEqual(constants.assembly_method, "cosine_pou")
        self.assertIs(constants.assembly, CosinePOUAssembly)
        self.assertFalse(constants.assembly_explicit)
        self.assertEqual(constants.run, "AdaptiveFCN_All_4x1_DIC_ZNSSD")

    def test_multiple_rois_construct_independent_constants(self):
        masks = np.zeros((2, 12, 16), dtype=bool)
        masks[0, 1:7, 2:9] = True
        masks[1, 4:11, 7:15] = True
        BufferManager.mask = masks
        first = Constants(_config(assembly_method="quintic_pou"), roi_id=0)
        second = Constants(_config(assembly_method="quintic_pou"), roi_id=1)
        self.assertEqual(first.roi_bbox, (2, 8, 1, 6))
        self.assertEqual(second.roi_bbox, (7, 14, 4, 10))
        self.assertEqual(first.assembly_method, second.assembly_method)
        self.assertNotEqual(first.domain_init_kwargs["xmin"].tolist(),
                            second.domain_init_kwargs["xmin"].tolist())

    def test_single_pinn_forward_and_spatial_jacobian(self):
        domain_static, _ = RectangularDomainND.init_params(
            xmin=np.array([0.0, 0.0]), xmax=np.array([10.0, 8.0]))
        weights = jnp.array([[0.3, -0.2], [0.1, 0.5]])
        bias = jnp.array([0.7, -0.4])
        params = {
            "static": {"domain": domain_static},
            "trainable": {"network": {"subdomain": {
                "weights": weights, "bias": bias,
            }}},
        }

        def network_fn(all_params, x):
            local = all_params["trainable"]["network"]["subdomain"]
            return local["weights"] @ x + local["bias"]

        model_fns = (RectangularDomainND.norm_fn, network_fn, lambda u: u)
        points = jnp.array([[2.0, 3.0], [7.0, 6.0]])
        values = PINN_model(params, points, model_fns)[0]
        self.assertEqual(values.shape, (2, 2))
        jacobian = jax.jacfwd(
            lambda point: PINN_model(params, point[None, :], model_fns)[0][0]
        )(points[0])
        self.assertTrue(bool(jnp.all(jnp.isfinite(jacobian))))
        self.assertGreater(float(jnp.linalg.norm(jacobian)), 0.0)


class SeedAssemblyRegressionTests(unittest.TestCase):
    @staticmethod
    def _fixture(assembly):
        nodes_x = np.linspace(0.0, 40.0, 5)
        nodes_y = np.linspace(0.0, 20.0, 2)
        centers = [0.5*(nodes_x[:-1] + nodes_x[1:]),
                   0.5*(nodes_y[:-1] + nodes_y[1:])]
        widths = [1.6*np.diff(nodes_x), 1.6*np.diff(nodes_y)]
        decomp_static, _ = RectangularDecompositionND.init_params(
            centers, widths, (np.zeros(2), np.ones(2)))
        assembly_static, _ = assembly.init_params(bicubic_cell_size=4.0)
        coefficients = jnp.array([
            [[0.1, 0.02, -0.01], [-0.2, 0.01, 0.03]],
            [[0.4, -0.01, 0.02], [0.1, 0.03, -0.02]],
            [[-0.3, 0.04, 0.01], [0.2, -0.02, 0.01]],
            [[0.2, -0.03, 0.02], [-0.1, 0.02, 0.04]],
        ], dtype=jnp.float64)
        all_params = {
            "static": {
                "decomposition": decomp_static,
                "assembly": assembly_static,
            },
            "trainable": {"network": {"subdomain": {
                "coefficients": coefficients,
            }}},
        }

        def norm_fn(params, x):
            return x

        def network_fn(params, x):
            local = params["trainable"]["network"]["subdomain"]["coefficients"]
            return local[:, 0] + local[:, 1]*x[0] + local[:, 2]*x[1]

        model_fns = (
            norm_fn,
            network_fn,
            lambda params, value: value,
            RectangularDecompositionND.window_fn,
            assembly,
        )
        points = jnp.array([
            [4.0, 5.0], [12.0, 8.0], [19.0, 10.0],
            [27.0, 12.0], [36.0, 15.0],
        ], dtype=jnp.float64)
        plan, _, (_, cut_active, cut_fixed, cut_all, _) = get_inputs(
            points, jnp.ones(4, dtype=int), all_params,
            RectangularDecompositionND, assembly)
        return (
            cut_active(all_params["trainable"]),
            cut_fixed(all_params["trainable"]),
            cut_all(all_params["static"]),
            plan,
            points,
            model_fns,
        )

    def test_seed_loss_and_gradients_use_each_global_assembly(self):
        seed_target = jnp.zeros((5, 2), dtype=jnp.float64)
        for assembly in ASSEMBLIES:
            with self.subTest(assembly=assembly.__name__):
                active, fixed, static, plan, points, model_fns = self._fixture(assembly)
                loss, gradient = jax.value_and_grad(FBPINN_seed_loss)(
                    active, fixed, static, plan, points, model_fns, seed_target)
                leaves = jax.tree_util.tree_leaves(gradient)
                norm = sum(float(jnp.sum(jnp.abs(leaf))) for leaf in leaves)
                self.assertTrue(np.isfinite(float(loss)))
                self.assertGreater(float(loss), 0.0)
                self.assertTrue(all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves))
                self.assertGreater(norm, 0.0)


if __name__ == "__main__":
    unittest.main()
