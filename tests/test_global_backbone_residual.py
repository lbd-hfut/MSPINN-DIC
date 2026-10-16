import unittest

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import optax

from segpinndic.DIC_assemblies import GlobalBackboneLocalResidualAssembly
from segpinndic.DIC_decompositions import RectangularDecompositionND
from segpinndic.DIC_seed_trainer import FBPINN_seed_loss
from segpinndic.DIC_trainers import FBPINN_loss, get_inputs


jax.config.update("jax_enable_x64", True)


def _fixture():
    centers = [np.array([5.0, 15.0, 25.0, 35.0]), np.array([10.0])]
    widths = [np.full(4, 16.0), np.array([32.0])]
    decomposition_static, _ = RectangularDecompositionND.init_params(
        centers, widths, (np.zeros(2), np.ones(2)))
    assembly_static, _ = GlobalBackboneLocalResidualAssembly.init_params()
    local = jnp.array([
        [[0.01, 0.001, -0.002], [-0.02, 0.002, 0.001]],
        [[0.02, -0.001, 0.001], [0.01, 0.001, -0.002]],
        [[-0.01, 0.002, 0.001], [0.02, -0.001, 0.001]],
        [[0.01, -0.002, 0.002], [-0.01, 0.002, -0.001]],
    ], dtype=jnp.float64)
    global_coefficients = jnp.array(
        [[0.4, 0.03, -0.01], [-0.2, 0.015, 0.025]], dtype=jnp.float64)
    params = {
        "static": {
            "assembly": assembly_static,
            "decomposition": decomposition_static,
            "domain": {
                "xmin": jnp.array([0.0, 0.0]),
                "xmax": jnp.array([40.0, 20.0]),
            },
        },
        "trainable": {
            "network": {"subdomain": {"coefficients": local}},
            "global_network": {"coefficients": global_coefficients},
        },
    }

    def norm_fn(params, point):
        return point

    def network_fn(params, point):
        coefficients = params["trainable"]["network"]["subdomain"]["coefficients"]
        return coefficients[:, 0] + coefficients[:, 1]*point[0] + coefficients[:, 2]*point[1]

    local_fns = (
        norm_fn, network_fn, lambda params, value: value,
        RectangularDecompositionND.window_fn)
    return params, local_fns


class GlobalBackboneResidualTests(unittest.TestCase):
    def test_total_is_global_plus_cosine_residual(self):
        params, local_fns = _fixture()
        points = jnp.array([
            [3.0, 4.0], [9.0, 6.0], [12.0, 8.0],
            [21.0, 10.0], [31.0, 12.0], [37.0, 14.0],
        ], dtype=jnp.float64)
        plan, _, (_, cut_active, _, cut_all, _) = get_inputs(
            points, jnp.zeros(4, dtype=int), params,
            RectangularDecompositionND, GlobalBackboneLocalResidualAssembly)
        cut_params = {
            "static": cut_all(params["static"]),
            "trainable": cut_active(params["trainable"]),
        }
        global_value, residual, *_ = GlobalBackboneLocalResidualAssembly.components(
            cut_params, points, plan, local_fns)
        total = GlobalBackboneLocalResidualAssembly.assemble(
            cut_params, points, plan, local_fns)[0]
        np.testing.assert_allclose(total, global_value+residual, atol=1e-13)

        total_jacobian = jax.jacfwd(
            lambda current: GlobalBackboneLocalResidualAssembly.assemble(
                cut_params, current, plan, local_fns)[0])(points)
        component_jacobian = jax.jacfwd(
            lambda current: sum(
                GlobalBackboneLocalResidualAssembly.components(
                    cut_params, current, plan, local_fns)[:2]))(points)
        np.testing.assert_allclose(total_jacobian, component_jacobian, atol=1e-12)

    def test_gradients_reach_global_and_local_parameters(self):
        params, local_fns = _fixture()
        points = jnp.array([[9.0, 6.0], [10.0, 8.0], [11.0, 10.0]],
                           dtype=jnp.float64)
        plan, _, (_, cut_active, _, cut_all, _) = get_inputs(
            points, jnp.zeros(4, dtype=int), params,
            RectangularDecompositionND, GlobalBackboneLocalResidualAssembly)
        cut_params = {
            "static": cut_all(params["static"]),
            "trainable": cut_active(params["trainable"]),
        }

        def loss(trainable):
            current = {"static": cut_params["static"], "trainable": trainable}
            value = GlobalBackboneLocalResidualAssembly.assemble(
                current, points, plan, local_fns)[0]
            return jnp.sum(value**2)

        gradient = jax.grad(loss)(cut_params["trainable"])
        local_norm = jnp.linalg.norm(
            gradient["network"]["subdomain"]["coefficients"])
        global_norm = jnp.linalg.norm(
            gradient["global_network"]["coefficients"])
        self.assertGreater(float(local_norm), 0.0)
        self.assertGreater(float(global_norm), 0.0)

    def test_seed_adam_and_lbfgs_paths_use_global_and_local_parameters(self):
        params, local_fns = _fixture()
        assembly = GlobalBackboneLocalResidualAssembly
        model_fns = local_fns+(assembly,)
        points = jnp.array([
            [3.0, 4.0], [9.0, 6.0], [12.0, 8.0],
            [21.0, 10.0], [31.0, 12.0], [37.0, 14.0],
        ], dtype=jnp.float64)
        plan, _, (_, cut_active, cut_fixed, cut_all, _) = get_inputs(
            points, jnp.ones(4, dtype=int), params,
            RectangularDecompositionND, assembly)
        active = cut_active(params["trainable"])
        fixed = cut_fixed(params["trainable"])
        static = cut_all(params["static"])
        target = jnp.zeros((len(points), 2), dtype=points.dtype)

        seed_loss, seed_gradient = jax.value_and_grad(FBPINN_seed_loss)(
            active, fixed, static, plan, points, model_fns, target)
        self.assertTrue(np.isfinite(float(seed_loss)))
        self.assertGreater(float(jnp.linalg.norm(
            seed_gradient["network"]["subdomain"]["coefficients"])), 0.0)
        self.assertGreater(float(jnp.linalg.norm(
            seed_gradient["global_network"]["coefficients"])), 0.0)

        loss_fn = lambda _params, _points, value: jnp.mean((value-target)**2)
        adam_loss, adam_gradient = jax.value_and_grad(FBPINN_loss)(
            active, fixed, static, plan, points, model_fns, loss_fn)
        optimiser = optax.adam(1e-3)
        updates, _ = optimiser.update(
            adam_gradient, optimiser.init(active), active)
        adam_params = optax.apply_updates(active, updates)
        self.assertTrue(np.isfinite(float(adam_loss)))
        self.assertFalse(np.allclose(
            np.asarray(adam_params["global_network"]["coefficients"]),
            np.asarray(active["global_network"]["coefficients"])))

        solver = jaxopt.LBFGS(
            fun=lambda current: FBPINN_loss(
                current, fixed, static, plan, points, model_fns, loss_fn),
            maxiter=1, tol=0.0, implicit_diff=False)
        state = solver.init_state(adam_params)
        lbfgs_params, state = solver.update(adam_params, state)
        self.assertTrue(np.isfinite(float(state.value)))
        self.assertFalse(np.allclose(
            np.asarray(lbfgs_params["global_network"]["coefficients"]),
            np.asarray(adam_params["global_network"]["coefficients"])))


if __name__ == "__main__":
    unittest.main()
