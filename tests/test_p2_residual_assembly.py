import unittest

import jax
import jax.numpy as jnp
import numpy as np

from segpinndic.DIC_assemblies import BackboneFirstP2ResidualAssembly
from segpinndic.DIC_decompositions import RectangularDecompositionND
from segpinndic.DIC_trainers import get_inputs


jax.config.update("jax_enable_x64", True)


def _fixture(coefficients):
    centers = [np.array([5.0, 15.0, 25.0, 35.0]), np.array([10.0])]
    widths = [np.full(4, 16.0), np.array([32.0])]
    decomposition_static, _ = RectangularDecompositionND.init_params(
        centers, widths, (np.zeros(2), np.ones(2)))
    assembly_static, _ = BackboneFirstP2ResidualAssembly.init_params(
        residual_gate=1.0)
    local = jnp.tile(jnp.asarray(coefficients)[None, :, :], (4, 1, 1))
    params = {
        "static": {
            "assembly": assembly_static,
            "decomposition": decomposition_static,
            "domain": {"xmin": jnp.array([0.0, 0.0]),
                       "xmax": jnp.array([40.0, 20.0])},
        },
        "trainable": {
            "network": {"subdomain": {"coefficients": local}},
            "global_network": {"coefficients": jnp.zeros((2, 3))},
        },
    }

    def norm_fn(_params, point):
        return point

    def network_fn(params_, point):
        c = params_["trainable"]["network"]["subdomain"]["coefficients"]
        return c[:, 0] + c[:, 1]*point[0] + c[:, 2]*point[0]**2

    return params, (norm_fn, network_fn, lambda _p, value: value,
                    RectangularDecompositionND.window_fn)


class P2ResidualAssemblyTests(unittest.TestCase):
    def _value(self, coefficients):
        params, local_fns = _fixture(coefficients)
        points = jnp.array([[3., 4.], [9., 7.], [18., 8.], [31., 11.]],
                           dtype=jnp.float64)
        plan, _, (_, cut_active, _, cut_all, _) = get_inputs(
            points, jnp.zeros(4, dtype=int), params,
            RectangularDecompositionND, BackboneFirstP2ResidualAssembly)
        cut = {"static": cut_all(params["static"]),
               "trainable": cut_active(params["trainable"])}
        total = BackboneFirstP2ResidualAssembly.assemble(
            cut, points, plan, local_fns)[0]
        return total, cut, points, plan, local_fns

    def test_p0_p1_p2_value_and_x_derivative_reproduction(self):
        for coefficients in (
                [[.2, 0., 0.], [-.1, 0., 0.]],
                [[.2, .03, 0.], [-.1, -.02, 0.]],
                [[.2, .03, .001], [-.1, -.02, -.002]],
        ):
            with self.subTest(coefficients=coefficients):
                total, cut, points, plan, local_fns = self._value(coefficients)
                c = jnp.asarray(coefficients)
                expected = c[:, 0] + c[:, 1]*points[:, 0, None] + c[:, 2]*points[:, 0, None]**2
                np.testing.assert_allclose(total, expected, atol=2e-10)
                tangent = jnp.tile(jnp.array([[1., 0.]]), (len(points), 1))
                _, dx = jax.jvp(lambda q: BackboneFirstP2ResidualAssembly.assemble(
                    cut, q, plan, local_fns)[0], (points,), (tangent,))
                expected_dx = c[:, 1] + 2*c[:, 2]*points[:, 0, None]
                np.testing.assert_allclose(dx, expected_dx, atol=2e-9)

    def test_gate_zero_removes_local_forward_and_gradient(self):
        coefficients = [[.2, .03, .001], [-.1, -.02, -.002]]
        total, cut, points, plan, local_fns = self._value(coefficients)
        cut["static"]["assembly"]["residual_gate"] = jnp.array(0.)
        value = BackboneFirstP2ResidualAssembly.assemble(cut, points, plan, local_fns)[0]
        np.testing.assert_allclose(value, 0.0, atol=1e-12)

        def loss(trainable):
            return jnp.sum(BackboneFirstP2ResidualAssembly.assemble(
                {"static": cut["static"], "trainable": trainable},
                points, plan, local_fns)[0]**2)
        gradient = jax.grad(loss)(cut["trainable"])
        self.assertEqual(float(jnp.linalg.norm(
            gradient["network"]["subdomain"]["coefficients"])), 0.0)

    def test_p2_residual_gradient_path_reaches_local_parameters(self):
        total, cut, points, plan, local_fns = self._value(
            [[.2, .03, .001], [-.1, -.02, -.002]])

        def loss(trainable):
            return jnp.sum(BackboneFirstP2ResidualAssembly.assemble(
                {"static": cut["static"], "trainable": trainable},
                points, plan, local_fns)[0]**2)
        gradient = jax.grad(loss)(cut["trainable"])
        self.assertGreater(float(jnp.linalg.norm(
            gradient["network"]["subdomain"]["coefficients"])), 0.0)


if __name__ == "__main__":
    unittest.main()
