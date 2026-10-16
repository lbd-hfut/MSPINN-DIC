import unittest

import jax
import jax.numpy as jnp
import numpy as np

from segpinndic.DIC_assemblies import (
    C1MinimalCorrectionPOUAssembly,
    CosinePOUAssembly,
)
from segpinndic.DIC_decompositions import RectangularDecompositionND
from segpinndic.DIC_trainers import FBPINN_model, get_inputs


jax.config.update("jax_enable_x64", True)


def _fixture(coefficients, support_ratio=0.5):
    centers = [np.array([5.0, 15.0, 25.0, 35.0]), np.array([10.0])]
    widths = [np.full(4, 16.0), np.array([32.0])]
    decomposition_static, _ = RectangularDecompositionND.init_params(
        centers, widths, (np.zeros(2), np.ones(2)))
    assembly_static, _ = C1MinimalCorrectionPOUAssembly.init_params(
        correction_support_ratio=support_ratio)
    all_params = {
        "static": {
            "decomposition": decomposition_static,
            "assembly": assembly_static,
        },
        "trainable": {"network": {"subdomain": {
            "coefficients": jnp.asarray(coefficients, dtype=jnp.float64),
        }}},
    }

    def norm_fn(params, point):
        return point

    def network_fn(params, point):
        local = params["trainable"]["network"]["subdomain"]["coefficients"]
        return local[:, 0] + local[:, 1]*point[0] + local[:, 2]*point[1]

    local_fns = (
        norm_fn, network_fn, lambda params, value: value,
        RectangularDecompositionND.window_fn)
    return all_params, local_fns


def _cut_and_plan(all_params, points, assembly):
    plan, _, (_, cut_active, _, cut_all, _) = get_inputs(
        points, jnp.zeros(4, dtype=int), all_params,
        RectangularDecompositionND, assembly)
    return {
        "static": cut_all(all_params["static"]),
        "trainable": cut_active(all_params["trainable"]),
    }, plan


class MinimalCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.coefficients = np.array([
            [[0.10, 0.020, -0.010], [-0.20, 0.010, 0.030]],
            [[0.45, -0.012, 0.018], [0.15, 0.028, -0.020]],
            [[-0.30, 0.035, 0.012], [0.20, -0.018, 0.014]],
            [[0.20, -0.025, 0.021], [-0.10, 0.022, 0.037]],
        ])

    def test_zero_mismatch_is_exact_cosine_identity(self):
        common = np.array([[0.2, 0.03, -0.01], [-0.4, 0.02, 0.015]])
        params, local_fns = _fixture(np.tile(common[None, :, :], (4, 1, 1)))
        points = jnp.array([
            [4.0, 3.0], [8.5, 6.0], [10.0, 8.0], [11.5, 10.0],
            [20.0, 12.0], [30.0, 14.0], [36.0, 16.0],
        ], dtype=jnp.float64)
        corrected_params, corrected_plan = _cut_and_plan(
            params, points, C1MinimalCorrectionPOUAssembly)
        cosine_params, cosine_plan = _cut_and_plan(
            params, points, CosinePOUAssembly)
        corrected = C1MinimalCorrectionPOUAssembly.assemble(
            corrected_params, points, corrected_plan, local_fns)
        cosine = CosinePOUAssembly.assemble(
            cosine_params, points, cosine_plan, local_fns)
        np.testing.assert_allclose(corrected[0], cosine[0], atol=2e-13)
        np.testing.assert_allclose(corrected[2], 0.0, atol=2e-13)

    def test_interface_matches_compatible_value_and_normal_derivative(self):
        params, local_fns = _fixture(self.coefficients)
        point = jnp.array([[10.0, 7.0]], dtype=jnp.float64)
        cut_params, plan = _cut_and_plan(
            params, point, C1MinimalCorrectionPOUAssembly)

        def field(current):
            return C1MinimalCorrectionPOUAssembly.assemble(
                cut_params, current[None, :], plan, local_fns)[0][0]

        value = field(point[0])
        derivative = jax.jacfwd(field)(point[0])[:, 0]
        left, right = map(jnp.asarray, self.coefficients[:2])
        expected_value = 0.5*(
            left[:, 0] + left[:, 1]*point[0, 0] + left[:, 2]*point[0, 1]
            + right[:, 0] + right[:, 1]*point[0, 0] + right[:, 2]*point[0, 1])
        expected_derivative = 0.5*(left[:, 1] + right[:, 1])
        np.testing.assert_allclose(value, expected_value, atol=2e-12)
        np.testing.assert_allclose(derivative, expected_derivative, atol=2e-12)

    def test_outer_support_has_zero_correction_and_slope(self):
        params, local_fns = _fixture(self.coefficients, support_ratio=0.5)
        # First overlap width is 6, so ratio 0.5 gives [8.5, 11.5].
        for x_value in (8.5, 11.5):
            with self.subTest(x=x_value):
                point = jnp.array([[x_value, 7.0]], dtype=jnp.float64)
                cut_params, plan = _cut_and_plan(
                    params, point, C1MinimalCorrectionPOUAssembly)

                def correction(current):
                    return C1MinimalCorrectionPOUAssembly.assemble(
                        cut_params, current[None, :], plan, local_fns)[2][0]

                np.testing.assert_allclose(correction(point[0]), 0.0, atol=2e-13)
                np.testing.assert_allclose(
                    jax.jacfwd(correction)(point[0])[:, 0], 0.0, atol=2e-12)

    def test_grad_jvp_vjp_reach_both_neighboring_networks(self):
        params, local_fns = _fixture(self.coefficients)
        points = jnp.array([[9.2, 6.0], [10.0, 8.0], [10.8, 11.0]],
                           dtype=jnp.float64)
        cut_params, plan = _cut_and_plan(
            params, points, C1MinimalCorrectionPOUAssembly)
        coefficients = cut_params["trainable"]["network"]["subdomain"]["coefficients"]

        def with_coefficients(current_coefficients, current_points=points):
            current_params = {
                "static": cut_params["static"],
                "trainable": {"network": {"subdomain": {
                    "coefficients": current_coefficients,
                }}},
            }
            return C1MinimalCorrectionPOUAssembly.assemble(
                current_params, current_points, plan, local_fns)[0]

        gradient = jax.grad(
            lambda current: jnp.sum(with_coefficients(current)))(coefficients)
        per_model = jnp.sum(jnp.abs(gradient), axis=(1, 2))
        self.assertGreater(float(per_model[0]), 0.0)
        self.assertGreater(float(per_model[1]), 0.0)

        _, tangent = jax.jvp(
            lambda current: with_coefficients(coefficients, current),
            (points,), (jnp.full_like(points, 0.1),))
        self.assertTrue(bool(jnp.all(jnp.isfinite(tangent))))
        values, pullback = jax.vjp(with_coefficients, coefficients)
        cotangent = pullback(jnp.ones_like(values))[0]
        self.assertGreater(float(jnp.linalg.norm(cotangent)), 0.0)

    def test_integer_roi_points_keep_interface_trace_differentiable(self):
        params, local_fns = _fixture(self.coefficients)
        points = jnp.array([[9, 6], [10, 8], [11, 11]], dtype=jnp.int32)
        cut_params, plan = _cut_and_plan(
            params, points, C1MinimalCorrectionPOUAssembly)
        value = C1MinimalCorrectionPOUAssembly.assemble(
            cut_params, points, plan, local_fns)[0]
        self.assertEqual(value.shape, (3, 2))
        self.assertTrue(bool(jnp.all(jnp.isfinite(value))))


if __name__ == "__main__":
    unittest.main()
