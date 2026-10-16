import unittest

import jax
import jax.numpy as jnp
import numpy as np

from segpinndic.DIC_assemblies import CosinePOUAssembly, QuinticPOUAssembly
from segpinndic.DIC_decompositions import RectangularDecompositionND
from segpinndic.DIC_field_diagnostics import (
    decompose_normalized_pou_gradient,
    evaluate_assembled_fields,
    evaluate_local_fields,
)


jax.config.update("jax_enable_x64", True)


def _fixture(assembly):
    centers = [np.array([5.0, 15.0, 25.0, 35.0]), np.array([10.0])]
    widths = [np.full(4, 16.0), np.array([32.0])]
    decomposition_static, _ = RectangularDecompositionND.init_params(
        centers, widths, (np.zeros(2), np.ones(2)))
    assembly_static, _ = assembly.init_params()
    coefficients = jnp.array([
        [[0.1, 0.020, -0.010], [-0.2, 0.010, 0.030]],
        [[0.4, -0.012, 0.018], [0.1, 0.028, -0.020]],
        [[-0.3, 0.035, 0.012], [0.2, -0.018, 0.014]],
        [[0.2, -0.025, 0.021], [-0.1, 0.022, 0.037]],
    ], dtype=jnp.float64)
    all_params = {
        "static": {
            "decomposition": decomposition_static,
            "assembly": assembly_static,
        },
        "trainable": {"network": {"subdomain": {
            "coefficients": coefficients,
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
    model_fns = local_fns + (assembly,)
    return all_params, local_fns, model_fns


class LocalFieldDiagnosticsTests(unittest.TestCase):
    def test_local_evaluation_is_before_assembly(self):
        params, local_fns, _ = _fixture(CosinePOUAssembly)
        points = jnp.array([[10.0, 4.0], [20.0, 8.0]], dtype=jnp.float64)
        models = jnp.array([0, 2])
        result = evaluate_local_fields(params, points, models, local_fns)
        coefficients = params["trainable"]["network"]["subdomain"]["coefficients"]
        expected_value = jnp.stack([
            coefficients[model, :, 0]
            + coefficients[model, :, 1]*point[0]
            + coefficients[model, :, 2]*point[1]
            for point, model in zip(points, models)
        ])
        expected_gradient = jnp.stack([
            coefficients[model, :, 1:3] for model in models])
        np.testing.assert_allclose(result["value"], expected_value, atol=1e-13)
        np.testing.assert_allclose(result["gradient"], expected_gradient, atol=1e-13)

    def test_pou_gradient_decomposition_identity(self):
        points = jnp.array([
            [3.0, 5.0], [9.0, 7.0], [12.0, 8.0], [19.0, 11.0],
            [22.0, 13.0], [29.0, 9.0], [36.0, 6.0],
        ], dtype=jnp.float64)
        for assembly in (CosinePOUAssembly, QuinticPOUAssembly):
            with self.subTest(assembly=assembly.__name__):
                params, _, model_fns = _fixture(assembly)
                decomposition = decompose_normalized_pou_gradient(
                    params, points, RectangularDecompositionND,
                    assembly, model_fns)
                assembled = evaluate_assembled_fields(
                    params, points, RectangularDecompositionND,
                    assembly, model_fns)
                np.testing.assert_allclose(
                    decomposition["value"], assembled["value"],
                    rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(
                    decomposition["total_gradient"], assembled["gradient"],
                    rtol=2e-11, atol=2e-11)
                np.testing.assert_allclose(
                    decomposition["total_gradient"],
                    decomposition["local_gradient"]
                    + decomposition["window_gradient"],
                    rtol=1e-13, atol=1e-13)


if __name__ == "__main__":
    unittest.main()
