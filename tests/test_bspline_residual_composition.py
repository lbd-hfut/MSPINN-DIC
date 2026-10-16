import unittest

import jax
import jax.numpy as jnp
import numpy as np

from segpinndic.DIC_assemblies import (
    BSplineCosineResidualAssembly, BSplineP2ResidualAssembly,
)
from tests.bspline_residual_fixture import fixture


class BSplineResidualCompositionTests(unittest.TestCase):
    def test_zero_residual_and_zero_global_composition(self):
        for assembly in (BSplineCosineResidualAssembly,
                         BSplineP2ResidualAssembly):
            with self.subTest(assembly=assembly.__name__):
                params, points, plan, fns = fixture(assembly, local_scale=0.)
                global_value, residual, *_ = assembly.components(
                    params, points, plan, fns)
                total = assembly.assemble(params, points, plan, fns)[0]
                np.testing.assert_allclose(residual, 0.0, atol=1e-8)
                np.testing.assert_allclose(total, global_value, atol=1e-8)

                params, points, plan, fns = fixture(assembly, global_scale=0.)
                global_value, residual, *_ = assembly.components(
                    params, points, plan, fns)
                total = assembly.assemble(params, points, plan, fns)[0]
                np.testing.assert_allclose(global_value, 0.0, atol=1e-8)
                np.testing.assert_allclose(total, residual, atol=1e-8)

    def test_gradients_reach_controls_and_local_networks(self):
        for assembly in (BSplineCosineResidualAssembly,
                         BSplineP2ResidualAssembly):
            with self.subTest(assembly=assembly.__name__):
                params, points, plan, fns = fixture(assembly)

                def loss(trainable):
                    current = {"static": params["static"],
                               "trainable": trainable}
                    return jnp.sum(assembly.assemble(
                        current, points, plan, fns)[0]**2)

                gradient = jax.grad(loss)(params["trainable"])
                self.assertGreater(float(jnp.linalg.norm(
                    gradient["global_field"]["controls"])), 0.0)
                self.assertGreater(float(jnp.linalg.norm(
                    gradient["network"]["subdomain"]["coefficients"])), 0.0)


if __name__ == "__main__":
    unittest.main()

