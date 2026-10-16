import unittest

import jax
import jax.numpy as jnp
import numpy as np

from segpinndic.DIC_assemblies import BSplineProjectedCosineResidualAssembly
from tests.projected_residual_fixture import fixture


class ProjectedResidualGradientPathTests(unittest.TestCase):
    def test_grad_jvp_vjp_cross_projection_coefficients_and_subtraction(self):
        assembly = BSplineProjectedCosineResidualAssembly
        params, points, plan, fns = fixture(order=2)

        def output(trainable):
            return assembly.assemble(
                {"static": params["static"], "trainable": trainable},
                points, plan, fns)[0]

        gradient = jax.grad(lambda tree: jnp.sum(output(tree)**2))(
            params["trainable"])
        self.assertGreater(float(jnp.linalg.norm(
            gradient["network"]["subdomain"]["coefficients"])), 0.0)
        self.assertGreater(float(jnp.linalg.norm(
            gradient["global_field"]["controls"])), 0.0)

        tangent = jax.tree_util.tree_map(jnp.ones_like, params["trainable"])
        _, directional = jax.jvp(output, (params["trainable"],), (tangent,))
        self.assertGreater(float(jnp.linalg.norm(directional)), 0.0)
        value, pullback = jax.vjp(output, params["trainable"])
        cotangent = pullback(jnp.ones_like(value))[0]
        self.assertGreater(float(jnp.linalg.norm(
            cotangent["network"]["subdomain"]["coefficients"])), 0.0)

        _, spatial = jax.jvp(
            lambda query: assembly.assemble(
                params, query, plan, fns)[0],
            (points,), (jnp.ones_like(points)*jnp.array([1., 0.]),))
        self.assertTrue(np.all(np.isfinite(spatial)))
        self.assertGreater(float(jnp.linalg.norm(spatial)), 0.0)


if __name__ == "__main__":
    unittest.main()
