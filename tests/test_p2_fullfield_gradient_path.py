import unittest

import jax
import jax.numpy as jnp
import numpy as np

from segpinndic.DIC_assemblies import P2FullFieldAssembly
from tests.p2_fullfield_fixture import fixture


jax.config.update("jax_enable_x64", True)


class P2FullFieldGradientPathTests(unittest.TestCase):
    def test_grad_jvp_and_vjp_cross_the_p2_solve(self):
        params, _fixed, points, plan, local_fns = fixture()

        def value(trainable, query):
            current = {"static": params["static"], "trainable": trainable}
            return P2FullFieldAssembly.assemble(
                current, query, plan, local_fns)[0]

        gradient = jax.grad(lambda trainable: jnp.sum(
            value(trainable, points)**2))(params["trainable"])
        self.assertGreater(float(jnp.linalg.norm(
            gradient["network"]["subdomain"]["coefficients"])), 0.0)

        tangent = jnp.tile(jnp.array([[1., 0.]]), (len(points), 1))
        _, spatial = jax.jvp(
            lambda query: value(params["trainable"], query),
            (points,), (tangent,))
        self.assertTrue(np.all(np.isfinite(np.asarray(spatial))))

        output, pullback = jax.vjp(
            lambda trainable: value(trainable, points), params["trainable"])
        cotangent = pullback(jnp.ones_like(output))[0]
        self.assertGreater(float(jnp.linalg.norm(
            cotangent["network"]["subdomain"]["coefficients"])), 0.0)


if __name__ == "__main__":
    unittest.main()

