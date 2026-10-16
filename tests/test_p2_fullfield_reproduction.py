import unittest

import jax
import jax.numpy as jnp
import numpy as np

from segpinndic.DIC_assemblies import P2FullFieldAssembly, p2_chain_conditioning
from tests.p2_fullfield_fixture import fixture


jax.config.update("jax_enable_x64", True)


class P2FullFieldReproductionTests(unittest.TestCase):
    def test_p0_p1_p2_value_gradient_and_strain(self):
        for coefficients in (
                [[.2, 0., 0.], [-.1, 0., 0.]],
                [[.2, .03, 0.], [-.1, -.02, 0.]],
                [[.2, .03, .001], [-.1, -.02, -.002]]):
            with self.subTest(coefficients=coefficients):
                params, _fixed, points, plan, local_fns = fixture(coefficients)
                field = lambda query: P2FullFieldAssembly.assemble(
                    params, query, plan, local_fns)[0]
                value = field(points)
                tangent_x = jnp.tile(jnp.array([[1., 0.]]), (len(points), 1))
                _, dx = jax.jvp(field, (points,), (tangent_x,))
                c = jnp.asarray(coefficients)
                exact = c[:, 0] + c[:, 1]*points[:, 0, None] + c[:, 2]*points[:, 0, None]**2
                exact_dx = c[:, 1] + 2*c[:, 2]*points[:, 0, None]
                np.testing.assert_allclose(value, exact, atol=2e-10)
                np.testing.assert_allclose(dx, exact_dx, atol=2e-10)
                strain = jnp.stack([dx[:, 0], jnp.zeros(len(points)),
                                    jnp.zeros(len(points))], axis=1)
                exact_strain = jnp.stack([
                    exact_dx[:, 0], jnp.zeros(len(points)),
                    jnp.zeros(len(points))], axis=1)
                np.testing.assert_allclose(strain, exact_strain, atol=2e-10)

    def test_conditioning_is_rank_three_and_finite(self):
        stats = p2_chain_conditioning(
            np.array([5., 15., 25., 35.]), np.linspace(0., 40., 81), 1e-12)
        self.assertEqual(stats["design_rank"], 3)
        self.assertTrue(np.isfinite(stats["condition_max"]))
        self.assertLess(stats["condition_max"], 1.0e6)


if __name__ == "__main__":
    unittest.main()

