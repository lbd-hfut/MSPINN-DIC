import unittest

import jax
import jax.numpy as jnp
import numpy as np

from segpinndic.DIC_assemblies import polynomial_reproducing_reconstruction


jax.config.update("jax_enable_x64", True)


class PolynomialReproducingPrototypeTests(unittest.TestCase):
    def setUp(self):
        x, y = np.meshgrid(
            np.linspace(-3.0, 4.0, 8), np.linspace(1.0, 6.0, 7),
            indexing="xy")
        self.samples = jnp.asarray(np.stack([x.ravel(), y.ravel()], axis=1))
        self.queries = jnp.array([
            [-2.2, 1.4], [-0.3, 2.7], [1.8, 4.2], [3.7, 5.8],
        ], dtype=jnp.float64)

    def test_p1_affine_value_and_gradient_reproduction(self):
        def field(point):
            x, y = point
            return jnp.array([0.3+0.02*x-0.01*y, -0.2+0.01*x+0.03*y])

        values = jax.vmap(field)(self.samples)
        reconstructed = lambda query: polynomial_reproducing_reconstruction(
            self.samples, values, query[None, :], order=1)[0][0]
        np.testing.assert_allclose(
            jax.vmap(reconstructed)(self.queries),
            jax.vmap(field)(self.queries), rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(
            jax.vmap(jax.jacfwd(reconstructed))(self.queries),
            jax.vmap(jax.jacfwd(field))(self.queries),
            rtol=2e-11, atol=2e-11)

    def test_p2_quadratic_value_and_gradient_reproduction(self):
        def field(point):
            x, y = point
            return jnp.array([
                0.2+0.01*x-0.008*y+2e-4*x*x-1.5e-4*x*y+1e-4*y*y,
                -0.1+0.006*x+0.012*y-1e-4*x*x+2.5e-4*x*y-1.2e-4*y*y,
            ])

        values = jax.vmap(field)(self.samples)
        reconstructed = lambda query: polynomial_reproducing_reconstruction(
            self.samples, values, query[None, :], order=2)[0][0]
        np.testing.assert_allclose(
            jax.vmap(reconstructed)(self.queries),
            jax.vmap(field)(self.queries), rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(
            jax.vmap(jax.jacfwd(reconstructed))(self.queries),
            jax.vmap(jax.jacfwd(field))(self.queries),
            rtol=2e-10, atol=2e-10)


if __name__ == "__main__":
    unittest.main()
