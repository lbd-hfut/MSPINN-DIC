import unittest

import jax
import jax.numpy as jnp
import numpy as np

from segpinndic.DIC_coarse_fields import CoarseBSplineField, bspline_basis_1d


jax.config.update("jax_enable_x64", True)


class CoarseBSplineFieldTests(unittest.TestCase):
    def test_grid_construction_partition_unity_and_boundary(self):
        for grid in ((4, 4), (6, 6), (8, 8)):
            with self.subTest(grid=grid):
                static, trainable = CoarseBSplineField.init_params(
                    grid, 3, [10., 20.], [50., 80.])
                self.assertEqual(trainable["controls"].shape, grid+(2,))
                for x in jnp.linspace(0., 1., 31):
                    bx = bspline_basis_1d(
                        x, static["knots_x"], 3, grid[0])
                    self.assertAlmostEqual(float(jnp.sum(bx)), 1.0, places=6)
                    self.assertGreaterEqual(float(jnp.min(bx)), -1e-7)
                left = bspline_basis_1d(
                    jnp.array(0.), static["knots_x"], 3, grid[0])
                right = bspline_basis_1d(
                    jnp.array(1.), static["knots_x"], 3, grid[0])
                np.testing.assert_allclose(left, np.eye(grid[0])[0], atol=1e-7)
                np.testing.assert_allclose(right, np.eye(grid[0])[-1], atol=1e-7)

    def test_constant_field_spatial_and_parameter_gradients_jit_vmap(self):
        static, trainable = CoarseBSplineField.init_params(
            (6, 6), 3, [0., 0.], [40., 20.])
        constant = jnp.array([2.5, -1.25])
        trainable = {"controls": jnp.broadcast_to(
            constant, trainable["controls"].shape)}
        points = jnp.array([[0., 0.], [7., 3.], [20., 10.], [40., 20.]])
        evaluate = jax.jit(lambda current: CoarseBSplineField.evaluate_batch(
            static, trainable, current))
        values = evaluate(points)
        np.testing.assert_allclose(values, np.broadcast_to(constant, values.shape),
                                   atol=2e-6)
        spatial = jax.jacfwd(lambda point: CoarseBSplineField.evaluate(
            static, trainable, point))(jnp.array([17., 9.]))
        np.testing.assert_allclose(spatial, 0.0, atol=2e-6)

        gradient = jax.grad(lambda controls: jnp.sum(
            CoarseBSplineField.evaluate(
                static, {"controls": controls}, jnp.array([17., 9.]))**2))(
                    trainable["controls"])
        self.assertTrue(np.all(np.isfinite(np.asarray(gradient))))
        self.assertGreater(float(jnp.linalg.norm(gradient)), 0.0)


if __name__ == "__main__":
    unittest.main()
