import unittest

import jax
import jax.numpy as jnp
import numpy as np

from segpinndic.DIC_projections import (
    build_projection_plan, moment_residual, polynomial_basis,
    project_low, project_orthogonal,
)


def grid(nx=17, ny=15):
    x = np.linspace(-1, 1, nx, dtype=np.float32)
    y = np.linspace(-1, 1, ny, dtype=np.float32)
    return np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1).reshape(-1, 2)


class ModalProjectionTests(unittest.TestCase):
    def test_p0_p1_p2_identity_and_moment_orthogonality(self):
        points = grid()
        values = jnp.stack([
            jnp.sin(3.2*points[:, 0])+0.1*points[:, 1]**3,
            jnp.cos(2.7*points[:, 1])+0.2*points[:, 0]**3,
        ], axis=1)
        for order in (0, 1, 2):
            plan = build_projection_plan(points, order)
            low = project_low(plan, values)
            orthogonal = project_orthogonal(plan, values)
            np.testing.assert_allclose(low+orthogonal, values, atol=2e-7)
            self.assertLess(float(jnp.max(jnp.abs(
                moment_residual(plan, orthogonal)))), 1e-6)

    def test_exact_p1_and_p2_modes_are_removed(self):
        points = grid()
        p1 = polynomial_basis(points, 1)@jnp.asarray([
            [1.0, -.2], [.3, .4], [-.1, .2]])
        p2 = polynomial_basis(points, 2)@jnp.asarray([
            [1., -.2], [.3, .4], [-.1, .2], [.05, -.03],
            [.02, .04], [-.06, .01]])
        for order, values in ((1, p1), (2, p2)):
            residual = project_orthogonal(
                build_projection_plan(points, order), values)
            self.assertLess(float(jnp.sqrt(jnp.mean(residual**2))), 3e-7)

    def test_high_frequency_content_is_preserved_after_low_mode_removal(self):
        points = grid(65, 17)
        high = .01*jnp.sin(6*jnp.pi*points[:, 0])
        low = 1+.2*points[:, 0]-.1*points[:, 1]
        plan = build_projection_plan(points, 1)
        residual = project_orthogonal(plan, low+high)
        expected = project_orthogonal(plan, high)
        np.testing.assert_allclose(residual, expected, atol=4e-7)
        self.assertGreater(float(jnp.linalg.norm(residual)), 0.0)

    def test_affine_and_quadratic_residual_strain_vanishes(self):
        samples = jnp.asarray(grid())
        for order in (1, 2):
            plan = build_projection_plan(samples, order)
            coefficients = jnp.arange(
                polynomial_basis(samples, order).shape[1]*2,
                dtype=jnp.float32).reshape(-1, 2)/10
            sampled = polynomial_basis(samples, order)@coefficients

            def residual(point):
                raw = polynomial_basis(point[None, :], order)[0]@coefficients
                low = project_low(plan, sampled, point[None, :])[0]
                return raw-low

            gradient = jax.jacfwd(residual)(jnp.asarray([.13, -.27]))
            self.assertLess(float(jnp.linalg.norm(gradient)), 2e-6)

    def test_rank_conditioning_and_explicit_p2_to_p1_fallback(self):
        healthy = build_projection_plan(grid(), 2)
        self.assertEqual(healthy["rank"], 6)
        self.assertFalse(healthy["fallback"])
        self.assertTrue(np.isfinite(healthy["condition_number"]))

        points = np.asarray([
            [-1., -1.], [-1., 1.], [1., -1.], [1., 1.], [0., 0.]],
            dtype=np.float32)
        fallback = build_projection_plan(points, 2)
        self.assertTrue(fallback["fallback"])
        self.assertEqual(fallback["requested_order"], 2)
        self.assertEqual(fallback["effective_order"], 1)
        self.assertEqual(fallback["rank"], 3)
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            build_projection_plan(points, 2, allow_fallback=False)


if __name__ == "__main__":
    unittest.main()
