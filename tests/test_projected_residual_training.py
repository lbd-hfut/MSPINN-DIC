import pickle
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from segpinndic.DIC_assemblies import BSplineProjectedCosineResidualAssembly
from tests.projected_residual_fixture import fixture


class ProjectedResidualTrainingTests(unittest.TestCase):
    def test_stage_one_excludes_local_and_stage_two_satisfies_moments(self):
        assembly = BSplineProjectedCosineResidualAssembly
        stage1, points, plan, fns = fixture(order=1, gate=0.)
        components = assembly.projection_components(stage1, points, plan, fns)
        np.testing.assert_allclose(components["residual"], 0., atol=1e-8)

        def loss1(trainable):
            return jnp.sum(assembly.assemble(
                {"static": stage1["static"], "trainable": trainable},
                points, plan, fns)[0]**2)

        gradient1 = jax.grad(loss1)(stage1["trainable"])
        self.assertEqual(float(jnp.linalg.norm(
            gradient1["network"]["subdomain"]["coefficients"])), 0.0)
        self.assertGreater(float(jnp.linalg.norm(
            gradient1["global_field"]["controls"])), 0.0)

        stage2, points, plan, fns = fixture(order=1, gate=1.)
        components = assembly.projection_components(stage2, points, plan, fns)
        self.assertGreater(float(jnp.linalg.norm(
            components["raw_local_values"])), 0.0)
        projection = components["projection"]
        moments = jnp.einsum(
            "kb,k,mkd->mbd", projection["sample_basis"],
            projection["weights"], components["projected_sample_values"])
        self.assertLess(float(jnp.max(jnp.abs(moments))), 2e-6)

        def loss2(trainable):
            return jnp.sum(assembly.assemble(
                {"static": stage2["static"], "trainable": trainable},
                points, plan, fns)[0]**2)

        gradient2 = jax.grad(loss2)(stage2["trainable"])
        self.assertGreater(float(jnp.linalg.norm(
            gradient2["network"]["subdomain"]["coefficients"])), 0.0)
        self.assertGreater(float(jnp.linalg.norm(
            gradient2["global_field"]["controls"])), 0.0)

    def test_p1_p2_small_adam_strain_and_checkpoint_roundtrip(self):
        assembly = BSplineProjectedCosineResidualAssembly
        for order in (1, 2):
            with self.subTest(order=order):
                params, points, plan, fns = fixture(order=order, gate=1.)
                optimiser = optax.adam(1e-3)
                state = optimiser.init(params["trainable"])

                def loss(trainable):
                    current = {"static": params["static"],
                               "trainable": trainable}
                    return jnp.mean(assembly.assemble(
                        current, points, plan, fns)[0]**2)

                trainable = params["trainable"]
                for _ in range(3):
                    value, gradient = jax.value_and_grad(loss)(trainable)
                    updates, state = optimiser.update(
                        gradient, state, trainable)
                    trainable = optax.apply_updates(trainable, updates)
                self.assertTrue(np.isfinite(float(value)))

                current = {"static": params["static"],
                           "trainable": trainable}
                tangent_x = jnp.ones_like(points)*jnp.array([1., 0.])
                tangent_y = jnp.ones_like(points)*jnp.array([0., 1.])
                field = lambda query: assembly.assemble(
                    current, query, plan, fns)[0]
                displacement, dx = jax.jvp(field, (points,), (tangent_x,))
                _, dy = jax.jvp(field, (points,), (tangent_y,))
                strain = jnp.stack(
                    [dx[:, 0], .5*(dy[:, 0]+dx[:, 1]), dy[:, 1]], axis=1)
                self.assertTrue(np.all(np.isfinite(displacement)))
                self.assertTrue(np.all(np.isfinite(strain)))

                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)/"checkpoint.jax"
                    with path.open("wb") as stream:
                        pickle.dump(jax.tree_util.tree_map(np.asarray, current), stream)
                    with path.open("rb") as stream:
                        restored = pickle.load(stream)
                restored_value = assembly.assemble(
                    restored, points, plan, fns)[0]
                np.testing.assert_allclose(
                    restored_value, displacement, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
