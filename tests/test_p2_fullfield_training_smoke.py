import unittest

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import optax

from segpinndic.DIC_assemblies import P2FullFieldAssembly
from segpinndic.DIC_seed_trainer import FBPINN_seed_loss
from segpinndic.DIC_trainers import FBPINN_loss
from tests.p2_fullfield_fixture import fixture


jax.config.update("jax_enable_x64", True)


class P2FullFieldTrainingSmokeTests(unittest.TestCase):
    def test_seed_adam_and_lbfgs_use_the_same_assembled_forward(self):
        params, fixed, points, plan, local_fns = fixture()
        model_fns = local_fns + (P2FullFieldAssembly,)
        target = jnp.zeros((len(points), 2), dtype=points.dtype)
        active = params["trainable"]

        seed_loss, seed_grad = jax.value_and_grad(FBPINN_seed_loss)(
            active, fixed, params["static"], plan, points, model_fns, target)
        self.assertTrue(np.isfinite(float(seed_loss)))
        self.assertGreater(float(jnp.linalg.norm(
            seed_grad["network"]["subdomain"]["coefficients"])), 0.0)

        loss_fn = lambda _p, _x, value: jnp.mean((value-target)**2)
        adam_loss, adam_grad = jax.value_and_grad(FBPINN_loss)(
            active, fixed, params["static"], plan, points, model_fns, loss_fn)
        optimiser = optax.adam(1e-3)
        updates, _ = optimiser.update(
            adam_grad, optimiser.init(active), active)
        after_adam = optax.apply_updates(active, updates)
        self.assertTrue(np.isfinite(float(adam_loss)))
        self.assertFalse(np.allclose(
            np.asarray(after_adam["network"]["subdomain"]["coefficients"]),
            np.asarray(active["network"]["subdomain"]["coefficients"])))

        solver = jaxopt.LBFGS(
            fun=lambda current: FBPINN_loss(
                current, fixed, params["static"], plan, points,
                model_fns, loss_fn),
            maxiter=1, tol=0.0, implicit_diff=False)
        state = solver.init_state(after_adam)
        _after_lbfgs, state = solver.update(after_adam, state)
        self.assertTrue(np.isfinite(float(state.value)))


if __name__ == "__main__":
    unittest.main()
