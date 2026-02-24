import jax.nn
from segpinndic.DIC_importlib import jnp, np
from segpinndic.utils.logger import logger

class Problem:
    """Base problem class to be inherited by different problem classes.

    Note all methods in this class are jit compiled / used by JAX,
    so they must not include any side-effects!
    (A side-effect is any effect of a function that doesn’t appear in its output)
    This is why only static methods are defined.
    """

    # required methods

    @staticmethod
    def init_params(*args):
        """Initialise class parameters.
        Returns tuple of dicts ({k: pytree}, {k: pytree}) containing static and trainable parameters"""

        # below parameters need to be defined
        static_params = {
            "dims":None,# (ud, xd)# dimensionality of u and x
            }
        raise NotImplementedError

    @staticmethod
    def sample_constraints(all_params, domain):
        """Samples all constraints.
        Returns [[x_batch, *any_constraining_values, required_ujs], ...]. Each list element contains
        the x_batch points and any constraining values passed to the loss function, and the required
        solution and gradient components required in the loss function, for each constraint."""
        raise NotImplementedError

    @staticmethod
    def loss_fn(all_params, constraints):
        """Computes the PINN loss function, using constraints with the same structure output by sample_constraints"""
        raise NotImplementedError
    

class DIC_MSE(Problem):
    """DIC problem class with MSE loss function"""

    @staticmethod
    def init_params(ref_img, QKBQKT_def, mask):
        static_params = {
            "dims":(2,2),
            "QKBQKT_def": QKBQKT_def,
            "ref_img": ref_img,
            "mask": mask,
        }
        return static_params, {}
    
    @staticmethod
    def sample_constraints(all_params, domain):
        x_batch_global = domain.sample_interior(all_params["static"]["problem"]["mask"])

        required_ujs = (
            (0,()),
            (1,()),
            (0,(0,)),
            (0,(1,)),
            (1,(0,)),
            (1,(1,))
        )
        return [[x_batch_global, required_ujs]]
    
    @staticmethod
    def loss_fn(all_params, x_batch, constraints):
        ref_img = all_params["static"]["problem"]["ref_img"]
        QKBQKT_def = all_params["static"]["problem"]["QKBQKT_def"]
        u, v = constraints

        xref, yref = x_batch[:,0], x_batch[:,1]
        xs, ys = xref + u, yref + v

        # warp defimg
        H, W = QKBQKT_def.shape[:2]

        xs_floor = jnp.floor(xs).astype(jnp.int32)
        ys_floor = jnp.floor(ys).astype(jnp.int32)

        xs_oob = (xs_floor < 0) | (xs_floor >= W)
        ys_oob = (ys_floor < 0) | (ys_floor >= H)
        mask = xs_oob | ys_oob

        xs_floor = jnp.clip(xs_floor, 0, W - 1)
        ys_floor = jnp.clip(ys_floor, 0, H - 1)

        # (N,6,6)
        QK_B_QKT = QKBQKT_def[ys_floor, xs_floor]

        xd = xs - xs_floor
        yd = ys - ys_floor

        powers = jnp.arange(6)
        x_vec = xd[:, None] ** powers[None, :]
        y_vec = yd[:, None] ** powers[None, :]

        tmp = jnp.einsum("ni,nij->nj", y_vec, QK_B_QKT)
        warp_values = jnp.einsum("ni,ni->n", tmp, x_vec)

        valus = ref_img[yref.astype(jnp.int32), xref.astype(jnp.int32)]

        mse = jnp.mean((warp_values - valus) ** 2)
        return mse
    

class DIC_ZNCC(Problem):
    """DIC problem class with MSE loss function"""

    @staticmethod
    def init_params(ref_img, QKBQKT_def, mask):
        static_params = {
            "QKBQKT_def": QKBQKT_def,
            "ref_img": ref_img,
            "mask": mask,
        }
        return static_params, {}

    
