import jax.nn
from segpinndic.DIC_importlib import jnp, np, jax
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
    def init_params(ref_img, QKBQKT_def, mask, degree):
        static_params = {
            "dims":(2,2),
            "QKBQKT_def": QKBQKT_def,
            "ref_img": ref_img,
            "mask": mask,
            "degree": degree,
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
    def loss_fn(all_params, x_batch, uv, takes, num_models_shape):
        ref_img = all_params["static"]["problem"]["ref_img"]
        QKBQKT_def = all_params["static"]["problem"]["QKBQKT_def"]
        degree = all_params["static"]["problem"]["degree"]
        u, v = uv[:,0], uv[:,1]

        xref, yref = x_batch[:,0], x_batch[:,1]
        xs, ys = xref + u, yref + v

        # warp defimg
        H, W = QKBQKT_def.shape[:2]

        xs_floor = jax.lax.stop_gradient(jnp.floor(xs)).astype(jnp.int32)
        ys_floor = jax.lax.stop_gradient(jnp.floor(ys)).astype(jnp.int32)

        xs_oob = (xs_floor < 0) | (xs_floor >= W)
        ys_oob = (ys_floor < 0) | (ys_floor >= H)
        mask = xs_oob | ys_oob

        xs_floor = jnp.clip(xs_floor, 0, W - 1)
        ys_floor = jnp.clip(ys_floor, 0, H - 1)

        # (N,6,6)
        QK_B_QKT = QKBQKT_def[ys_floor, xs_floor]

        xd = xs - xs_floor
        yd = ys - ys_floor

        powers = jnp.arange(degree+1)
        x_vec = xd[:, None] ** powers[None, :]
        y_vec = yd[:, None] ** powers[None, :]

        tmp = jnp.einsum("ni,nij->nj", y_vec, QK_B_QKT)
        warp_values = jnp.einsum("ni,ni->n", tmp, x_vec)

        values = ref_img[yref.astype(jnp.int32), xref.astype(jnp.int32)]
        # ---------- global MSE ----------
        mse = jnp.mean((warp_values - values) ** 2)
        
        # ---------- per-partition MSE ----------
        m = takes[0]   # partition id
        n = takes[1]   # valid indices

        f = values[n]
        g = warp_values[n]

        num_models = num_models_shape.shape[0]

        err = (f - g) ** 2

        # 每个分区的点数
        counts = jax.ops.segment_sum(jnp.ones_like(err), m, num_models)

        # 每个分区误差和
        err_sum = jax.ops.segment_sum(err, m, num_models)

        # 每个分区 MSE
        mse_per_partition = err_sum / (counts + 1e-8)

        return mse, mse_per_partition
    

class DIC_ZNSSD(Problem):
    """DIC problem class with ZNSSD loss function"""

    @staticmethod
    def init_params(ref_img, QKBQKT_def, mask, degree):
        static_params = {
            "dims":(2,2),
            "QKBQKT_def": QKBQKT_def,
            "ref_img": ref_img,
            "mask": mask,
            "degree": degree,
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
    def loss_fn(all_params, x_batch, uv, takes, num_models_shape):
        ref_img = all_params["static"]["problem"]["ref_img"]
        QKBQKT_def = all_params["static"]["problem"]["QKBQKT_def"]
        degree = all_params["static"]["problem"]["degree"]
        u, v = uv[:,0], uv[:,1]

        xref, yref = x_batch[:,0], x_batch[:,1]
        xs, ys = xref + u, yref + v

        # warp defimg
        H, W = QKBQKT_def.shape[:2]

        xs_floor = jax.lax.stop_gradient(jnp.floor(xs)).astype(jnp.int32)
        ys_floor = jax.lax.stop_gradient(jnp.floor(ys)).astype(jnp.int32)

        xs_oob = (xs_floor < 0) | (xs_floor >= W)
        ys_oob = (ys_floor < 0) | (ys_floor >= H)
        mask = xs_oob | ys_oob

        xs_floor = jnp.clip(xs_floor, 0, W - 1)
        ys_floor = jnp.clip(ys_floor, 0, H - 1)

        # (N,6,6)
        QK_B_QKT = QKBQKT_def[ys_floor, xs_floor]

        xd = xs - xs_floor
        yd = ys - ys_floor

        powers = jnp.arange(degree+1)
        x_vec = xd[:, None] ** powers[None, :]
        y_vec = yd[:, None] ** powers[None, :]

        tmp = jnp.einsum("ni,nij->nj", y_vec, QK_B_QKT)
        warp_values = jnp.einsum("ni,ni->n", tmp, x_vec)

        values = ref_img[yref.astype(jnp.int32), xref.astype(jnp.int32)]

        # ---------- ZNSSD ----------
        m = takes[0]
        n = takes[1]
        
        f = values[n]
        g = warp_values[n]
        
        num_models = num_models_shape.shape[0]
        counts = jax.ops.segment_sum(jnp.ones_like(f), m, num_models)
        
        # ---------- mean ----------
        f_mean = jax.ops.segment_sum(f, m, num_models) / counts
        g_mean = jax.ops.segment_sum(g, m, num_models) / counts
        
        f_mean = jax.lax.stop_gradient(f_mean)
        g_mean = jax.lax.stop_gradient(g_mean)
        
        f_mean_p = f_mean[m]
        g_mean_p = g_mean[m]
        
        # ---------- std ----------
        f_var = jax.ops.segment_sum((f - f_mean_p)**2, m, num_models) / counts
        g_var = jax.ops.segment_sum((g - g_mean_p)**2, m, num_models) / counts

        f_std = jax.lax.stop_gradient(jnp.sqrt(f_var)) + 1e-8
        g_std = jax.lax.stop_gradient(jnp.sqrt(g_var)) + 1e-8

        f_std_p = f_std[m]
        g_std_p = g_std[m]
        
        # ---------- normalize ----------
        # f_norm = (f - f_mean_p) / f_std_p
        # g_norm = (g - g_mean_p) / g_std_p
        
        # Avoid excessively small gradients
        f_norm = (f - f_mean_p)
        g_norm = (g - g_mean_p) / g_std_p * f_std_p
        
        # ---------- global ZNSSD ----------
        err = (f_norm - g_norm) ** 2
        znssd = jnp.mean(err)

        # ---------- per-partition ZNSSD ----------
        num_models = num_models_shape.shape[0]

        # 每个分区点数
        counts = jax.ops.segment_sum(jnp.ones_like(err), m, num_models)

        # 每个分区误差和
        err_sum = jax.ops.segment_sum(err, m, num_models)

        # 每个分区 ZNSSD
        znssd_per_partition = err_sum / (counts + 1e-8)

        return znssd, znssd_per_partition
    

    
