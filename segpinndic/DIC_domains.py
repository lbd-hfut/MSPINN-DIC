from segpinndic.DIC_importlib import jax, jnp, np
import scipy.stats

from segpinndic import DIC_networks

class Domain:
    """Base domain class to be inherited by different domain classes.

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
            "xd":None,# dimensionality of x
            }
        raise NotImplementedError

    @staticmethod
    def sample_interior(all_params, batch_shape):
        """Samples interior of domain.
        Returns x_batch points in interior of domain"""
        raise NotImplementedError

    @staticmethod
    def norm_fn(all_params, x):
        """"Applies norm function, for a SINGLE point with shape (xd,)"""# note only used for PINNs, FBPINN norm function defined in Decomposition
        raise NotImplementedError
    
    
class RectangularDomainND(Domain):

    @staticmethod
    def init_params(xmin, xmax):

        assert xmin.shape == xmax.shape
        assert xmin.ndim == 1
        xd = len(xmin)
        
        static_params = {
            "xd":xd,
            "xmin":jnp.array(xmin),
            "xmax":jnp.array(xmax),
            }
        return static_params, {}
    
    @staticmethod
    def sample_interior(all_params, batch_shape):
        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        return RectangularDomainND._rectangle_samplerND(xmin, xmax, batch_shape)

    @staticmethod
    def norm_fn(all_params, x):
        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        mu, sd = (xmax+xmin)/2, (xmax-xmin)/2
        x = DIC_networks.norm(mu, sd, x)
        return x
    
    @staticmethod
    def _rectangle_samplerND(xmin, xmax, batch_shape):
        "Get flattened samples of x in a rectangle, either on mesh or random"

        assert xmin.shape == xmax.shape
        assert xmin.ndim == 1
        xd = len(xmin)
        assert len(batch_shape) == xd

        xs = [jnp.linspace(xmin, xmax, b) for xmin,xmax,b in zip(xmin, xmax, batch_shape)]
        xx = jnp.stack(jnp.meshgrid(*xs, indexing="ij"), -1)# (batch_shape, xd)
        x_batch = xx.reshape((-1, xd))

        return jnp.array(x_batch)
    
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    domain = RectangularDomainND
    # 2D
    xmin, xmax = jnp.array([0,1]), jnp.array([1,2])
    batch_shape = (10,20)
    
    ps_ = domain.init_params(xmin, xmax)
    all_params = {"static":{"domain":ps_[0]}, "trainable":{"domain":ps_[1]}}
    x_batch = domain.sample_interior(all_params, batch_shape)
    
    plt.figure()
    plt.scatter(x_batch[:,0], x_batch[:,1])
    plt.show()