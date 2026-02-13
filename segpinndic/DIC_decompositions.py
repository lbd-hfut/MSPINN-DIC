from segpinndic.DIC_importlib import jnp, vmap, tree_map, np, plt, mcoll, jax, partial

from segpinndic import DIC_windows
from segpinndic import DIC_networks
from segpinndic.utils.jax_util import tree_index
from segpinndic.utils.other import colors











@partial(jax.jit, static_argnums=(3,4))
def _inside_sum_batch(all_params, x_batch, ims, batch_size, inside_fn):
    """
    Computes summary statistics of which (point, model) pairs satisfy `inside_fn`,
    processing the data in fixed-size batches.

    Args:
        all_params: Model parameters (pytree).
        x_batch: Array of input points of shape (N, xd).
        ims: Model indices or model-specific data (size M).
        batch_size: Number of points per batch (static for JIT).
        inside_fn: Function (params, x_batch_sub, ims) -> (n, m) boolean array
                   indicating whether each point belongs to each model.

    Returns:
        (s, inside_ips, inside_ims, d):
            s: Total number of (point, model) pairs where inside == True.
            inside_ips: Boolean mask over points (length N).
            inside_ims: Boolean mask over models (length M).
            d: Estimated average spatial density of points per model.
        irange: Starting indices of each batch.
        mask: Boolean mask correcting padding in last batch.
    """
    def batch_step(x):
        i, mask = x
        x_batch_ = jax.lax.dynamic_slice(x_batch, (i,0), (batch_size, x_batch.shape[1]))# (n, xd)
        inside_ = jnp.expand_dims(mask,1)*inside_fn(all_params, x_batch_, ims)# (n, m)
        # s1: Does each point in this batch belong to at least one model?
        # s2: How many points does each model contain in this batch?
        s1, s2 = jnp.any(inside_, axis=1), inside_.sum(0)
        return (s1, s2)# (n), (m)

    # get fully-populated batches by shifting last value of irange
    r = x_batch.shape[0]%batch_size
    shift = batch_size-r if r else 0
    irange = jnp.arange(0, x_batch.shape[0], batch_size)# (k)
    mask = jnp.ones((len(irange), batch_size), dtype=bool)# (k, n)
    irange = irange.at[-1].add(-shift)
    mask = mask.at[-1,:shift].set(False)
    s1, s2 = jax.lax.map(batch_step, (irange, mask)) # auto stack, return ((k, n), (k, m))

    # parse ims and ips
    # inside_ips: Which points belong to at least one model
    # inside_ims: How many points does each model contain?
    inside_ips = jnp.concatenate([s1[:-1].ravel(), s1[-1][shift:]], axis=0)# (n)
    inside_ims = s2.sum(0)# (m)
    d = (inside_ims.mean()**(1/x_batch.shape[1]))# average number of points per model
    s = inside_ims.sum() # How many pairs are inside in total?
    inside_ims = inside_ims.astype(bool)
    return (s, inside_ips, inside_ims, d), irange, mask

@partial(jax.jit, static_argnums=(3,4,5))
def _inside_take_batch(all_params, x_batch, ims, batch_size, inside_fn, s, irange, mask):
    """
    Processes a single batch.

    Args:
        x: Tuple (i, mask)
            i: starting index of batch
            mask: boolean mask to ignore padded elements in last batch

    Returns:
        s1: Boolean mask (batch_size,) → whether each point is inside any model
        s2: Integer counts (M,) → number of inside points per model
    """
    def batch_step(carry, x):
        i, mask = x
        x_batch_ = jax.lax.dynamic_slice(x_batch, (i,0), (batch_size, x_batch.shape[1]))# (n, xd)
        inside_ = jnp.expand_dims(mask,1)*inside_fn(all_params, x_batch_, ims)# (n, m)
        inside_ = inside_.ravel()# (n*m)
        itake = jnp.cumsum(inside_)-1# (n*m) Give us the index (starting from 0) of this True value.
        ii_ = jnp.expand_dims(inside_,1)*ii.at[:,0].add(i)# (n*m, 2)
        take, s = carry
        take = take.at[s+itake].add(ii_)# (s, 2)
        return (take, s+itake[-1]+1), None

    ix,iy = jnp.meshgrid(jnp.arange(batch_size), jnp.arange(ims.shape[0]), indexing="ij")# (n, m)
    ii = jnp.stack([ix.ravel(), iy.ravel()], axis=1)# (n*m, 2)
    take = jnp.zeros((s,2), dtype=int)# (s, 2)
    (take, _), _ = jax.lax.scan(batch_step, (take, 0), (irange, mask))
    return take # each element is (point_index, model_index)

def inside_points_batch(all_params, x_batch, ims, batch_size, inside_fn):
    """
    Constructs the explicit list of (point_index, model_index) pairs
    where inside_fn is True.

    Args:
        all_params: Model parameters.
        x_batch: Input points (N, xd).
        ims: Model indices or data (M).
        batch_size: Batch size (static).
        inside_fn: Boolean membership function.
        s: Total number of inside pairs (static).
        irange: Batch starting indices.
        mask: Padding mask.

    Returns:
        take: Array of shape (s, 2), where each row is (point_idx, model_idx).
    """
    assert batch_size <= x_batch.shape[0]
    (s, inside_ips, inside_ims, d), irange, mask = _inside_sum_batch(all_params, x_batch, ims, batch_size, inside_fn)
    inside_ims = jnp.arange(ims.shape[0])[inside_ims] # Convert the Boolean mask into a model index
    s = s.item()
    take = _inside_take_batch(all_params, x_batch, ims, batch_size, inside_fn, s, irange, mask)
    # point_indices model_indices valid_model_indices
    return take[:,0], take[:,1], inside_ims

def inside_models_batch(all_params, x_batch, ims, batch_size, inside_fn):
    """
        Iterates over batches and accumulates valid (point, model) pairs.

        carry:
            take: Output array being filled
            s: Current write pointer

        x:
            (i, mask) batch start index and padding mask
        """
    assert batch_size <= x_batch.shape[0]
    (s, inside_ips, inside_ims, d), irange, mask = _inside_sum_batch(all_params, x_batch, ims, batch_size, inside_fn)
    inside_ips = jnp.arange(x_batch.shape[0])[inside_ips] # Convert the Boolean mask into a point index
    return inside_ips, d


if __name__ == "__main__":

    import jax.random as random

    def inside_fn(all_params, x_batch, ims):
        "Code for assessing if point is in ND hyperrectangle"
        x_batch = jnp.expand_dims(x_batch, 1)# (n,1,xd)
        xmins = jnp.expand_dims(all_params[0][ims], 0)# (1,mc,xd)
        xmaxs = jnp.expand_dims(all_params[1][ims], 0)# (1,mc,xd)
        inside = (x_batch >= xmins) & (x_batch <= xmaxs)# (n,mc,xd)
        inside = jnp.all(inside, -1)# (n,mc) keep as bool to reduce memory
        return inside

    def inside(all_params, x_batch, ims, inside_fn):
        "full batch code to compare to"
        inside = inside_fn(all_params, x_batch, ims)# (n, m)
        n_take, m_take = jnp.nonzero(inside)
        inside_ims = jnp.nonzero(jnp.any(inside, axis=0))[0]
        inside_ips = jnp.nonzero(jnp.any(inside, axis=1))[0]
        return n_take, m_take, inside_ims, inside_ips

    n,m = 10000, 1000
    x_batch = random.uniform(random.PRNGKey(0), (n,2), minval=0, maxval=2)
    c = random.uniform(random.PRNGKey(0), (m,2), minval=1, maxval=3)
    xmin, xmax = c.copy(), c.copy()
    xmin -= 0.1
    xmax += 0.1
    all_params = [xmin, xmax]
    ims = jnp.arange(m)

    n_take_true, m_take_true, inside_ims_true, inside_ips_true = inside(all_params, x_batch, ims, inside_fn)

    for batch_size in [1, 9, 10, 128, n]:
        print(batch_size)

        n_take, m_take, inside_ims = inside_points_batch(all_params, x_batch, ims, batch_size, inside_fn)
        inside_ips, d = inside_models_batch(all_params, x_batch, ims, batch_size, inside_fn)

        assert (n_take_true==n_take).all()
        assert (m_take_true==m_take).all()
        assert (inside_ims_true==inside_ims).all()
        assert (inside_ips_true==inside_ips).all()