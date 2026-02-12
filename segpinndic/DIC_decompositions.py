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
        s1, s2 = jnp.any(inside_, axis=1), inside_.sum(0)
        return (s1, s2)# (n), (m)

    # get fully-populated batches by shifting last value of irange
    r = x_batch.shape[0]%batch_size
    shift = batch_size-r if r else 0
    irange = jnp.arange(0, x_batch.shape[0], batch_size)# (k)
    mask = jnp.ones((len(irange), batch_size), dtype=bool)# (k, n)
    irange = irange.at[-1].add(-shift)
    mask = mask.at[-1,:shift].set(False)
    s1, s2 = jax.lax.map(batch_step, (irange, mask))

    # parse ims and ips
    inside_ips = jnp.concatenate([s1[:-1].ravel(), s1[-1][shift:]], axis=0)# (n)
    inside_ims = s2.sum(0)# (m)
    d = (inside_ims.mean()**(1/x_batch.shape[1]))# average number of points per model
    s = inside_ims.sum()
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
        itake = jnp.cumsum(inside_)-1# (n*m)
        ii_ = jnp.expand_dims(inside_,1)*ii.at[:,0].add(i)# (n*m, 2)
        take, s = carry
        take = take.at[s+itake].add(ii_)# (s, 2)
        return (take, s+itake[-1]+1), None

    ix,iy = jnp.meshgrid(jnp.arange(batch_size), jnp.arange(ims.shape[0]), indexing="ij")# (n, m)
    ii = jnp.stack([ix.ravel(), iy.ravel()], axis=1)# (n*m, 2)
    take = jnp.zeros((s,2), dtype=int)# (s, 2)
    (take, _), _ = jax.lax.scan(batch_step, (take, 0), (irange, mask))
    return take

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
    inside_ims = jnp.arange(ims.shape[0])[inside_ims]
    s = s.item()
    take = _inside_take_batch(all_params, x_batch, ims, batch_size, inside_fn, s, irange, mask)
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
    inside_ips = jnp.arange(x_batch.shape[0])[inside_ips]
    return inside_ips, d