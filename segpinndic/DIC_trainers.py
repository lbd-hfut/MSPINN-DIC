from segpinndic.DIC_importlib import os, time, pickle, jax, jnp, np, plt, SummaryWriter, \
    jit, vmap, value_and_grad, jvp, partial, random, optax
import IPython.display

from segpinndic.utils.logger import switch_to_file_logger, logger
from segpinndic.utils.jax_util import tree_index, total_size, str_tensor, partition, combine
from segpinndic import DIC_networks, DIC_plot_trainer


class _Trainer:
    "Generic model trainer class"

    def __init__(self, c):
        "Initialise device and output directories"

        # clear directories
        c.get_outdirs()
        c.save_constants_file()
        logger.info(c)

        # initialise summary writer
        writer = SummaryWriter(c.summary_out_dir)
        writer.add_text("constants", str(c).replace("\n","  \n"))# uses markdown

        self.c, self.writer = c, writer

    def _print_summary(self, i, loss, rate, start):
        "Prints training summary"

        logger.info("[i: %i/%i] loss: %.4f rate: %.1f elapsed: %.2f hr %s" % (
               i,
               self.c.n_steps,
               loss,
               rate,
               (time.time()-start)/(60*60),
               self.c.run,
                ))
        self.writer.add_scalar("loss/train", loss, i)
        self.writer.add_scalar("stats/rate", rate, i)

    def _save_figs(self, i, fs):
        "Saves figures"

        if self.c.clear_output: IPython.display.clear_output(wait=True)
        for name,f in fs:
            if self.c.save_figures:
                f.savefig(self.c.summary_out_dir+f"{name}_{i:08d}.png",
                          bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
            self.writer.add_figure(name, f, i, close=False)
        if self.c.show_figures: plt.show()
        else: plt.close("all")

    def _save_model(self, i, model):
        "Saves a model"

        model = jax.tree_map(lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x, model)# convert jax arrays to np
        with open(self.c.model_out_dir+f"model_{i:08d}.jax", "wb") as f:
            pickle.dump(model, f)

    def train(self):

        raise NotImplementedError
    

'''
LABELLING CONVENTIONS: 
    xd = dimensionality of point
    ud = dimensionality of solution
    dims = (ud, xd)
    n = number of points
    m = number of models (i.e. subdomains)
    c = number of constraints

    x = single coordinate (xd)
    x_batch = batch of coordinates (n, xd)
    uj = solution and gradient component list

    j = index in uj
    im = index of model
    ip = index of point
    ic = index of constraint
    i = generic index

    nm = shape of rectangular DDs
    ii = for grid index in nm
'''

def tree_map_dicts(f, *trees):
    "Apply function to top-level dicts in tree(s)"

    is_dict = lambda x: isinstance(x, dict)
    def apply(leaf, *leaves):
        if is_dict(leaf):# if top-level dict
            return f(leaf, *leaves)
        else:
            return leaf# if leaf (i.e. at bottom of tree), return first tree's leaf only (!)
    return jax.tree_util.tree_map(apply, *trees, is_leaf=is_dict)# stop traverse on top-level dicts

def get_jmaps(required_ujs):
    "Generate tree for computing chained jacobians"

    logger.debug("get_jmaps")

    # build tree of required gradients
    tree = {}
    for iu,ixs in required_ujs:
        # iu: The nth output function u
        # ixs: The sequence of variable indices to be differentiated
        t = tree
        for ix in ixs:
            if ix not in t:
                t[ix] = {}
            t = t[ix]

    # parse tree nodes
    def get_nodes(t, n=(), ks=()):
        ni = len(n)-1 + 1# index of parent node (including u at start)
        for k in t:
            ks_ = ks+(k,)
            if t[k]:
                n += (((ni,k),ks_,0),)# node
                n = get_nodes(t[k], n, ks_)
            else:
                n += (((ni,k),ks_,1),)# leaf
        return n

    # list of chained grad functions
    nodes = get_nodes(tree)
    logger.debug(nodes)

    # list of grad functions to evaluate
    leaves = tuple((i + 1, node[1]) for i,node in enumerate(nodes) if node[2])
    if not leaves:
        leaves = ((0,()),)# special case where only solution required. tree/nodes are empty in this case
    logger.debug(leaves)

    # get map between required_ujs and list of chained gradients
    jac_is = ()# il (leaf index), io (order index), iu (u index)
    for iu,ixs in required_ujs:
        io = len(ixs)
        il = [leaf[1][:io] for leaf in leaves].index(ixs)# also works for 0,()
        jac_is += ((il, io, iu),)
    logger.debug(jac_is)

    return nodes, leaves, jac_is


# JITTED FUNCTIONS
def FBPINN_model_inner(params, x, norm_fn, network_fn, unnorm_fn, window_fn):
    x_norm = norm_fn(params, x)# normalise
    u_raw = network_fn(params, x_norm)# network
    u = unnorm_fn(params, u_raw)# unnormalise
    w = window_fn(params, x)# window
    return u*w, w, u_raw

def PINN_model_inner(all_params, x, norm_fn, network_fn, unnorm_fn):
    x_norm = norm_fn(all_params, x)# normalise
    u_raw = network_fn(all_params, x_norm)# network
    u = unnorm_fn(u_raw)# unnormalise
    return u, u_raw

def FBPINN_model(all_params, x_batch, takes, model_fns, verbose=True):
    "Defines FBPINN model"

    norm_fn, network_fn, unnorm_fn, window_fn, constraining_fn = model_fns
    m_take, n_take, p_take, np_take, npou = takes

    # take x_batch
    x_take = x_batch[n_take]# (s, xd)
    log_ = logger.info if verbose else logger.debug
    log_("x_batch")
    log_(str_tensor(x_batch))# (n, xd)
    log_("x_take")
    log_(str_tensor(x_take))

    # take subdomain params
    d = all_params
    all_params_take = {t_k: {cl_k: {k: jax.tree_map(lambda p:p[m_take], d[t_k][cl_k][k]) if k=="subdomain" else d[t_k][cl_k][k]
        for k in d[t_k][cl_k]}
        for cl_k in d[t_k]}
        for t_k in ["static", "trainable"]}
    f = {t_k: {cl_k: {k: jax.tree_map(lambda p: 0, d[t_k][cl_k][k]) if k=="subdomain" else jax.tree_map(lambda p: None, d[t_k][cl_k][k])
        for k in d[t_k][cl_k]}
        for cl_k in d[t_k]}
        for t_k in ["static", "trainable"]}
    logger.debug("all_params")
    logger.debug(jax.tree_map(lambda x: str_tensor(x), all_params))
    logger.debug("all_params_take")
    logger.debug(jax.tree_map(lambda x: str_tensor(x), all_params_take))
    logger.debug("vmap f")
    logger.debug(f)

    # batch over parameters and points
    us, ws, us_raw = vmap(FBPINN_model_inner, in_axes=(f,0,None,None,None,None))(all_params_take, x_take, norm_fn, network_fn, unnorm_fn, window_fn)# (s, ud)
    logger.debug("u")
    logger.debug(str_tensor(us))

    # apply POU and sum
    u = jnp.concatenate([us, ws], axis=1)# (s, ud+1)
    u = jax.ops.segment_sum(u, p_take, indices_are_sorted=False, num_segments=len(np_take))# (_, ud+1)
    wp = u[:,-1:]
    u = u[:,:-1]/wp
    logger.debug(str_tensor(u))
    u = jax.ops.segment_sum(u, np_take, indices_are_sorted=False, num_segments=len(x_batch))# (n, ud)
    logger.debug(str_tensor(u))
    u = u/npou
    logger.debug(str_tensor(u))

    return u, wp, us, ws, us_raw

def PINN_model(all_params, x_batch, model_fns, verbose=True):
    "Defines PINN model"

    norm_fn, network_fn, unnorm_fn, constraining_fn = model_fns
    log_ = logger.info if verbose else logger.debug
    log_("x_batch")
    log_(str_tensor(x_batch))# (n, xd)

    # batch over parameters and points
    u, u_raw = vmap(PINN_model_inner, in_axes=(None,0,None,None,None))(all_params, x_batch, norm_fn, network_fn, unnorm_fn)# (n, ud)
    logger.debug("u")
    logger.debug(str_tensor(u))

    return u, u_raw

def FBPINN_forward(all_params, x_batch, takes, model_fns, jmaps):
    "Computes gradients of FBPINN model"

    # isolate model function
    def u(x_batch):
        return FBPINN_model(all_params, x_batch, takes, model_fns)[0], ()
    return _get_ujs(x_batch, jmaps, u)

def PINN_forward(all_params, x_batch, model_fns, jmaps):
    "Computes gradients of PINN model"

    # isolate model function
    def u(x_batch):
        return PINN_model(all_params, x_batch, model_fns)[0], ()
    return _get_ujs(x_batch, jmaps, u)

def _get_ujs(x_batch, jmaps, u):

    nodes, leaves, jac_is = jmaps
    vs = jnp.tile(jnp.eye(x_batch.shape[1]), (x_batch.shape[0],1,1))

    # chain required jacobian functions
    fs = [u]
    for (ni, ix), _, _ in nodes:
        fs.append(jacfwd(fs[ni], vs[:,ix]))

    # evaluate required jacobian functions
    jacs = []
    for ie,_ in leaves:
        fin, jac = fs[ie](x_batch)
        jacs.append(jac+(fin,))

    # index required jacobians
    ujs = [jacs[il][io][:,iu:iu+1] for il,io,iu in jac_is]

    logger.debug("fs")
    logger.debug(fs)
    logger.debug("jacs")
    for jac in jacs: logger.debug([j.shape for j in jac])# (n, ud)
    logger.debug("ujs")
    for uj in ujs: logger.debug(str_tensor(uj))

    return ujs

def jacfwd(f, v):
    "Computes jacobian for single x, for all y, fully chained"
    def jacfun(x):
        y, j, aux = jvp(f, (x,), (v,), has_aux=True)
        aux = aux + (y,)
        return j, aux
    return jacfun

def FBPINN_loss(active_params, fixed_params, static_params, takess, constraints, model_fns, jmapss, loss_fn):

    # add fixed params to active, recombine all_params
    d, da = active_params, fixed_params
    trainable_params = {cl_k: {k: jax.tree_map(lambda p1, p2:jnp.concatenate([p1,p2],0), d[cl_k][k], da[cl_k][k]) if k=="subdomain" else d[cl_k][k]
        for k in d[cl_k]}
        for cl_k in d}
    all_params = {"static":static_params, "trainable":trainable_params}

    # run FBPINN for each constraint, with shared params
    constraints_ = []
    for takes, jmaps, constraint in zip(takess, jmapss, constraints):
        logger.debug("constraint")
        for c_ in constraint:
            logger.debug(str_tensor(c_))
        x_batch = constraint[0]
        ujs = FBPINN_forward(all_params, x_batch, takes, model_fns, jmaps)
        constraints_.append(constraint+ujs)
    return loss_fn(all_params, constraints_)

def PINN_loss(active_params, static_params, constraints, model_fns, jmapss, loss_fn):

    # recombine all_params
    all_params = {"static":static_params, "trainable":active_params}

    # run PINN for each constraint, with shared params
    constraints_ = []
    for jmaps, constraint in zip(jmapss, constraints):
        logger.debug("constraint")
        for c_ in constraint:
            logger.debug(str_tensor(c_))
        x_batch = constraint[0]
        ujs = PINN_forward(all_params, x_batch, model_fns, jmaps)
        constraints_.append(constraint+ujs)
    return loss_fn(all_params, constraints_)

@partial(jit, static_argnums=(0, 5, 8, 9, 10))
def FBPINN_update(optimiser_fn, active_opt_states,
                  active_params, fixed_params, static_params_dynamic, static_params_static,
                  takess, constraints, model_fns, jmapss, loss_fn):
    # recombine static params
    static_params = combine(static_params_dynamic, static_params_static)
    # update step
    lossval, grads = value_and_grad(FBPINN_loss, argnums=0)(
        active_params, fixed_params, static_params, takess, constraints, model_fns, jmapss, loss_fn)
    updates, active_opt_states = optimiser_fn(grads, active_opt_states, active_params)
    active_params = optax.apply_updates(active_params, updates)
    return lossval, active_opt_states, active_params

@partial(jit, static_argnums=(0, 4, 6, 7, 8))
def PINN_update(optimiser_fn, active_opt_states,
                active_params, static_params_dynamic, static_params_static,
                constraints, model_fns, jmapss, loss_fn):
    # recombine static params
    static_params = combine(static_params_dynamic, static_params_static)
    # update step
    lossval, grads = value_and_grad(PINN_loss, argnums=0)(
        active_params, static_params, constraints, model_fns, jmapss, loss_fn)
    updates, active_opt_states = optimiser_fn(grads, active_opt_states, active_params)
    active_params = optax.apply_updates(active_params, updates)
    return lossval, active_opt_states, active_params


# For fast test inference only
@partial(jax.jit, static_argnums=(1,4,5))
def _FBPINN_model_jit(all_params_dynamic, all_params_static, x_batch, takes, model_fns, verbose):
    all_params = combine(all_params_dynamic, all_params_static)
    return FBPINN_model(all_params, x_batch, takes, model_fns, verbose)
def FBPINN_model_jit(all_params, x_batch, takes, model_fns, verbose=True):
    all_params_dynamic, all_params_static = partition(all_params)
    return _FBPINN_model_jit(all_params_dynamic, all_params_static, x_batch, takes, model_fns, verbose)

@partial(jax.jit, static_argnums=(1,3,4))
def _PINN_model_jit(all_params_dynamic, all_params_static, x_batch, model_fns, verbose):
    all_params = combine(all_params_dynamic, all_params_static)
    return PINN_model(all_params, x_batch, model_fns, verbose)
def PINN_model_jit(all_params, x_batch, model_fns, verbose=True):
    all_params_dynamic, all_params_static = partition(all_params)
    return _PINN_model_jit(all_params_dynamic, all_params_static, x_batch, model_fns, verbose)


def get_inputs(x_batch, active, all_params, decomposition):
    "Get the inputs to the FBPINN model based on x_batch and the active models"

    # get the ims inside x_batch
    n_take, m_take, training_ims = decomposition.inside_points(all_params, x_batch)# (nc, m)

    # get active_ims and fixed_ims
    # scheduler should return
    # 0 = inactive (but still trained if it intersects with active models)
    # 1 = active
    # 2 = fixed
    # now modify active to
    # 0 = discard (not in current training points)
    # 1 = active
    # 2 = fixed
    active = jnp.array(active).copy()
    assert jnp.isin(active, jnp.array([0,1,2])).all()
    assert active.shape == (all_params["static"]["decomposition"]["m"],)

    active = active.at[active==0].set(1)# set inactive models to active
    mask = jnp.zeros_like(active)# mask out models in training points
    mask = mask.at[training_ims].set(1)
    active = active*mask
    ims_ = jnp.arange(all_params["static"]["decomposition"]["m"])
    active_ims = ims_[active==1]# assume unsorted
    fixed_ims = ims_[active==2]
    logger.debug("updated active")
    logger.debug(active)
    logger.debug("active_ims")
    logger.debug(active_ims)
    logger.debug("fixed_ims")
    logger.debug(fixed_ims)

    # note, numbers in all_ims == numbers in training_ims == numbers in m_take
    # which also means we need all m_take points above
    all_ims = jnp.concatenate([active_ims, fixed_ims])

    # re-index m_take to all_ims index
    inv = jnp.zeros(all_params["static"]["decomposition"]["m"], dtype=int)
    inv = inv.at[all_ims].set(jnp.arange(len(all_ims)))# assumes all_ims is unique
    m_take = inv[m_take]

    # (!) note: make sure n_take, pous (and therefore p_take / np_take) are sorted - makes segment_sum quicker
    logger.debug("takes")
    logger.debug(str_tensor(m_take))
    logger.debug(str_tensor(n_take))

    # get POUs
    pous = all_params["static"]["decomposition"]["subdomain"]["pou"][all_ims].astype(int)
    np = jnp.stack([n_take, pous[m_take,0]], axis=-1).astype(int)# points and pous
    logger.debug(str_tensor(np))
    npu,p_take = jnp.unique(np, axis=0, return_inverse=True)# unique points and pous (sorted), point-pou takes
    np_take = npu[:,0]
    logger.debug(str_tensor(p_take))
    logger.debug(str_tensor(np_take))
    npou = len(jnp.unique(all_params["static"]["decomposition"]["subdomain"]["pou"].astype(int)))# global npou
    logger.debug(f"Total number of POUs: {npou}")

    takes = (m_take, n_take, p_take, np_take, npou)

    # cut active and fixed parameter trees
    def cut_active(d):
        "Cuts active_ims from param dict"
        return {cl_k: {k: jax.tree_map(lambda p:p[active_ims], d[cl_k][k]) if k=="subdomain" else d[cl_k][k]
                for k in d[cl_k]}
                for cl_k in d}
    def cut_fixed(d):
        "Cuts fixed_ims from param dict"
        return {cl_k: {k: jax.tree_map(lambda p:p[fixed_ims],  d[cl_k][k]) if k=="subdomain" else d[cl_k][k]
                for k in d[cl_k]}
                for cl_k in d}
    def cut_all(d):
        "Cuts all_ims from param dict"
        return {cl_k: {k: jax.tree_map(lambda p:p[all_ims],    d[cl_k][k]) if k=="subdomain" else d[cl_k][k]
                for k in d[cl_k]}
                for cl_k in d}
    def merge_active(da, d):
        "Merges active_ims from param dict da to d"
        for cl_k in d:
            for k in d[cl_k]:
                if k=="subdomain":
                    d[cl_k][k] = jax.tree_map(lambda pa, p: p.copy().at[active_ims].set(pa), da[cl_k][k], d[cl_k][k])
                else:
                    d[cl_k][k] = da[cl_k][k]
        return d

    return takes, all_ims, (active, cut_active, cut_fixed, cut_all, merge_active)
