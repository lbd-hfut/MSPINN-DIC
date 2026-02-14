from segpinndic.DIC_importlib import jax, jnp, np, plt

from segpinndic.utils.other import colors

def _lim(v, factor=1.1):
    mi, ma = v.min(0), v.max(0)
    c = (mi+ma)/2
    w = factor*(ma-mi)/2
    return (c-w, c+w)

def _plot_setup(x_batch_test, u_exact):
    # get general setup for plotting
    xlim, ulim = _lim(x_batch_test), _lim(u_exact)
    return xlim, ulim

def _to_numpy(f):
    # converts jnp arrays to np arrays
    def wrapper(*args):
        args = jax.tree_map(lambda a: np.array(a) if isinstance(a, jnp.ndarray) else a, args)
        return f(*args)
    return wrapper

@_to_numpy
def plot_1D_FBPINN(x_batch_test, u_exact, u_test, us_test, ws_test, us_raw_test, x_batch, all_params, i, active, decomposition, n_test):

    xlim, ulim = _plot_setup(x_batch_test, u_exact)

    f = plt.figure(figsize=(8,4*10/3))

    # plot domain + x_batch
    plt.subplot(4,1,1)
    plt.title(f"[{i}] Domain decomposition")
    plt.scatter(x_batch[:,0], 0.1*np.ones_like(x_batch)[:,0], alpha=0.5, color="k", s=40)
    decomposition.plot(all_params, active=active, create_fig=False)
    plt.xlim(*xlim)

    plt.subplot(4,1,2)
    plt.title(f"[{i}] POU window functions")
    for im in range(all_params["static"]["decomposition"]["m"]):
        plt.plot(x_batch_test[:,0], ws_test[im,:,0], color=colors[im])
    plt.xlim(*xlim)

    # plot full + individual solutions
    plt.subplot(4,1,3)
    plt.title(f"[{i}] Full and individual solutions")
    for im in range(all_params["static"]["decomposition"]["m"]):
        plt.plot(x_batch_test[:,0], us_test[im,:,0], color=colors[im])
    plt.plot(x_batch_test[:,0], u_exact[:,0], lw=4, color="tab:grey", label="Ground truth")
    plt.plot(x_batch_test[:,0], u_test[:,0], color="k", label="FBPINN")
    plt.legend()
    plt.xlim(*xlim)
    plt.ylim(*ulim)

    # plot raw solutions
    plt.subplot(4,1,4)
    plt.title(f"[{i}] Raw solutions")
    for im in range(all_params["static"]["decomposition"]["m"]):
        plt.plot(x_batch_test[:,0], us_raw_test[im,:,0], color=colors[im])
    plt.xlim(*xlim)

    plt.tight_layout()

    return (("test",f),)

@_to_numpy
def plot_1D_PINN(x_batch_test, u_exact, u_test, u_raw_test, x_batch, all_params, i, n_test):

    xlim, ulim = _plot_setup(x_batch_test, u_exact)

    f = plt.figure(figsize=(8,10))

    # plot x_batch
    plt.subplot(3,1,1)
    plt.title(f"[{i}] Training points")
    plt.scatter(x_batch[:,0], 0.1*np.ones_like(x_batch)[:,0], alpha=0.5, color="k", s=40)
    plt.xlim(*xlim)

    # plot full solution
    plt.subplot(3,1,2)
    plt.title(f"[{i}] Full solution")
    plt.plot(x_batch_test[:,0], u_exact[:,0], lw=4, color="tab:grey", label="Ground truth")
    plt.plot(x_batch_test[:,0], u_test[:,0], color="k", label="PINN")
    plt.legend()
    plt.xlim(*xlim)
    plt.ylim(*ulim)

    # plot raw solution
    plt.subplot(3,1,3)
    plt.title(f"[{i}] Raw solution")
    plt.plot(x_batch_test[:,0], u_raw_test[:,0], color="k")
    plt.xlim(*xlim)

    plt.tight_layout()

    return (("test",f),)

def _plot_test_im(u_test, xlim, ulim, n_test, it=None):
    u_test = u_test.reshape(n_test)
    if it is not None:
        u_test = u_test[:,:,it]# for 3D
    plt.imshow(u_test.T,# transpose as jnp.meshgrid uses indexing="ij"
               origin="lower", extent=(xlim[0][0], xlim[1][0], xlim[0][1], xlim[1][1]),
               cmap="viridis", vmin=ulim[0], vmax=ulim[1])
    plt.colorbar()
    plt.xlim(xlim[0][0], xlim[1][0])
    plt.ylim(xlim[0][1], xlim[1][1])
    plt.gca().set_aspect("equal")

@_to_numpy
def plot_2D_FBPINN(x_batch_test, u_exact, u_test, us_test, ws_test, us_raw_test, x_batch, all_params, i, active, decomposition, n_test):

    xlim, ulim = _plot_setup(x_batch_test, u_exact)
    xlim0 = x_batch_test.min(0), x_batch_test.max(0)

    f = plt.figure(figsize=(8,10))

    # plot domain + x_batch
    plt.subplot(3,2,1)
    plt.title(f"[{i}] Domain decomposition")
    plt.scatter(x_batch[:,0], x_batch[:,1], alpha=0.5, color="k", s=1)
    decomposition.plot(all_params, active=active, create_fig=False)
    plt.xlim(xlim[0][0], xlim[1][0])
    plt.ylim(xlim[0][1], xlim[1][1])
    plt.gca().set_aspect("equal")

    # plot full solutions
    plt.subplot(3,2,2)
    plt.title(f"[{i}] Difference")
    _plot_test_im(u_exact - u_test, xlim0, ulim, n_test)

    plt.subplot(3,2,3)
    plt.title(f"[{i}] Full solution")
    _plot_test_im(u_test, xlim0, ulim, n_test)

    plt.subplot(3,2,4)
    plt.title(f"[{i}] Ground truth")
    _plot_test_im(u_exact, xlim0, ulim, n_test)

    # plot raw hist
    plt.subplot(3,2,5)
    plt.title(f"[{i}] Raw solutions")
    plt.hist(us_raw_test.flatten(), bins=100, label=f"{us_raw_test.min():.1f}, {us_raw_test.max():.1f}")
    plt.legend(loc=1)
    plt.xlim(-5,5)

    plt.tight_layout()

    return (("test",f),)

@_to_numpy
def plot_2D_PINN(x_batch_test, u_exact, u_test, u_raw_test, x_batch, all_params, i, n_test):

    xlim, ulim = _plot_setup(x_batch_test, u_exact)
    xlim0 = x_batch.min(0), x_batch.max(0)

    f = plt.figure(figsize=(8,10))

    # plot x_batch
    plt.subplot(3,2,1)
    plt.title(f"[{i}] Training points")
    plt.scatter(x_batch[:,0], x_batch[:,1], alpha=0.5, color="k", s=1)
    plt.xlim(xlim[0][0], xlim[1][0])
    plt.ylim(xlim[0][1], xlim[1][1])
    plt.gca().set_aspect("equal")

    # plot full solution
    plt.subplot(3,2,2)
    plt.title(f"[{i}] Difference")
    _plot_test_im(u_exact - u_test, xlim0, ulim, n_test)

    plt.subplot(3,2,3)
    plt.title(f"[{i}] Full solution")
    _plot_test_im(u_test, xlim0, ulim, n_test)

    plt.subplot(3,2,4)
    plt.title(f"[{i}] Ground truth")
    _plot_test_im(u_exact, xlim0, ulim, n_test)

    # plot raw hist
    plt.subplot(3,2,5)
    plt.title(f"[{i}] Raw solution")
    plt.hist(u_raw_test.flatten(), bins=100, label=f"{u_raw_test.min():.1f}, {u_raw_test.max():.1f}")
    plt.legend(loc=1)
    plt.xlim(-5,5)

    plt.tight_layout()

    return (("test",f),)

_plotters = {
    "FBPINN":{1: plot_1D_FBPINN,
              2: plot_2D_FBPINN
        },
    "PINN":  {1: plot_1D_PINN,
              2: plot_2D_PINN
        },
    }

def plot(trainer, dims, *args):
    "Plots FBPINN and PINN results"

    nx = dims[1]
    if trainer in _plotters and nx in _plotters[trainer]:
        return _plotters[trainer][nx](*args)
    else:
        return ()# TODO: add higher-dim plots