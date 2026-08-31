# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
pip install -e .                    # Install in development mode
python -m segpinndic.DIC_analysis   # Run with default config paths
```

No linting or test framework is configured. The project uses `hatchling` as the build backend (`pyproject.toml`).

## GPU / CPU Toggle

Set `use_gpu = False` in `segpinndic/DIC_importlib.py` to switch to CPU via `JAX_PLATFORM_NAME`. This file also centralizes **all** third-party imports — every other module imports from here rather than directly.

## Architecture

### Data Flow

```
Config files (.txt) → DIC_analysis.main() → ImgDataset (load images + mask)
                                           → CalcSeeds (NCC + IC-GN seed matching)
                                           → Constants (wire components per ROI)
                                           → PINNTrainer or FBPINNTrainer
                                           → Save .mat files, figures, TensorBoard summaries
```

### Core Abstractions

All core classes follow a **functional-JAX pattern**: `@staticmethod` methods with explicit `params` dicts. The params tree always has the structure `{"static": {...}, "trainable": {...}}`.

| Component | File | Role |
|---|---|---|
| `Domain` | `DIC_domains.py` | Sample interior points, normalize coordinates |
| `Problem` | `DIC_problem.py` | Loss functions (`DIC_MSE`, `DIC_ZNSSD`) — sample constraints + compute loss |
| `Network` | `DIC_networks.py` | NN architectures: FCN, SIREN, ResNet, FourierNet, and their "Adaptive" variants |
| `Decomposition` | `DIC_decompositions.py` | FBPINN domain decomposition — norm/unnorm/window per subdomain, point-to-model routing |
| `ActiveScheduler` | `DIC_schedulers.py` | Controls which subdomains are active/fixed per training step |
| `Constants` | `DIC_constants.py` | Wires together domain, problem, network, decomposition, scheduler, optimizer for each ROI |

- `segpinndic/utils/jax_util.py`: `partition()` / `combine()` split params into hashable static and dynamic parts for JIT. `total_size()` counts parameters.
- `segpinndic/DIC_readImg.py`: `BufferManager` is a class-level global buffer holding images, masks, and precomputed B-spline coefficient tensors (`QKBQKT_def_DIC`). `ImgDataset` scans the input directory (first image = reference, last = mask, middle = deformed frames) and builds B-spline buffers.
- `segpinndic/DIC_seedcalc.py`: `CalcSeeds` runs K-means seed point generation, multi-scale NCC integer-pixel matching, then IC-GN sub-pixel refinement. Includes MAD-based outlier rejection.
- `segpinndic/DIC_seed_trainer.py`: Optional supervised pre-training — fits network outputs to IC-GN seed displacements before photometric DIC optimization. Has separate `train_seeds_pinn` and `train_seeds_fbpinn` paths. Automatically skipped for small deformations (`scale_uv < 5`).

### Trainer Design (`DIC_trainers.py`)

- **`PINNTrainer`**: Single network over the entire ROI. Adam + optional L-BFGS refinement. AOT-compiles the update step via `jax.jit(...).lower(...).compile()`.
- **`FBPINNTrainer`**: Multiple subdomain networks combined via partition of unity (POU). Uses `get_inputs()` to route points to active/fixed subdomains. Active subdomains change per the scheduler. Re-AOT-compiles whenever the active set changes.
- Both trainers return displacement (`u`, `v`) and strain (`exx`, `exy`, `eyy`) fields over the full ROI pixel grid.

### B-Spline Interpolation

FFT-based B-spline coefficient precomputation + efficient tensor product reconstruction. The key precomputed buffer is `QKBQKT_def_DIC`: a `(H, W, degree+1, degree+1)` tensor that enables O(1) per-pixel spline evaluation at wrapped coordinates. Degree 5 is the default; degrees 1 and 3 are also supported.

### Multi-ROI Handling

Multiple disconnected regions in the mask image are automatically labeled (via `scipy.ndimage.label`) and processed as independent ROIs. Each ROI gets its own `Constants` instance, seed points, and training. Results are assembled back into full-image arrays.

### Configuration

Two text config files with `# key: comment` / `value` format, parsed into `SimpleNamespace` objects:
- **`config/PINN-DIC-2D.txt`**: Network architecture, loss function, training hyperparameters, subdomain layout
- **`config/Seed_Configuration.txt`**: Seed matching method, number of seeds, subset radii, IC-GN parameters

## Key Notes

- **JIT requires careful static/dynamic separation**: The `partition()` / `combine()` pattern in `jax_util.py` is used throughout — static params (shapes, image data) are separated from dynamic params (trainable weights) before JIT compilation.
- **`BufferManager` is mutable global state**: It's cleared and repopulated per image pair. All modules reference it directly rather than passing it as a parameter.
- **Seed pre-training is conditionally skipped**: When `scale_uv` (estimated displacement range from seed matching) is below 5 pixels, seed pre-training is disabled regardless of config, since the network can converge from random initialization for small deformations.
- **L-BFGS refinement**: Controlled by `lbfgs_epochs` in DIC config. Runs after Adam training on all subdomains using `jaxopt.LBFGS`.
