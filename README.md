<p align="center">
  <h1>SegPINN-DIC</h1>
  <p align="center">
    <em>Physics-Informed Neural Network Framework for Digital Image Correlation</em>
  </p>
  <p align="center">
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python version"></a>
    <a href="https://github.com/lbd-hfut/SegPINN-DIC/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"></a>
    <a href="https://github.com/lbd-hfut/SegPINN-DIC"><img src="https://img.shields.io/badge/status-alpha-orange.svg" alt="Development Status"></a>
  </p>
</p>

---

**SegPINN-DIC** is a physics-informed neural network (PINN) framework for digital image correlation (DIC). It combines classical subset-based DIC seed matching with PINN-based full-field displacement inference, optionally using feature-based domain decomposition (FBPINN) to handle large domains and complex deformations via parallel subdomain networks.

---

## Features

- **Classical DIC seed matching** — Multi-scale normalized cross-correlation (NCC) for integer-pixel initialization, followed by inverse compositional Gauss-Newton (IC-GN) sub-pixel refinement with B-spline interpolation.
- **PINN full-field DIC** — Learn continuous displacement fields `u(x,y)` and `v(x,y)` directly from image intensity conservation using neural networks.
- **FBPINN domain decomposition** — Split the image domain into overlapping rectangular subdomains, each solved by its own network, combined via partition of unity. Supports multi-scale decomposition and gradual subdomain activation schedulers.
- **Multiple network architectures** — FCN, AdaptiveFCN, SIREN, AdaptiveSIREN, ResNet, AdaptiveResNet, and FourierNet (multi-scale Fourier features).
- **Multiple loss functions** — MSE and ZNSSD (zero-mean normalized sum of squared differences, robust to illumination variations).
- **B-spline image interpolation** — Degree 1/3/5 B-spline with JIT-compiled FFT-based coefficient precomputation.
- **Multi-ROI support** — Automatically detects and processes disconnected regions of interest from a labeled mask.
- **Strain computation** — Automatically computes full-field strain components (`exx`, `exy`, `eyy`) from displacement gradients.
- **GPU / CPU toggle** — Simple flag to switch between GPU and CPU backends via JAX.
- **Comprehensive outputs** — Results saved as `.mat` files, visualization figures, TensorBoard summaries, and model checkpoints.

---

## Pipeline

```
                    ┌──────────────────────────────┐
                    │   Config Files (DIC + Seed)  │
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │      DIC_analysis.main()      │
                    │  - Parse configs              │
                    │  - Load image dataset         │
                    │  - Initialize seed calculator │
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │        ImgDataset             │
                    │  - Read reference + mask      │
                    │  - Build B-spline buffers     │
                    │  - Label multi-ROI masks      │
                    └──────────┬───────────────────┘
                               │
          ┌────────────────────┼──────────────────────┐
          ▼                    ▼                      ▼
  ┌────────────────┐  ┌──────────────────┐  ┌──────────────────┐
  │  Seed Matching │  │   PINN Trainer   │  │ FBPINN Trainer   │
  │  (DIC_seedcalc)│  │ (single network) │  │ (subdomain nets) │
  │  • K-means     │  │  • Adam optimize │  │  • Scheduler     │
  │  • NCC integer │  │  • MSE / ZNSSD  │  │  • Partition of  │
  │  • IC-GN sub-  │  │                   │  │    unity         │
  │    pixel       │  └────────┬─────────┘  └────────┬─────────┘
  └────────┬───────┘           │                      │
           │          ┌────────▼──────────────────────▼─────────┐
           └─────────►│   Predict u, v, exx, exy, eyy          │
                      │   (DIC_trainers)                        │
                      └────────────────┬───────────────────────┘
                                       │
                      ┌────────────────▼───────────────────────┐
                      │           Save Results                 │
                      │  • .mat files   • Figures             │
                      │  • TensorBoard  • Model checkpoints   │
                      └───────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python >= 3.10
- JAX (with or without GPU support)

### Install from source

```bash
git clone https://github.com/lbd-hfut/SegPINN-DIC.git
cd SegPINN-DIC
pip install .
```

### Install in development mode

```bash
pip install -e .
```

---

## Quick Start

### 1. Prepare your data

Place images in a directory with the following naming convention (sorted alphabetically):

```
input_dir/
├── 001.bmp      # Reference image (first file)
├── 002.bmp      # Deformed image
├── ...          # More deformed images (optional)
└── 010.bmp      # ROI mask image (last file, binary)
```

The mask image should contain white pixels (255) for the region of interest and black pixels (0) for the background. Multiple disconnected ROIs are automatically detected and processed independently.

### 2. Configure

Edit `config/PINN-DIC-2D.txt`:

```ini
input_dir  = ./case/ring
output_dir = ./case/ring/results
network    = AdaptiveFCN
hidden_units = [32, 32, 32, 32]
loss_fun   = DIC_ZNSSD
adam_epochs = 100
adam_lr    = 0.01
seed_flag  = True
```

Edit `config/Seed_Configuration.txt` (optional, only used when `seed_flag = True`):

```ini
method             = Sub_pixels
seeds_number       = 128
coarse_subset_radius = 28
fine_subset_radius   = 9
max_iterations    = 50
```

### 3. Run

```python
from segpinndic.DIC_analysis import main

main(
    seed_config_path="./config/Seed_Configuration.txt",
    dic_config_path="./config/PINN-DIC-2D.txt"
)
```

Or from the command line:

```bash
python -m segpinndic.DIC_analysis
```

### 4. Outputs

Results are saved under the configured `output_dir`:

```
output_dir/
├── mats/          # .mat files with u, v, exx, exy, eyy
├── figs/          # Visualization figures
├── summaries/     # TensorBoard event files
├── models/        # Model checkpoint (.jax files)
└── seed/          # Seed matching visualization
```

---

## Configuration

### DIC Solver Config (`config/PINN-DIC-2D.txt`)

| Key | Type | Default | Description |
|---|---|---|---|
| `input_dir` | `str` | — | Path to input image directory |
| `output_dir` | `str` | — | Path to output results |
| `n_subdomains` | `list[int]` | `[2, 2]` | Number of subdomains in (nx, ny) |
| `train_schedulers` | `str` | `AllActiveSchedulerND` | Scheduler type |
| `hidden_units` | `list[int]` | `[32,32,32,32]` | Neurons per hidden layer |
| `network` | `str` | `AdaptiveFCN` | Network architecture |
| `spline_degree` | `int` | `5` | B-spline degree (1, 3, or 5) |
| `loss_fun` | `str` | `DIC_ZNSSD` | Loss function (`DIC_MSE` or `DIC_ZNSSD`) |
| `adam_epochs` | `int` | `100` | Number of Adam training epochs |
| `seed_flag` | `bool` | `True` | Enable seed point initialization |
| `seed_train_epochs` | `int` | `0` | Seed pre-training epochs |
| `adam_lr` | `float` | `0.01` | Adam learning rate |
| `summary_freq` | `int` | `10` | Loss print frequency (epochs) |
| `test_freq` | `int` | `1000` | Test loss frequency (epochs) |
| `model_save_freq` | `int` | `10000` | Model save frequency (epochs) |
| `show_figures` | `bool` | `False` | Display result figures |
| `save_figures` | `bool` | `True` | Save result figures |

### Seed Config (`config/Seed_Configuration.txt`)

| Key | Type | Default | Description |
|---|---|---|---|
| `method` | `str` | `Sub_pixels` | Matching method (`Integer_pixels` or `Sub_pixels`) |
| `seeds_number` | `int` | `128` | Number of seed points |
| `max_workers` | `int` | `4` | Parallel threads |
| `coarse_subset_radius` | `int` | `28` | Coarse NCC subset radius (pixels) |
| `fine_subset_radius` | `int` | `9` | Fine IC-GN subset radius (pixels) |
| `max_iterations` | `int` | `50` | IC-GN max iterations |
| `cutoff_diffnorm` | `float` | `1e-5` | IC-GN convergence tolerance |
| `plot_seed_flage` | `bool` | `True` | Enable seed match visualization |

---

## Project Structure

```
segpinndic/
├── DIC_analysis.py        # Main pipeline orchestrator
├── DIC_config.py          # Configuration file parsers
├── DIC_constants.py       # Global solver constants & wired components
├── DIC_decompositions.py  # Domain decomposition (rectangular, multilevel)
├── DIC_domains.py         # Domain definitions
├── DIC_importlib.py       # Centralized imports (JAX, numpy, etc.) & GPU/CPU control
├── DIC_networks.py        # Neural network architectures (FCN, SIREN, ResNet, FourierNet, etc.)
├── DIC_plot_trainer.py    # Result visualization
├── DIC_problem.py         # PINN loss functions (MSE, ZNSSD)
├── DIC_readImg.py         # Image I/O, B-spline buffers, ImgDataset
├── DIC_schedulers.py      # Active subdomain schedulers
├── DIC_seedcalc.py        # Seed point matching (NCC + IC-GN)
├── DIC_trainers.py        # PINN & FBPINN training loops
├── DIC_windows.py         # Window functions for partition of unity
└── utils/
    ├── io.py              # File/directory I/O helpers
    ├── jax_util.py        # JAX pytree partitioning utilities
    ├── logger.py          # Logging setup
    └── other.py           # Misc utilities (Timer, Cycle, DictToObj, save_gif)
```

---

## Test Cases

Pre-configured datasets are available under `case/`:

| Case | Description |
|---|---|
| `ring/` | Ring test — 10 images of a deforming ring |
| `scale/` | Scale variation study — 20 datasets with scale factors from 0.001× to 80× |
| `speckle_quality/` | Speckle quality analysis with varying speckle parameters |
| `case9/` | Case 9 dataset (3 images) |
| `DIC_challenge_star/` | DIC Challenge star pattern datasets (6 variants) |
| `illumination_Variation/` | Illumination variation tests (global and local) |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{SegPINN-DIC,
  author = {Boda Li},
  title = {SegPINN-DIC: Physics-Informed Neural Network Framework for Digital Image Correlation},
  year = {2025},
  url = {https://github.com/lbd-hfut/SegPINN-DIC}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Built with [JAX](https://github.com/google/jax), [Optax](https://github.com/google-deepmind/optax), and inspired by the FBPINN approach from [mikkelbueholm/FBPINNs](https://github.com/mikkelbueholm/FBPINNs).
