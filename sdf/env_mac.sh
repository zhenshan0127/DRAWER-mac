# macOS (Apple Silicon / M-series) environment for DRAWER stage-1 (SDF reconstruction).
#
# Target: macOS 14+ (Sonoma/Sequoia), Apple M-series, 64 GB unified memory.
# Run these from the `sdf/` directory in an activated environment.
#
# This installs ONLY what stage-1 needs. tiny-cuda-nn, xformers, nvdiffrast, kaolin,
# pytorch3d and torch-scatter are CUDA-only and are intentionally SKIPPED — the code
# falls back to pure-PyTorch / MPS implementations (see ../README_MAC.md).

# Python 3.10: recent enough for a modern MPS-capable torch, old enough that the
# 2022-era pinned deps in pyproject.toml still ship macOS arm64 wheels.
conda create --name drawer_sdf_mac -y python=3.10
conda activate drawer_sdf_mac

# --- PyTorch with the native MPS (Metal) backend --------------------------------
# Use torch >= 2.7 so MPS implements the ops stage-1 needs (3D grid_sample forward,
# antialiased resize used by Marigold). The default PyPI wheel for macOS arm64 IS
# the MPS build — do NOT add a CUDA (cu118) index URL.
pip install "torch>=2.7" torchvision

# Sanity check — MPS must be available:
python -c "import torch; assert torch.backends.mps.is_available(), 'MPS unavailable: need macOS 14+ and an arm64 torch'; print('MPS OK, torch', torch.__version__)"

# --- nerfstudio fork (this repo) ------------------------------------------------
# Pure-Python deps only; no native CUDA extension is compiled.
pip install -e .

pip install torchmetrics[image]
pip install torchtyping
pip install "typeguard==2.12.1"
pip install --upgrade tyro

# --- Marigold (monocular depth + normal) ----------------------------------------
# Runs on MPS via diffusers' PyTorch SDPA attention; xformers is not installed and
# diffusers falls back automatically.
pip install accelerate==0.27.2
pip install diffusers==0.30.2
pip install tokenizers==0.15.2
pip install transformers==4.37.2
pip install omegaconf
pip install tabulate
pip install pandas
pip install scikit-learn
pip install "imageio[ffmpeg]"
pip install transformations

# NOTE: do NOT `pip install functorch` — on Apple Silicon the code uses torch.func
# (functorch was merged into torch). And do NOT install tiny-cuda-nn / xformers /
# nvdiffrast / pytorch3d / kaolin / torch-scatter — they are CUDA-only.

# Before running, route any MPS-unimplemented op to CPU instead of crashing
# (scripts/run_stage1_sdf.sh exports this for you):
#   export PYTORCH_ENABLE_MPS_FALLBACK=1
