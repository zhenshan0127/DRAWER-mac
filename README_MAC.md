# DRAWER stage-1 on Apple Silicon (MacBook M-series)

This branch (`mac-mps`) ports **stage 1** of DRAWER — `scripts/run_stage1_sdf.sh`
(Marigold monocular depth/normal → BakedSDF reconstruction → mesh extraction →
texture → pose export) — to run on a MacBook with Apple Silicon using PyTorch's
**MPS (Metal)** backend instead of CUDA.

> **Scope:** only stage 1 is ported. Stage 2+ (`fit_doors.py`, `sam_project.py`,
> `combine_separate.py`) depends on `nvdiffrast` and `kaolin`'s CUDA rasterizers,
> which have no Apple Silicon path and are **not** ported here.

## Requirements

- macOS 14+ (Sonoma/Sequoia), Apple M-series, ideally 32 GB+ unified memory
  (developed against a 64 GB M5).
- Xcode command-line tools, and `conda` (or a venv).

## Install

```bash
cd sdf
# follow the recipe (uses Python 3.10 + torch>=2.7 MPS build):
bash env_mac.sh          # or run the lines interactively so `conda activate` works
```

`env_mac.sh` installs only the stage-1 dependencies. The CUDA-only packages from
the Linux `env.sh` — **tiny-cuda-nn, xformers, nvdiffrast, kaolin, pytorch3d,
torch-scatter** — are intentionally omitted; the code falls back to pure-PyTorch
/ MPS paths.

## Run

```bash
bash scripts/run_stage1_sdf.sh
```

The script sets `PYTORCH_ENABLE_MPS_FALLBACK=1` and passes `--apple_silicon`
to Marigold automatically. Put your scene at `<repo>/<data_name>` (default
`cs_kitchen`); paths are derived from the repo root, not hard-coded.

## What changed vs. the CUDA version

All CUDA paths are **preserved** — every change is guarded so the same code still
runs unchanged on an NVIDIA box (`tcnn`/`cuda` are used when available).

| Area | Change |
| --- | --- |
| **tiny-cuda-nn** | `import tinycudann` is wrapped in `try/except` in 6 files. When absent, the SDF field and the two proposal density fields use the in-repo pure-PyTorch `HashEncoding(implementation="torch")` + `MLP` instead of the fused CUDA kernels. (`sdf_field.py`, `density_fields.py`, `encodings.py`, `nerfacto_field.py`, `instant_ngp_field.py`, `feature_grid.py`) |
| **Device selection** | `trainer.py`, `eval_utils.py` now pick `cuda → mps → cpu`. `train.py` guards `torch.cuda.set_device`; `writer.py` uses `torch.mps.synchronize()` on MPS; autocast maps `mps→cpu` (disabled anyway). |
| **Mesh extraction** | Hard-coded `.cuda()` in `marching_cubes.py` become `.to(device)` derived from the model/mask device. |
| **functorch** | `spatial_distortions.py` falls back to `torch.func` (functorch was merged into torch). |
| **torch_scatter** | Import guarded in `nerfacto.py` (dead code for bakedsdf). |
| **pymeshlab** | `pyproject.toml` bumped to `>=2023.12` (first arm64 wheels); renamed filter calls handled with a version-compatible fallback. `av` dependency removed (no arm64 wheel, unused). |
| **Marigold** | `run_stage1_sdf.sh` re-enables the two `run.py` calls and adds `--apple_silicon` (Marigold's built-in MPS flag). |

## Expectations / caveats

- **Speed.** The pure-PyTorch hash grid is several× slower per iteration than
  tiny-cuda-nn. The default `run_stage1_sdf.sh` runs **250k** iterations at 4096
  rays/batch — expect *days* on an M-series GPU. For a first run, lower
  `--trainer.max-num-iterations` (e.g. 50k–100k) and scale the three scheduler
  `*.max-steps` and the two `*-anneal-max-num-iters` flags proportionally.
- **Precision.** Training runs in fp32 (bakedsdf sets `mixed_precision=False`).
- **Checkpoints are not portable** between the tcnn and torch hash-grid
  implementations (different parameter layout). Train *and* extract on the Mac;
  don't load a CUDA-trained checkpoint here. Stage 1 trains from scratch, so this
  is a non-issue in practice.
- **Quality.** The torch HashEncoding supports linear interpolation only
  (Smoothstep is dropped); BakedSDF uses analytic autograd gradients, so geometry
  quality is essentially unchanged.
