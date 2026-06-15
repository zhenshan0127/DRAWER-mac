data_name="cs_kitchen"
# Repo root (this script lives in scripts/); data lives at <repo>/<data_name>.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir=${REPO_ROOT}/${data_name}
image_dir="images_2"
downscale_factor=2

# Training iterations: reduced from the paper's 250000 to 50000 for Apple Silicon (MPS)
# speed (a first-run setting per README_MAC.md). Raise toward 250000 for full quality
# (expect days). The scheduler/anneal steps below all track this value.
max_iters=50000

# Activate the Python env (uv venv created at the repo root by sdf/env_mac.sh).
if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi

# Apple Silicon (MPS): route any op MPS hasn't implemented to CPU instead of crashing.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Log Weights & Biases locally so no wandb.ai login is needed. To upload runs instead,
# remove this line (or set WANDB_MODE=online) and run `wandb login` first.
export WANDB_MODE=offline

# monocular depth and normal
# Skip Marigold inference if its outputs already exist — it re-processes EVERY frame and
# is very slow on MPS (no resume). Delete ${data_dir}/marigold_ft to force regeneration.
if [ -n "$(ls -A "${data_dir}/marigold_ft/depth" 2>/dev/null)" ] && \
   [ -n "$(ls -A "${data_dir}/marigold_ft/normal" 2>/dev/null)" ]; then
    echo "[stage1] Marigold depth/normal already present in ${data_dir}/marigold_ft — skipping inference."
else
    cd "${REPO_ROOT}/marigold"

    python run.py \
        --checkpoint "GonzaloMG/marigold-e2e-ft-depth" \
        --modality depth \
        --input_rgb_dir ${data_dir}/${image_dir} \
        --output_dir ${data_dir}/marigold_ft \
        --apple_silicon

    python run.py \
        --checkpoint "GonzaloMG/marigold-e2e-ft-normals" \
        --modality normals \
        --input_rgb_dir ${data_dir}/${image_dir} \
        --output_dir ${data_dir}/marigold_ft \
        --apple_silicon

    python read_marigold.py --data_dir ${data_dir}/marigold_ft
fi

# Expose depth/normal to the SDF dataparser. Replace any stale 0-byte placeholders first
# (plain `ln -s` fails with "File exists" on macOS if the target name already exists).
rm -f "${data_dir}/depth" "${data_dir}/normal"
ln -s "${data_dir}/marigold_ft/depth" "${data_dir}/depth"
ln -s "${data_dir}/marigold_ft/normal" "${data_dir}/normal"

# sdf reconstruction
cd "${REPO_ROOT}/sdf"

python scripts/train.py bakedsdf --vis wandb \
    --output-dir outputs/${data_name} --experiment-name ${data_name}_sdf_recon \
    --trainer.steps-per-eval-image 2000 --trainer.steps-per-eval-all-images $((max_iters + 1)) \
    --trainer.max-num-iterations $((max_iters + 1)) --trainer.steps-per-eval-batch $((max_iters + 1)) \
    --optimizers.fields.scheduler.max-steps ${max_iters} \
    --optimizers.field-background.scheduler.max-steps ${max_iters} \
    --optimizers.proposal-networks.scheduler.max-steps ${max_iters} \
    --pipeline.model.eikonal-anneal-max-num-iters ${max_iters} \
    --pipeline.model.beta-anneal-max-num-iters ${max_iters} \
    --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True \
    --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 24 \
    --pipeline.datamanager.train-num-rays-per-batch 4096 \
    --machine.num-gpus 1 --pipeline.model.scene-contraction-norm inf \
    --pipeline.model.mono-normal-loss-mult 0.2 \
    --pipeline.model.mono-depth-loss-mult 1.0 \
    --pipeline.model.near-plane 1e-6 \
    --pipeline.model.far-plane 100 \
    panoptic-data \
    --data ${data_dir} \
    --panoptic_data False \
    --mono_normal_data True \
    --mono_depth_data True \
    --panoptic_segment False \
    --downscale_factor ${downscale_factor} \
    --num_max_image 2000 # only use if memory is not enough

sdf_dir=outputs/${data_name}/${data_name}_sdf_recon

# extract mesh from mesh
python scripts/extract_mesh.py --load-config ${sdf_dir}/config.yml \
   --output-path ${sdf_dir}/mesh.ply \
   --bounding-box-min -2.0 -2.0 -2.0 --bounding-box-max 2.0 2.0 2.0 \
   --resolution 2048 --marching_cube_threshold 0.0035 --create_visibility_mask True --simplify-mesh True

# extract texture for mesh
mkdir -p ${sdf_dir}/texture_mesh
python scripts/texture.py --load-config ${sdf_dir}/config.yml \
   --output-dir ${sdf_dir}/texture_mesh \
   --input_mesh_filename ${sdf_dir}/mesh-simplify.ply \
   --target_num_faces 300000

# save pose for mesh
python scripts/save_pose.py \
    --ckpt_dir ${sdf_dir} \
    --save_dir ${data_dir}