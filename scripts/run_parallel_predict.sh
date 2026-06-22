#!/bin/bash
# Run Decaf prediction in parallel across multiple GPUs.
#
# Usage:
#   bash scripts/run_parallel_predict.sh <input_dir> <output_dir> [num_gpus] [sampling_steps] [diffusion_samples] [flags...]
#
# Flags:
#   --use_potentials          Enable FK steering + gradient guidance
#   --no_sde                  Use ODE (deterministic) instead of SDE
#   --sde_gamma=<float>       γ-sampling (0=ODE, 1=full SDE, 0<γ<1=partial denoise+renoise)
#   --use_mc_grad             Enable MC-GRAD sampler (requires --use_potentials)
#   --mc_grad_particles=<int> MC-GRAD particles K (default: 4)
#   --mc_grad_snr_ratio=<float> MC-GRAD SNR ratio λ (default: 10.0)
#   --no_mc_grad_gd           Disable GD guidance steps after mc_grad correction
#   --use_mcts                Enable MCTS Full Rollout sampler
#   --mcts_simulations=<int>  MCTS simulations (default: 50)
#   --mcts_children=<int>     MCTS expansion children (default: 4)
#   --mcts_num_roots=<int>    Independent MCTS roots (default: 0)
#
# Examples:
#   # Standard SDE with FK steering
#   bash scripts/run_parallel_predict.sh /path/to/queries /path/to/output 4 10 5 --use_potentials
#
#   # γ-sampling (γ=0.5) with FK steering
#   bash scripts/run_parallel_predict.sh /path/to/queries /path/to/output 4 10 5 --use_potentials --sde_gamma=0.5
#
#   # MC-GRAD with γ-sampling
#   bash scripts/run_parallel_predict.sh /path/to/queries /path/to/output 4 10 5 --use_potentials --use_mc_grad --sde_gamma=0.7

set -euo pipefail

INPUT_DIR="${1:?Usage: $0 <input_dir> <output_dir> [num_gpus] [sampling_steps] [diffusion_samples] [--use_potentials]}"
OUTPUT_DIR="${2:?Usage: $0 <input_dir> <output_dir> [num_gpus] [sampling_steps] [diffusion_samples] [--use_potentials]}"
NUM_GPUS="${3:-4}"
SAMPLING_STEPS="${4:-10}"
DIFFUSION_SAMPLES="${5:-5}"
USE_POTENTIALS=false
NO_SDE=false
SDE_GAMMA=""
USE_MC_GRAD=false
MC_GRAD_PARTICLES=4
MC_GRAD_SNR_RATIO=10.0
NO_MC_GRAD_GD=false
USE_MCTS=false
MCTS_SIMULATIONS=50
MCTS_CHILDREN=4
MCTS_NUM_ROOTS=0
BOLTZ_GAMMA_0=""
NUM_PARTICLES=""
FK_RESAMPLING_INTERVAL=""
EXTRA_ARGS=()
for arg in "${@:6}"; do
    case "$arg" in
        --use_potentials) USE_POTENTIALS=true ;;
        --no_sde) NO_SDE=true ;;
        --sde_gamma=*) SDE_GAMMA="${arg#*=}" ;;
        --gamma_0=*) BOLTZ_GAMMA_0="${arg#*=}" ;;
        --step_scale=*) STEP_SCALE="${arg#*=}" ;;
        --noise_scale=*) NOISE_SCALE="${arg#*=}" ;;
        --diffusion_rho=*) DIFFUSION_RHO="${arg#*=}" ;;
        --use_mc_grad) USE_MC_GRAD=true ;;
        --mc_grad_particles=*) MC_GRAD_PARTICLES="${arg#*=}" ;;
        --mc_grad_snr_ratio=*) MC_GRAD_SNR_RATIO="${arg#*=}" ;;
        --no_mc_grad_gd) NO_MC_GRAD_GD=true ;;
        --num_particles=*) NUM_PARTICLES="${arg#*=}" ;;
        --fk_resampling_interval=*) FK_RESAMPLING_INTERVAL="${arg#*=}" ;;
        --use_mcts) USE_MCTS=true ;;
        --mcts_simulations=*) MCTS_SIMULATIONS="${arg#*=}" ;;
        --mcts_children=*) MCTS_CHILDREN="${arg#*=}" ;;
        --mcts_num_roots=*) MCTS_NUM_ROOTS="${arg#*=}" ;;
        *) EXTRA_ARGS+=("$arg") ;;
    esac
done
# Also handle two-arg forms (--flag value)
i=6
while [ $i -le $# ]; do
    arg="${!i}"
    case "$arg" in
        --sde_gamma) ((i++)); SDE_GAMMA="${!i}" ;;
        --gamma_0) ((i++)); BOLTZ_GAMMA_0="${!i}" ;;
        --step_scale) ((i++)); STEP_SCALE="${!i}" ;;
        --noise_scale) ((i++)); NOISE_SCALE="${!i}" ;;
        --diffusion_rho) ((i++)); DIFFUSION_RHO="${!i}" ;;
        --mc_grad_particles) ((i++)); MC_GRAD_PARTICLES="${!i}" ;;
        --mc_grad_snr_ratio) ((i++)); MC_GRAD_SNR_RATIO="${!i}" ;;
        --mcts_simulations) ((i++)); MCTS_SIMULATIONS="${!i}" ;;
        --mcts_children) ((i++)); MCTS_CHILDREN="${!i}" ;;
        --mcts_num_roots) ((i++)); MCTS_NUM_ROOTS="${!i}" ;;
        --num_particles) ((i++)); NUM_PARTICLES="${!i}" ;;
        --fk_resampling_interval) ((i++)); FK_RESAMPLING_INTERVAL="${!i}" ;;
    esac
    ((i++))
done
CHECKPOINT="${CHECKPOINT:-/tmp/decaf_ckpt.ckpt}"
PYTHON="${PYTHON:-python}"

# GPU_LIST: comma-separated physical GPU IDs (e.g. "2,3"). Defaults to 0..NUM_GPUS-1.
if [ -n "${GPU_LIST:-}" ]; then
    IFS=',' read -ra GPU_IDS <<< "$GPU_LIST"
    NUM_GPUS=${#GPU_IDS[@]}
else
    GPU_IDS=()
    for i in $(seq 0 $((NUM_GPUS - 1))); do GPU_IDS+=("$i"); done
fi

SHARD_DIR=$(mktemp -d "$OUTPUT_DIR/.shards_XXXX")

for i in $(seq 0 $((NUM_GPUS - 1))); do
    mkdir -p "$SHARD_DIR/shard_$i"
done

files=("$INPUT_DIR"/*.yaml)
n=${#files[@]}
extra_label=""
if $USE_POTENTIALS; then extra_label+=", potentials ON"; fi
if $USE_MC_GRAD; then
    extra_label+=", MC-GRAD K=${MC_GRAD_PARTICLES} snr=${MC_GRAD_SNR_RATIO}"
    if $NO_MC_GRAD_GD; then extra_label+=" (pure, no GD)"; fi
fi
if $USE_MCTS; then extra_label+=", MCTS sims=${MCTS_SIMULATIONS} children=${MCTS_CHILDREN} roots=${MCTS_NUM_ROOTS}"; fi
mode_label="SDE mode"
if $NO_SDE; then mode_label="ODE mode"; fi
if [ -n "$SDE_GAMMA" ]; then mode_label="γ=${SDE_GAMMA}"; fi
echo "Splitting $n inputs across $NUM_GPUS GPUs ($SAMPLING_STEPS steps, $DIFFUSION_SAMPLES samples, ${mode_label}${extra_label})"

for i in "${!files[@]}"; do
    shard=$((i % NUM_GPUS))
    ln -sf "${files[$i]}" "$SHARD_DIR/shard_$shard/"
done

for i in $(seq 0 $((NUM_GPUS - 1))); do
    count=$(ls "$SHARD_DIR/shard_$i" | wc -l)
    echo "  GPU $i: $count targets"
done

pids=()
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    physical_gpu="${GPU_IDS[$gpu]}"
    logfile="$OUTPUT_DIR/gpu_${gpu}.log"
    mkdir -p "$OUTPUT_DIR"
    extra_flags=()
    if $USE_POTENTIALS; then extra_flags+=(--use_potentials); fi
    if $NO_SDE; then extra_flags+=(--no_sde); fi
    if [ -n "$SDE_GAMMA" ]; then extra_flags+=(--sde_gamma "$SDE_GAMMA"); fi
    if [ -n "$BOLTZ_GAMMA_0" ]; then extra_flags+=(--gamma_0 "$BOLTZ_GAMMA_0"); fi
    if [ -n "${STEP_SCALE:-}" ]; then extra_flags+=(--step_scale "$STEP_SCALE"); fi
    if [ -n "${NOISE_SCALE:-}" ]; then extra_flags+=(--noise_scale "$NOISE_SCALE"); fi
    if [ -n "${DIFFUSION_RHO:-}" ]; then extra_flags+=(--diffusion_rho "$DIFFUSION_RHO"); fi
    if [ -n "$NUM_PARTICLES" ]; then extra_flags+=(--num_particles "$NUM_PARTICLES"); fi
    if [ -n "$FK_RESAMPLING_INTERVAL" ]; then extra_flags+=(--fk_resampling_interval "$FK_RESAMPLING_INTERVAL"); fi
    if $USE_MC_GRAD; then
        extra_flags+=(--use_mc_grad --mc_grad_particles "$MC_GRAD_PARTICLES" --mc_grad_snr_ratio "$MC_GRAD_SNR_RATIO")
        if $NO_MC_GRAD_GD; then extra_flags+=(--no_mc_grad_gd); fi
    fi
    if $USE_MCTS; then
        extra_flags+=(--use_mcts --mcts_simulations "$MCTS_SIMULATIONS" --mcts_children "$MCTS_CHILDREN")
        if [ "$MCTS_NUM_ROOTS" -gt 0 ]; then extra_flags+=(--mcts_num_roots "$MCTS_NUM_ROOTS"); fi
    fi
    CUDA_VISIBLE_DEVICES=$physical_gpu PYTHONUNBUFFERED=1 "$PYTHON" -m boltz.main predict \
        "$SHARD_DIR/shard_$gpu" \
        --checkpoint "$CHECKPOINT" \
        --model boltz1 \
        --sampling_steps "$SAMPLING_STEPS" \
        --diffusion_samples "$DIFFUSION_SAMPLES" \
        --recycling_steps 3 \
        --accelerator gpu \
        --out_dir "$OUTPUT_DIR/shard_$gpu" \
        --no_kernels \
        "${extra_flags[@]+"${extra_flags[@]}"}" \
        > "$logfile" 2>&1 &
    pids+=($!)
    echo "Launched GPU $gpu (PID ${pids[-1]}) -> $logfile"
done

echo "Waiting for all ${#pids[@]} jobs..."
failed=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "  GPU $i finished OK"
    else
        echo "  GPU $i FAILED (exit code $?)"
        failed=$((failed + 1))
    fi
done

if [ $failed -gt 0 ]; then
    echo "ERROR: $failed job(s) failed. Check logs in $OUTPUT_DIR/gpu_*.log"
    exit 1
fi

# Merge predictions into a single directory
echo "Merging predictions..."
mkdir -p "$OUTPUT_DIR/predictions"
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    pred_dir=$(find "$OUTPUT_DIR/shard_$gpu" -type d -name predictions 2>/dev/null | head -1)
    if [ -n "$pred_dir" ]; then
        for target_dir in "$pred_dir"/*/; do
            target=$(basename "$target_dir")
            ln -sfn "$target_dir" "$OUTPUT_DIR/predictions/$target"
        done
    fi
done

total=$(ls "$OUTPUT_DIR/predictions" 2>/dev/null | wc -l)
echo "Done. $total targets in $OUTPUT_DIR/predictions/"
rm -rf "$SHARD_DIR"
