#!/bin/bash
# Run DeCAF end-to-end on the bundled example input — a minimal smoke test
# that exercises the full prediction pipeline (load checkpoint -> featurize ->
# few-step DecafSampler inference -> write CIF).
#
# Prerequisites:
#   - The `boltz` package installed (editable) in your environment.
#   - The DeCAF-Pearl checkpoint downloaded locally. Get it from Hugging Face:
#       hf download gianscarpe/decaf decaf_ckpt.ckpt --local-dir .
#     (https://huggingface.co/gianscarpe/decaf) and point CHECKPOINT at it
#     (default: /tmp/decaf_ckpt.ckpt).
#
# Usage:
#   bash scripts/run_decaf_example.sh [checkpoint_path] [out_dir]
#
# Examples:
#   bash scripts/run_decaf_example.sh                       # uses /tmp/decaf_ckpt.ckpt
#   bash scripts/run_decaf_example.sh ./decaf_ckpt.ckpt     # custom checkpoint path
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_decaf_example.sh ./decaf_ckpt.ckpt ./out
#
# On success you should see:
#   "Detected Decaf checkpoint — using DecafSampler for inference."
#   "Number of failed examples: 0"
# and a CIF at <out_dir>/boltz_results_prot_custom_msa/predictions/.../*_model_0.cif
set -euo pipefail

# Resolve repo root so the example's relative MSA path (./examples/msa/...) works.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CHECKPOINT="${1:-${CHECKPOINT:-/tmp/decaf_ckpt.ckpt}}"
OUT_DIR="${2:-./decaf_example_out}"
INPUT="examples/prot_custom_msa.yaml"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found at '$CHECKPOINT'."
    echo "Download it from Hugging Face: hf download gianscarpe/decaf decaf_ckpt.ckpt --local-dir ."
    echo "(https://huggingface.co/gianscarpe/decaf) then pass its path:"
    echo "  bash scripts/run_decaf_example.sh /path/to/decaf_ckpt.ckpt"
    exit 1
fi

echo "Running DeCAF example prediction"
echo "  input:      $INPUT"
echo "  checkpoint: $CHECKPOINT"
echo "  out_dir:    $OUT_DIR"

python -m boltz.main predict \
    "$INPUT" \
    --checkpoint "$CHECKPOINT" \
    --model boltz1 \
    --sampling_steps 10 \
    --diffusion_samples 1 \
    --recycling_steps 1 \
    --accelerator gpu \
    --out_dir "$OUT_DIR" \
    --no_kernels

echo
echo "Done. Output CIF(s):"
find "$OUT_DIR" -name "*_model_*.cif"
