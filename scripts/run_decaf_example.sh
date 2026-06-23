#!/bin/bash
# Run DeCAF end-to-end on the bundled protein-ligand cofolding example.
# Downloads an MSA via the public ColabFold server, then runs few-step
# DecafSampler inference and writes a predicted structure CIF.
#
# Prerequisites:
#   - The `boltz` package installed (editable) in your environment.
#   - The DeCAF-Pearl checkpoint downloaded locally. Get it from Hugging Face:
#       hf download gianscarpe/decaf decaf_ckpt.ckpt --local-dir .
#     (https://huggingface.co/gianscarpe/decaf) and point CHECKPOINT at it
#     (default: /tmp/decaf_ckpt.ckpt).
#   - Internet access for the MSA server (api.colabfold.com).
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
# and a CIF at <out_dir>/boltz_results_protlig_msa_server/predictions/.../*_model_0.cif
set -euo pipefail

# Resolve repo root so the example's relative paths work.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Always run THIS repo's `boltz` code, even if another `boltz`/`boltz-flowmap`
# package is installed in the environment. Without this, `python -m boltz.main`
# may resolve to a different install that does not understand the `decaf_head`
# checkpoint naming, silently drop the trained DeCAF head, and fall back to the
# teacher diffusion model — producing garbage few-step predictions.
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

CHECKPOINT="${1:-${CHECKPOINT:-/tmp/decaf_ckpt.ckpt}}"
OUT_DIR="${2:-./decaf_example_out}"
INPUT="examples/protlig_msa_server.yaml"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found at '$CHECKPOINT'."
    echo "Download it from Hugging Face: hf download gianscarpe/decaf decaf_ckpt.ckpt --local-dir ."
    echo "(https://huggingface.co/gianscarpe/decaf) then pass its path:"
    echo "  bash scripts/run_decaf_example.sh /path/to/decaf_ckpt.ckpt"
    exit 1
fi

echo "Running DeCAF cofolding example (protein + SAH ligand)"
echo "  input:      $INPUT"
echo "  checkpoint: $CHECKPOINT"
echo "  out_dir:    $OUT_DIR"
echo "  MSA:        api.colabfold.com (requires internet)"

python -m boltz.main predict \
    "$INPUT" \
    --checkpoint "$CHECKPOINT" \
    --model boltz1 \
    --sampling_steps 10 \
    --diffusion_samples 5 \
    --recycling_steps 3 \
    --use_msa_server \
    --accelerator gpu \
    --out_dir "$OUT_DIR" \
    --no_kernels

echo
echo "Done. Output CIF(s):"
find "$OUT_DIR" -name "*_model_*.cif"
