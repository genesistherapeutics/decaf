#!/usr/bin/env python
"""Fuse a DeCAF checkpoint with a Boltz-1 confidence module.

DeCAF checkpoints ship the shared Boltz-1 trunk plus a distilled ``decaf_head`` but
no confidence module, so ``boltz predict`` cannot emit pLDDT / ipTM / PAE for DeCAF
samples. The Boltz-1 confidence checkpoint (``boltz1_conf.ckpt``) carries a
*byte-identical* trunk (``input_embedder`` / ``msa_module`` / ``pairformer_module``)
plus a ``confidence_module``. Because the trunk is shared, both heads can hang off a
single trunk pass:

    trunk -> decaf_head samples coordinates -> confidence_module scores them

This script builds that fused checkpoint by taking the Boltz-1 confidence state dict
and grafting on DeCAF's ``decaf_head.*`` weights (and the hparams needed to route the
DeCAF sampler). Run the result exactly like a normal DeCAF checkpoint:

    boltz predict input.yaml --model boltz1 --checkpoint merged_decaf_conf.ckpt \
        --no_kernels --output_format pdb

and confidence outputs (B-factor pLDDT + ``confidence_*.json``) come along for free
on the trunk pass DeCAF already runs.

No weights are redistributed: you supply your own local ``decaf_ckpt.ckpt`` and
``boltz1_conf.ckpt`` (the latter is fetched automatically by ``boltz`` on first use,
or from the Boltz-1 release).

Usage
-----
    python scripts/merge_decaf_confidence.py \
        --decaf   /path/to/decaf_ckpt.ckpt \
        --conf    /path/to/boltz1_conf.ckpt \
        --out     /path/to/merged_decaf_conf.ckpt

Recipe (equivalent, if you'd rather not run the script)
-------------------------------------------------------
    merged.state_dict       = boltz1_conf.state_dict + decaf's decaf_head.* weights
    merged.hyper_parameters = boltz1_conf.hyper_parameters, then overlay from decaf:
                              diffusion_type, decaf_target_type, diffusion_process_args
    (keep only state_dict / hyper_parameters / epoch / global_step /
     pytorch-lightning_version; confidence_prediction stays True from boltz1_conf)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

# hparams copied from the DeCAF checkpoint so the fused model routes the DeCAF
# sampler (see boltz.main._detect_decaf_checkpoint / Boltz1.__init__).
DECAF_HPARAM_KEYS = ("diffusion_type", "decaf_target_type", "diffusion_process_args")

# state-dict prefixes that must match between the two checkpoints for the fusion to
# be valid (the shared, byte-identical trunk + structure module).
SHARED_TRUNK_PREFIXES = (
    "input_embedder.",
    "msa_module.",
    "pairformer_module.",
    "s_init.",
    "z_init_1.",
    "z_init_2.",
)


def _load(path: Path) -> dict:
    return torch.load(str(path), map_location="cpu", weights_only=False)


def _check_trunk_matches(decaf_sd: dict, conf_sd: dict) -> None:
    """Warn/error if the shared trunk weights are not identical.

    The whole approach relies on DeCAF and boltz1_conf sharing the same trunk. If
    they diverge (e.g. mismatched model versions) the grafted confidence head would
    be scoring off a different trunk than it was trained on, so fail loudly.
    """
    shared = [
        k
        for k in conf_sd
        if k.startswith(SHARED_TRUNK_PREFIXES) and k in decaf_sd
    ]
    if not shared:
        raise SystemExit(
            "No shared trunk parameters found between the two checkpoints — are these "
            "really a DeCAF checkpoint and a Boltz-1 confidence checkpoint?"
        )
    mismatched = []
    for k in shared:
        a, b = decaf_sd[k], conf_sd[k]
        if a.shape != b.shape or not torch.equal(a.float(), b.float()):
            mismatched.append(k)
    if mismatched:
        raise SystemExit(
            f"{len(mismatched)}/{len(shared)} shared trunk tensors differ between the "
            "DeCAF and confidence checkpoints (e.g. "
            f"{mismatched[0]}). These checkpoints do not share a trunk; fusing them "
            "would score DeCAF samples with a mismatched confidence head. Aborting."
        )
    print(f"[merge] verified {len(shared)} shared trunk tensors are identical.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decaf", required=True, type=Path, help="Path to decaf_ckpt.ckpt")
    ap.add_argument("--conf", required=True, type=Path, help="Path to boltz1_conf.ckpt")
    ap.add_argument("--out", required=True, type=Path, help="Output merged checkpoint path")
    ap.add_argument(
        "--skip-trunk-check",
        action="store_true",
        help="Skip verifying the shared trunk weights are identical (not recommended).",
    )
    args = ap.parse_args()

    print(f"[merge] loading DeCAF checkpoint:      {args.decaf}")
    decaf = _load(args.decaf)
    print(f"[merge] loading confidence checkpoint: {args.conf}")
    conf = _load(args.conf)

    decaf_sd = decaf["state_dict"]
    conf_sd = conf["state_dict"]

    decaf_head = {k: v for k, v in decaf_sd.items() if k.startswith("decaf_head.")}
    if not decaf_head:
        raise SystemExit(f"No 'decaf_head.*' weights found in {args.decaf} — not a DeCAF checkpoint?")
    if not any(k.startswith("confidence_module.") for k in conf_sd):
        raise SystemExit(
            f"No 'confidence_module.*' weights found in {args.conf} — expected a "
            "Boltz-1 *confidence* checkpoint (boltz1_conf.ckpt)."
        )

    if not args.skip_trunk_check:
        _check_trunk_matches(decaf_sd, conf_sd)

    # Fused state dict: full confidence checkpoint (trunk + structure + confidence)
    # plus the distilled DeCAF sampler head.
    merged_sd = dict(conf_sd)
    merged_sd.update(decaf_head)
    print(
        f"[merge] fused state dict: {len(conf_sd)} confidence keys + "
        f"{len(decaf_head)} decaf_head keys = {len(merged_sd)} total."
    )

    # Overlay the DeCAF routing hparams onto the confidence checkpoint's hparams so
    # boltz.main detects a DeCAF checkpoint and builds the decaf_head + sampler, while
    # keeping confidence_prediction=True from the confidence checkpoint.
    merged_hp = dict(conf.get("hyper_parameters", {}))
    decaf_hp = decaf.get("hyper_parameters", {})
    for key in DECAF_HPARAM_KEYS:
        if key in decaf_hp:
            merged_hp[key] = decaf_hp[key]
    if not merged_hp.get("confidence_prediction"):
        raise SystemExit(
            "The confidence checkpoint does not have confidence_prediction=True in its "
            "hparams; cannot fuse."
        )

    merged = {
        "state_dict": merged_sd,
        "hyper_parameters": merged_hp,
        "epoch": conf.get("epoch", 0),
        "global_step": conf.get("global_step", 0),
        "pytorch-lightning_version": conf.get("pytorch-lightning_version"),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[merge] writing merged checkpoint:     {args.out}")
    torch.save(merged, str(args.out))
    print(
        "[merge] done. Run with:\n"
        f"  boltz predict INPUT --model boltz1 --checkpoint {args.out} "
        "--no_kernels --output_format pdb"
    )


if __name__ == "__main__":
    main()
