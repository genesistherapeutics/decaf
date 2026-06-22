"""Load a Decaf checkpoint and run inference with the DecafSampler."""

import torch
from boltz.data import const
from boltz.model.models.boltz1 import Boltz1


def make_dummy_feats(
    batch_size: int = 1,
    n_tokens: int = 64,
    n_atoms: int = 64,
    n_msa: int = 4,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Create a minimal set of dummy features for a forward pass."""
    B, T, A = batch_size, n_tokens, n_atoms

    feats = {}

    # Token-level
    feats["res_type"] = torch.zeros(B, T, const.num_tokens, device=device)
    feats["res_type"][:, :, 0] = 1.0  # one-hot
    feats["profile"] = torch.zeros(B, T, const.num_tokens, device=device)
    feats["deletion_mean"] = torch.zeros(B, T, device=device)
    feats["pocket_feature"] = torch.zeros(B, T, len(const.pocket_contact_info), device=device)
    feats["token_pad_mask"] = torch.ones(B, T, dtype=torch.bool, device=device)
    feats["token_bonds"] = torch.zeros(B, T, T, 1, device=device)
    feats["token_index"] = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

    # Relative position features
    feats["asym_id"] = torch.zeros(B, T, dtype=torch.long, device=device)
    feats["entity_id"] = torch.zeros(B, T, dtype=torch.long, device=device)
    feats["sym_id"] = torch.zeros(B, T, dtype=torch.long, device=device)
    feats["residue_index"] = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    feats["cyclic_period"] = torch.zeros(B, T, dtype=torch.long, device=device)

    # Atom-level
    feats["atom_pad_mask"] = torch.ones(B, A, dtype=torch.bool, device=device)
    feats["ref_pos"] = torch.randn(B, A, 3, device=device)
    feats["ref_space_uid"] = torch.zeros(B, A, dtype=torch.long, device=device)
    feats["ref_charge"] = torch.zeros(B, A, device=device)
    feats["ref_element"] = torch.zeros(B, A, 128, device=device)
    feats["ref_element"][:, :, 0] = 1.0  # one-hot for element 0
    feats["ref_atom_name_chars"] = torch.zeros(B, A, 4, 64, device=device)
    feats["atom_to_token"] = torch.zeros(B, A, T, device=device)
    # Simple 1-to-1 atom-to-token mapping
    for i in range(min(A, T)):
        feats["atom_to_token"][:, i, i] = 1.0

    feats["atom_resolved_mask"] = torch.ones(B, A, dtype=torch.bool, device=device)
    feats["mol_type"] = torch.zeros(B, T, dtype=torch.long, device=device)

    # Coords (for training, not used in inference but referenced for shapes)
    feats["coords"] = torch.randn(B, 1, A, 3, device=device)

    # MSA features (msa is one-hot [B, n_msa, T, num_tokens])
    feats["msa"] = torch.zeros(B, n_msa, T, const.num_tokens, device=device)
    feats["msa"][:, :, :, 0] = 1.0
    feats["has_deletion"] = torch.zeros(B, n_msa, T, device=device)
    feats["deletion_value"] = torch.zeros(B, n_msa, T, device=device)
    feats["msa_paired"] = torch.zeros(B, n_msa, T, device=device)
    feats["msa_mask"] = torch.ones(B, n_msa, T, dtype=torch.bool, device=device)

    # Distogram targets
    feats["disto_center"] = torch.randn(B, T, 3, device=device)
    feats["token_disto_mask"] = torch.ones(B, T, dtype=torch.bool, device=device)

    return feats


def load_model_from_checkpoint(ckpt_path: str, device: str = "cpu") -> Boltz1:
    """Load a Boltz1 model from a Decaf checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hp = ckpt["hyper_parameters"]
    sd = ckpt["state_dict"]

    score_model_args = hp["score_model_args"]
    diffusion_process_args = {
        k: v for k, v in hp["diffusion_process_args"].items()
        if k in (
            "sigma_min", "sigma_max", "sigma_data", "rho",
            "P_mean", "P_std", "gamma_0", "gamma_min",
            "noise_scale", "step_scale", "coordinate_augmentation",
            "alignment_reverse_diff", "synchronize_sigmas",
            "use_inference_model_cache",
        )
    }
    decaf_args = {
        "sigma_min": hp["diffusion_process_args"]["sigma_min"],
        "sigma_max": hp["diffusion_process_args"]["sigma_max"],
        "sigma_data": hp["diffusion_process_args"]["sigma_data"],
        "rho": hp["diffusion_process_args"]["rho"],
        "coordinate_augmentation": hp["diffusion_process_args"]["coordinate_augmentation"],
        "target_type": hp.get("decaf_target_type", "teacher"),
        "diagonal_portion": hp["diffusion_process_args"].get("diagonal_portion", 0.5),
    }

    model = Boltz1(
        atom_s=hp["atom_s"],
        atom_z=hp["atom_z"],
        token_s=hp["token_s"],
        token_z=hp["token_z"],
        num_bins=hp["num_bins"],
        training_args=hp["training_args"],
        validation_args=hp["validation_args"],
        embedder_args=hp["embedder_args"],
        msa_args=hp["msa_args"],
        pairformer_args=hp["pairformer_args"],
        score_model_args=score_model_args,
        diffusion_process_args=diffusion_process_args,
        diffusion_loss_args=hp["diffusion_loss_args"],
        confidence_model_args=hp["confidence_model_args"],
        atom_feature_dim=hp["atom_feature_dim"],
        confidence_prediction=hp["confidence_prediction"],
        no_msa=hp["no_msa"],
        no_atom_encoder=hp["no_atom_encoder"],
        decaf_args=decaf_args,
    )

    result = model.load_state_dict(sd, strict=False)
    print(f"Missing keys: {len(result.missing_keys)}")
    print(f"Unexpected keys: {len(result.unexpected_keys)}")
    if result.missing_keys:
        print("  Missing:", result.missing_keys[:5])
    if result.unexpected_keys:
        print("  Unexpected:", result.unexpected_keys[:5])

    model.to(device)
    model.eval()
    return model


def main():
    import os
    ckpt_path = os.environ.get("DECAF_CKPT", "/tmp/decaf_ckpt.ckpt")
    device = os.environ.get("DECAF_DEVICE", "cpu")

    print("=" * 60)
    print("1. Loading model from checkpoint...")
    print("=" * 60)
    model = load_model_from_checkpoint(ckpt_path, device=device)
    print(f"   decaf_head present: {model.decaf_head is not None}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    print()
    print("=" * 60)
    print("2. Creating dummy features...")
    print("=" * 60)
    feats = make_dummy_feats(batch_size=1, n_tokens=32, n_atoms=32, device=device)
    print(f"   n_tokens={feats['token_pad_mask'].shape[1]}, n_atoms={feats['atom_pad_mask'].shape[1]}")

    print()
    print("=" * 60)
    print("3. Running Decaf inference (ODE, 5 steps)...")
    print("=" * 60)
    with torch.no_grad():
        out = model(
            feats=feats,
            recycling_steps=0,
            num_sampling_steps=5,
            diffusion_samples=1,
        )

    coords = out["sample_atom_coords"]
    print(f"   sample_atom_coords shape: {coords.shape}")
    print(f"   sample_atom_coords range: [{coords.min().item():.3f}, {coords.max().item():.3f}]")
    print(f"   sample_atom_coords mean:  {coords.mean().item():.3f}")
    print(f"   sample_atom_coords std:   {coords.std().item():.3f}")
    print(f"   NaN check: {torch.isnan(coords).any().item()}")
    print(f"   Inf check: {torch.isinf(coords).any().item()}")

    print()
    print("=" * 60)
    print("4. Running Decaf inference (SDE, 5 steps)...")
    print("=" * 60)
    from boltz.model.modules.decaf_sampler import DecafSampler, DecafSamplerConfig

    sde_config = DecafSamplerConfig(use_sde=True, random_augmentation=True)
    with torch.no_grad():
        out_sde = model.decaf_head.sample(
            atom_mask=feats["atom_pad_mask"],
            num_sampling_steps=5,
            multiplicity=1,
            sampler_config=sde_config,
            s_trunk=out["s"],
            z_trunk=out["z"],
            s_inputs=model.input_embedder(feats),
            feats=feats,
            relative_position_encoding=model.rel_pos(feats),
        )

    coords_sde = out_sde["sample_atom_coords"]
    print(f"   sample_atom_coords shape: {coords_sde.shape}")
    print(f"   sample_atom_coords range: [{coords_sde.min().item():.3f}, {coords_sde.max().item():.3f}]")
    print(f"   NaN check: {torch.isnan(coords_sde).any().item()}")

    print()
    print("=" * 60)
    print("5. Running Decaf inference (1-step)...")
    print("=" * 60)
    with torch.no_grad():
        out_1step = model(
            feats=feats,
            recycling_steps=0,
            num_sampling_steps=1,
            diffusion_samples=1,
        )

    coords_1 = out_1step["sample_atom_coords"]
    print(f"   sample_atom_coords shape: {coords_1.shape}")
    print(f"   NaN check: {torch.isnan(coords_1).any().item()}")

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
