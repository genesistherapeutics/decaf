# Decaf implementation based on "How to build a consistency model:
# Learning flow maps via self-distillation" by Boffi et al. (2025)
#
# Self-contained module — does not modify diffusion.py or AtomDiffusion.

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn import Module

import boltz.model.layers.initialize as init
from boltz.data import const
from boltz.model.loss.diffusion import (
    smooth_lddt_loss,
    weighted_rigid_align,
)
from boltz.model.modules.encoders import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    DualTimeConditioning,
    PairwiseConditioning,
)
from boltz.model.modules.transformers import DiffusionTransformer
from boltz.model.modules.utils import (
    LinearNoBias,
    center_random_augmentation,
    default,
    log,
)


# ---------------------------------------------------------------------------
# Score network with DualTimeConditioning (self-contained copy of DiffusionModule)
# ---------------------------------------------------------------------------


class DecafDiffusionModule(Module):
    """Score network for the Decaf head.

    Identical architecture to ``DiffusionModule`` except:
    - Uses ``DualTimeConditioning`` (two-time sinusoidal embeddings) instead
      of ``SingleConditioning`` (single-time Fourier features).
    - ``forward`` returns only ``{r_update, token_a}`` (no ``normed_fourier``).
    """

    def __init__(
        self,
        token_s: int,
        token_z: int,
        atom_s: int,
        atom_z: int,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        sigma_data: int = 16,
        dim_fourier: int = 256,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        token_transformer_depth: int = 24,
        token_transformer_heads: int = 8,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        atom_feature_dim: int = 128,
        conditioning_transition_layers: int = 2,
        activation_checkpointing: bool = False,
        offload_to_cpu: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.sigma_data = sigma_data

        self.single_conditioner = DualTimeConditioning(
            sigma_data=sigma_data,
            token_s=token_s,
            dim_fourier=dim_fourier,
            num_transitions=conditioning_transition_layers,
        )
        self.pairwise_conditioner = PairwiseConditioning(
            token_z=token_z,
            dim_token_rel_pos_feats=token_z,
            num_transitions=conditioning_transition_layers,
        )

        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            token_z=token_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_feature_dim=atom_feature_dim,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            structure_prediction=True,
            activation_checkpointing=activation_checkpointing,
        )

        self.s_to_a_linear = nn.Sequential(
            nn.LayerNorm(2 * token_s), LinearNoBias(2 * token_s, 2 * token_s)
        )
        init.final_init_(self.s_to_a_linear[1].weight)

        self.token_transformer = DiffusionTransformer(
            dim=2 * token_s,
            dim_single_cond=2 * token_s,
            dim_pairwise=token_z,
            depth=token_transformer_depth,
            heads=token_transformer_heads,
            activation_checkpointing=activation_checkpointing,
            offload_to_cpu=offload_to_cpu,
        )

        self.a_norm = nn.LayerNorm(2 * token_s)

        self.atom_attention_decoder = AtomAttentionDecoder(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            atom_decoder_depth=atom_decoder_depth,
            atom_decoder_heads=atom_decoder_heads,
            activation_checkpointing=activation_checkpointing,
        )

    def forward(
        self,
        s_inputs,
        s_trunk,
        z_trunk,
        r_noisy,
        times,
        relative_position_encoding,
        feats,
        multiplicity=1,
        model_cache=None,
    ):
        # DualTimeConditioning returns only s (no normed_fourier)
        s = self.single_conditioner(
            times=times,
            s_trunk=s_trunk.repeat_interleave(multiplicity, 0),
            s_inputs=s_inputs.repeat_interleave(multiplicity, 0),
        )

        if model_cache is None or len(model_cache) == 0:
            z = self.pairwise_conditioner(
                z_trunk=z_trunk, token_rel_pos_feats=relative_position_encoding
            )
        else:
            z = None

        a, q_skip, c_skip, p_skip, to_keys = self.atom_attention_encoder(
            feats=feats,
            s_trunk=s_trunk,
            z=z,
            r=r_noisy,
            multiplicity=multiplicity,
            model_cache=model_cache,
        )

        a = a + self.s_to_a_linear(s)

        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        a = self.token_transformer(
            a,
            mask=mask.float(),
            s=s,
            z=z,
            multiplicity=multiplicity,
            model_cache=model_cache,
        )
        a = self.a_norm(a)

        r_update = self.atom_attention_decoder(
            a=a,
            q=q_skip,
            c=c_skip,
            p=p_skip,
            feats=feats,
            multiplicity=multiplicity,
            to_keys=to_keys,
            model_cache=model_cache,
        )

        return {"r_update": r_update, "token_a": a.detach()}


# ---------------------------------------------------------------------------
# Loss configuration
# ---------------------------------------------------------------------------


@dataclass
class DecafLossConfig:
    """Configuration for Decaf loss weighting.

      - norm_power: exponent for the adaptive weight  w = 1 / (L + norm_const)^norm_power.
      - norm_const: additive constant for numerical stability in the weight.
    """

    norm_power: float = 0.5
    norm_const: float = 1e-6


# ---------------------------------------------------------------------------
# EDM helper functions (copied from AtomDiffusion to stay self-contained)
# ---------------------------------------------------------------------------


def _c_skip(sigma, sigma_data):
    return (sigma_data**2) / (sigma**2 + sigma_data**2)


def _c_out(sigma, sigma_data):
    return sigma * sigma_data / torch.sqrt(sigma_data**2 + sigma**2)


def _c_in(sigma, sigma_data):
    return 1 / torch.sqrt(sigma**2 + sigma_data**2)


def _c_noise(sigma, sigma_data):
    return log(sigma / sigma_data) * 0.25


def _t_to_sigma(t, sigma_max, sigma_min, rho, sigma_data):
    """Convert normalised time t in [0,1] to EDM sigma via Karras schedule."""
    inv_rho = 1.0 / rho
    sigmas = (
        sigma_max**inv_rho
        + t * (sigma_min**inv_rho - sigma_max**inv_rho)
    ) ** rho
    return sigmas * sigma_data


def _raw_teacher_forward(diffusion_module, noised_atom_coords, sigma, network_condition_kwargs):
    """Call a teacher AtomDiffusion's score_model with c_in scaling but
    *without* the c_skip/c_out EDM denoiser wrapping.  Returns the raw
    network ``r_update``."""
    padded_sigma = rearrange(sigma, "b -> b 1 1")
    sigma_data = diffusion_module.sigma_data
    c_in = _c_in(padded_sigma, sigma_data)

    net_out = diffusion_module.score_model(
        r_noisy=c_in * noised_atom_coords,
        times=_c_noise(sigma, sigma_data),
        **network_condition_kwargs,
    )
    return net_out["r_update"]


# ---------------------------------------------------------------------------
# AtomDecaf — self-contained Decaf head
# ---------------------------------------------------------------------------


class AtomDecaf(Module):
    """Flow Map head for learning flow maps between time points.

    Based on "How to build a consistency model: Learning flow maps via
    self-distillation" by Boffi et al. (2025).

    This is self-contained: it owns its own ``DecafDiffusionModule`` (score
    network with DualTimeConditioning) and does **not** inherit from or modify
    ``AtomDiffusion``.  During training the teacher EDM diffusion module
    (the regular ``AtomDiffusion`` / ``structure_module``) is passed as a
    parameter to ``forward()``.
    """

    def __init__(
        self,
        score_model_args: dict,
        num_sampling_steps: int = 5,
        sigma_min: float = 0.0004,
        sigma_max: float = 160.0,
        sigma_data: float = 16.0,
        rho: int = 7,
        coordinate_augmentation: bool = True,
        compile_score: bool = False,
        loss_config: DecafLossConfig | dict | None = None,
        diagonal_portion: float = 0.5,
        target_type: str = "teacher",
        **kwargs,
    ):
        super().__init__()

        self.score_model = DecafDiffusionModule(**score_model_args)
        if compile_score:
            self.score_model = torch.compile(
                self.score_model, dynamic=False, fullgraph=False
            )

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.inv_rho = 1.0 / rho
        self.num_sampling_steps = num_sampling_steps
        self.coordinate_augmentation = coordinate_augmentation

        self.diagonal_portion = diagonal_portion
        self.target_type = target_type

        if loss_config is None:
            self.loss_cfg = DecafLossConfig()
        elif isinstance(loss_config, dict):
            self.loss_cfg = DecafLossConfig(**loss_config)
        else:
            self.loss_cfg = loss_config

        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

    @property
    def device(self):
        return next(self.score_model.parameters()).device

    # -- interpolation helpers ------------------------------------------------

    def _teacher_sigma_from_flow_time(self, t, diffusion_module):
        """Convert flow time to EDM sigma using the teacher's Karras schedule."""
        return _t_to_sigma(
            1 - t,
            sigma_max=diffusion_module.sigma_max,
            sigma_min=diffusion_module.sigma_min,
            rho=diffusion_module.rho,
            sigma_data=diffusion_module.sigma_data,
        )

    def _velocity_from_x0(self, x0, z, sigma):
        padded_sigma = rearrange(sigma, "b -> b 1 1")
        return (z - x0) / padded_sigma

    def _x0_from_velocity(self, v, z, sigma):
        padded_sigma = rearrange(sigma, "b -> b 1 1")
        return z - padded_sigma * v

    # -- velocity prediction --------------------------------------------------

    def decaf_forward(
        self,
        noised_atom_coords_start,
        sigma_t,
        sigma_r,
        network_condition_kwargs: dict,
        training: bool = True,
        diffusion_module=None,
    ):
        """Forward pass — direct velocity prediction via DualTimeConditioning.

        Parameters
        ----------
        noised_atom_coords_start : Tensor  [B, N_atoms, 3]
            Noised atom coordinates z at source sigma.
        sigma_t : Tensor  [B]
            Source sigma (noisy, larger).  Must satisfy sigma_t >= sigma_r.
        sigma_r : Tensor  [B]
            Target sigma (cleaner, smaller).
        """
        assert torch.all(sigma_t >= sigma_r), (
            f"source sigma_t ({sigma_t}) must be >= target sigma_r ({sigma_r})"
        )

        multiplicity = network_condition_kwargs["multiplicity"]
        atom_mask = network_condition_kwargs["feats"]["atom_pad_mask"]
        noised_atom_coords_start = noised_atom_coords_start * atom_mask.repeat_interleave(
            multiplicity, 0
        ).unsqueeze(-1).float()

        c_in = 1 / torch.sqrt(sigma_t**2 + self.sigma_data**2)
        scaled_input = c_in[:, None, None] * noised_atom_coords_start

        concat_times = torch.cat([sigma_t, sigma_r], dim=0)

        net_out = self.score_model(
            r_noisy=scaled_input,
            times=concat_times,
            **network_condition_kwargs,
        )

        x0_pred = net_out["r_update"]
        return self._velocity_from_x0(x0_pred, noised_atom_coords_start, sigma_t)

    # -- schedule -------------------------------------------------------------

    def sample_schedule(
        self,
        num_sampling_steps=None,
        steps=None,
        sigma_max=None,
        sigma_data=None,
        dtype=torch.float32,
    ):
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        sigma_max = default(sigma_max, self.sigma_max)
        sigma_data = default(sigma_data, self.sigma_data)

        if num_sampling_steps == 1 and steps is None:
            steps = torch.tensor([0.0], device=self.device, dtype=dtype)

        if steps is None:
            steps = torch.arange(num_sampling_steps, device=self.device, dtype=dtype) / (
                num_sampling_steps - 1
            )
        elif isinstance(steps, list):
            steps = torch.tensor(steps, device=self.device, dtype=dtype)

        sigmas = (
            sigma_max**self.inv_rho
            + steps * (self.sigma_min**self.inv_rho - sigma_max**self.inv_rho)
        ) ** self.rho

        sigmas = sigmas * sigma_data
        sigmas = F.pad(sigmas, (0, 1), value=0.0)
        return sigmas

    # -- training forward (JVP) -----------------------------------------------

    def forward(
        self,
        s_inputs,
        s_trunk,
        z_trunk,
        relative_position_encoding,
        feats,
        sigmas=None,
        multiplicity=1,
        training=True,
        diffusion_module=None,
    ):
        """Decaf training forward — JVP-based self-distillation.

        ``diffusion_module`` is the teacher (the regular ``AtomDiffusion`` /
        ``structure_module``).  It is called under ``torch.no_grad()`` to
        produce the teacher x0 prediction that provides the velocity target.
        """
        batch_size = feats["coords"].shape[0]
        device = feats["coords"].device

        # Decaf convention: t = source time (noisy), r = target time (clean).
        t1 = torch.rand(batch_size * multiplicity, device=device)
        t2 = torch.rand(batch_size * multiplicity, device=device)
        t = torch.maximum(t1, t2)
        r = torch.minimum(t1, t2)

        diagonal_mask = torch.rand(batch_size * multiplicity, device=device) < self.diagonal_portion
        r = torch.where(diagonal_mask, t, r)

        atom_coords = feats["coords"]
        B, N, L = atom_coords.shape[0:3]
        atom_coords = atom_coords.reshape(B * N, L, 3)

        assert N == 1, f"Unexpectedly have {N=}, which is incompatible with the next line"
        atom_coords = atom_coords.repeat_interleave(multiplicity // N, 0)

        atom_mask = feats["atom_pad_mask"]
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        atom_coords = center_random_augmentation(
            atom_coords,
            atom_mask,
            augmentation=self.coordinate_augmentation and training,
        )

        noise = torch.randn_like(atom_coords)

        sigma_t = self._teacher_sigma_from_flow_time(t, diffusion_module)
        sigma_r = self._teacher_sigma_from_flow_time(r, diffusion_module)
        padded_sigma_t = rearrange(sigma_t, "b -> b 1 1")

        z = atom_coords + padded_sigma_t * noise

        network_condition_kwargs = dict(
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            relative_position_encoding=relative_position_encoding,
            feats=feats,
            multiplicity=multiplicity,
        )

        # Teacher prediction (no grad)
        z_ve = z

        with torch.no_grad():
            raw_output = _raw_teacher_forward(
                diffusion_module, z_ve, sigma_t, network_condition_kwargs
            )
            x_0_teacher = (
                _c_skip(padded_sigma_t, diffusion_module.sigma_data) * z_ve
                + _c_out(padded_sigma_t, diffusion_module.sigma_data) * raw_output
            )
            v = self._velocity_from_x0(x_0_teacher.detach(), z, sigma_t)

        # JVP through the student flow map
        def decaf_func_sigma(x, sig_t, sig_r):
            return self.decaf_forward(
                x, sig_t, sig_r,
                training=True,
                diffusion_module=diffusion_module,
                network_condition_kwargs=network_condition_kwargs,
            )

        s_1 = torch.ones_like(sigma_t)
        s_0 = torch.zeros_like(sigma_r)
        outputs, jvp_out = torch.autograd.functional.jvp(
            decaf_func_sigma,
            (z, sigma_t, sigma_r),
            (v, s_1, s_0),
            create_graph=True,
        )
        u = outputs
        du_ds = jvp_out

        sigma_diff = (sigma_t - sigma_r)[:, None, None]
        V = u + sigma_diff * du_ds.detach()
        x_0_student = self._x0_from_velocity(V, z, sigma_t)

        mse_target = x_0_teacher

        return dict(
            denoised_atom_coords=x_0_student,
            model_objective_pred=x_0_student,
            ts_end=r,
            ts_flow_matching=t,
            ts=t,
            sigma_t=sigma_t,
            aligned_true_atom_coords=atom_coords,
            batch_size=batch_size,
            network_condition_kwargs=network_condition_kwargs,
            mse_target=mse_target,
        )

    # -- loss ------------------------------------------------------------------

    @torch.autocast("cuda", enabled=False)
    def compute_loss(
        self,
        feats,
        out_dict,
        add_smooth_lddt_loss=True,
        nucleotide_loss_weight=5.0,
        ligand_loss_weight=10.0,
        multiplicity=1,
        **kwargs,
    ):
        V = out_dict["model_objective_pred"].to(torch.float32)
        x_0_student = V

        resolved_atom_mask = feats["atom_resolved_mask"].repeat_interleave(multiplicity, 0)
        mse_target = out_dict["mse_target"]
        sigma_t = out_dict["sigma_t"]

        atom_type = (
            torch.bmm(
                feats["atom_to_token"].float(),
                feats["mol_type"].unsqueeze(-1).float(),
            )
            .squeeze(-1)
            .long()
        )
        atom_type_mult = atom_type.repeat_interleave(multiplicity, 0)

        # Boltz-style per-type alignment weights
        align_weights = x_0_student.new_ones(x_0_student.shape[:2])
        align_weights = align_weights * (
            1
            + nucleotide_loss_weight
            * (
                torch.eq(atom_type_mult, const.chain_type_ids["DNA"]).float()
                + torch.eq(atom_type_mult, const.chain_type_ids["RNA"]).float()
            )
            + ligand_loss_weight
            * torch.eq(atom_type_mult, const.chain_type_ids["NONPOLYMER"]).float()
        )

        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            mse_target_aligned = weighted_rigid_align(
                mse_target.detach().float(),
                x_0_student.detach().float(),
                align_weights.detach().float(),
                mask=resolved_atom_mask.detach().float(),
            ).to(torch.float32)

        cfg = self.loss_cfg

        delta = x_0_student - mse_target_aligned
        delta_sq = (delta**2).sum(dim=-1)

        w_sigma = 1.0 / sigma_t**2

        L = (
            w_sigma
            * torch.sum(delta_sq * resolved_atom_mask * align_weights, dim=-1)
            / torch.sum(3 * align_weights * resolved_atom_mask, dim=-1)
        )

        w_delta = 1.0 / ((L.detach() + cfg.norm_const) ** cfg.norm_power)
        mse_loss = w_delta * L

        decaf_loss = mse_loss
        total_loss = decaf_loss

        lddt_loss = self.zero
        if add_smooth_lddt_loss:
            lddt_loss = smooth_lddt_loss(
                x_0_student,
                mse_target_aligned,
                torch.eq(atom_type, const.chain_type_ids["DNA"]).float()
                + torch.eq(atom_type, const.chain_type_ids["RNA"]).float(),
                coords_mask=feats["atom_resolved_mask"],
                multiplicity=multiplicity,
            )
            total_loss = total_loss + lddt_loss

        loss_breakdown = dict(
            decaf_loss=decaf_loss.mean(),
            mse_loss=decaf_loss.mean(),
            smooth_lddt_loss=lddt_loss if torch.is_tensor(lddt_loss) else self.zero,
            w_delta=w_delta.mean(),
        )
        loss_per_sample_breakdown = dict(
            diffusion_loss=total_loss.detach(),
            decaf_loss=decaf_loss.detach(),
            smooth_lddt_loss=lddt_loss.detach() if torch.is_tensor(lddt_loss) else self.zero,
        )

        return dict(
            loss=total_loss.mean(),
            loss_breakdown=loss_breakdown,
            loss_per_sample_breakdown=loss_per_sample_breakdown,
        )

    # -- sampling (delegates to DecafSampler) --------------------------------

    def sample(
        self,
        atom_mask,
        num_sampling_steps=None,
        multiplicity=1,
        max_parallel_samples=None,
        train_accumulate_token_repr=False,
        steering_args=None,
        sampler_config=None,
        **network_condition_kwargs,
    ):
        """Sample structures using Euler integration (ODE or SDE).

        Parameters
        ----------
        steering_args : dict, optional
            Steering configuration dict (fk_steering, physical_guidance_update, etc.).
        sampler_config : DecafSamplerConfig, optional
            Full sampler configuration.  If not provided, a default is created.
        """
        from boltz.model.modules.decaf_sampler import (
            DecafSampler, DecafSamplerConfig,
            MCGradSampler, MCGradSamplerConfig,
            MCTSFullRolloutDecafSampler, MCTSFullRolloutDecafSamplerConfig,
        )

        if sampler_config is not None:
            if isinstance(sampler_config, MCTSFullRolloutDecafSamplerConfig):
                sampler = MCTSFullRolloutDecafSampler(cfg=sampler_config)
            elif isinstance(sampler_config, MCGradSamplerConfig):
                sampler = MCGradSampler(cfg=sampler_config)
            else:
                sampler = DecafSampler(cfg=sampler_config)
        else:
            sampler = DecafSampler()

        return sampler.sample(
            decaf_module=self,
            atom_mask=atom_mask,
            num_sampling_steps=num_sampling_steps,
            multiplicity=multiplicity,
            steering_args=steering_args,
            **network_condition_kwargs,
        )
