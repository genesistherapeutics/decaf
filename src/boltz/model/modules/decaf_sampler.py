# Decaf sampler with optional FK steering and guidance.
#
# Self-contained: depends only on the Boltz potentials system
# (boltz.model.potentials) and standard Decaf methods
# (decaf_forward, sample_schedule).

from __future__ import annotations

import logging
import time
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from math import sqrt
from typing import Any

import torch
import torch.nn.functional as F

from boltz.model.modules.utils import (
    compute_random_augmentation,
    default,
)
from boltz.model.potentials.potentials import get_potentials

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DecafSamplerConfig:
    """Configuration for the DecafSampler."""

    use_sde: bool = True
    sde_gamma: float = 1.0  # γ-sampling: 1.0=full SDE, 0.0=ODE, 0<γ<1=partial denoise+renoise
    random_augmentation: bool = True

    # FK steering
    fk_steering: bool = False
    num_particles: int = 1
    fk_resampling_interval: int = 3
    fk_lambda: float = 1.0

    # Guidance (gradient-based)
    physical_guidance_update: bool = False
    num_gd_steps: int = 1
    guidance_step_scale: float = 1.5


@dataclass
class MCGradSamplerConfig(DecafSamplerConfig):
    """Configuration for MCGradSampler (Weighted MC-GRAD)."""

    num_mc_grad_particles: int = 4
    snr_ratio: float = 10.0
    use_score_correction: bool = True
    mc_grad_sigma_min: float = 0.0
    mc_grad_sigma_max: float = 1e30
    invert_reward: bool = False
    mc_grad_use_gd_guidance: bool = True  # Apply iterative GD steps on x0 after mc_grad correction


@dataclass
class MCTSFullRolloutDecafSamplerConfig(DecafSamplerConfig):
    """Configuration for MCTSFullRolloutDecafSampler."""

    # MCTS parameters
    num_simulations: int = 50
    simulation_batch_size: int = 4
    expansion_children: int = 4
    c_uct: float = 1.0
    inv_temp: float = 1.0
    output_selection_method: str = "root"  # "root" or "leaves"
    num_roots: int = 1

    # Progressive widening: max_children = pw_k * visits^pw_alpha
    pw_k: float = 2.0
    pw_alpha: float = 0.5

    # If non-empty, only branch at these step indices; else branch at every step
    branching_timesteps: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# DecafSampler
# ---------------------------------------------------------------------------


class DecafSampler:
    """Sampler for Decaf models.

    Supports two modes:
    - **ODE** (default): deterministic Euler integration
        x_next = x - (sigma_t - sigma_r) * flow(x, sigma_t, sigma_r)
    - **SDE**: velocity step followed by noise re-injection (FastGen-style)
        x_drift = x - sigma_t * flow(x, sigma_t, 0)
        x_next  = x_drift + sigma_r * eps          (if sigma_r > 0)

    When ``use_sde=True``, the model is always queried with ``sigma_r=0``
    (instantaneous velocity) and noise is re-injected at the target noise
    level ``sigma_r``.

    Additionally supports:
    - **FK steering**: Feynman-Kac resampling of particles using potentials
    - **Guidance**: iterative gradient-based guidance on x0 predictions
    """

    def __init__(
        self,
        cfg: DecafSamplerConfig | None = None,
    ):
        self.cfg = cfg if cfg is not None else DecafSamplerConfig()

    @property
    def _has_steering_or_guidance(self) -> bool:
        return self.cfg.fk_steering or self.cfg.physical_guidance_update

    # ------------------------------------------------------------------
    # x0 recovery from Decaf velocity
    # ------------------------------------------------------------------

    @staticmethod
    def _recover_x0(atom_coords, phi, sigma_t):
        """x0 = z - sigma_t * phi  (instantaneous velocity prediction to sigma=0)."""
        return atom_coords - sigma_t[:, None, None] * phi

    # ------------------------------------------------------------------
    # Overridable step methods
    # ------------------------------------------------------------------

    def _step_fast(self, atom_coords, sigma_t_vec, sigma_r_vec, decaf_module, nck):
        """Unguided step. Override in subclasses for different transition kernels.

        Implements γ-sampling (Algorithm 2) when use_sde=True:
          ˜σ_r = sqrt(1 - γ²) · σ_r       (partial denoise target)
          x_˜σ = Decaf(x, σ_t, ˜σ_r)    (denoise)
          x_σr = x_˜σ + γ · σ_r · ε       (diffuse)

        γ=1 recovers full SDE (denoise to x0, renoise to σ_r).
        γ=0 recovers ODE (direct flow map step σ_t → σ_r).
        """
        if self.cfg.use_sde:
            gamma = self.cfg.sde_gamma
            sigma_r_val = sigma_r_vec.item()

            if sigma_r_val <= 0 or gamma >= 1.0:
                # γ=1 or last step: full denoise to x0, then renoise
                sigma_zero = torch.zeros_like(sigma_t_vec)
                phi = decaf_module.decaf_forward(
                    atom_coords, sigma_t_vec, sigma_zero,
                    training=False,
                    network_condition_kwargs=nck,
                )
                x0 = atom_coords - sigma_t_vec[:, None, None] * phi
                if sigma_r_val > 0:
                    return x0 + sigma_r_vec[:, None, None] * torch.randn_like(atom_coords)
                return x0

            # γ-sampling: partial denoise then diffuse
            sigma_tilde = sqrt(1.0 - gamma**2) * sigma_r_val
            sigma_tilde_vec = torch.tensor([sigma_tilde], device=sigma_t_vec.device)
            phi = decaf_module.decaf_forward(
                atom_coords, sigma_t_vec, sigma_tilde_vec,
                training=False,
                network_condition_kwargs=nck,
            )
            x_tilde = atom_coords - (sigma_t_vec - sigma_tilde_vec)[:, None, None] * phi
            return x_tilde + gamma * sigma_r_vec[:, None, None] * torch.randn_like(atom_coords)
        else:
            phi = decaf_module.decaf_forward(
                atom_coords, sigma_t_vec, sigma_r_vec,
                training=False,
                network_condition_kwargs=nck,
            )
            return atom_coords - (sigma_t_vec - sigma_r_vec)[:, None, None] * phi

    def _recover_x0_for_guidance(self, atom_coords, sigma_t_vec, sigma_r_vec,
                                  decaf_module, nck):
        """Recover x0 for use in FK/guidance."""
        sigma_zero = torch.zeros_like(sigma_t_vec)
        phi = decaf_module.decaf_forward(
            atom_coords, sigma_t_vec, sigma_zero,
            training=False,
            network_condition_kwargs=nck,
        )
        return self._recover_x0(atom_coords, phi, sigma_t_vec)

    # ------------------------------------------------------------------
    # Main sampling entry point
    # ------------------------------------------------------------------

    def sample(
        self,
        decaf_module,
        atom_mask,
        num_sampling_steps=None,
        multiplicity=1,
        steering_args=None,
        **network_condition_kwargs,
    ):
        """Full sampling loop: initialize noise, iterate Euler steps.

        Parameters
        ----------
        decaf_module : AtomDecaf
            The flow map module providing ``decaf_forward`` and ``sample_schedule``.
        atom_mask : Tensor [B, N_atoms]
            Atom padding mask.
        num_sampling_steps : int, optional
            Number of sampling steps.  Uses module default if None.
        multiplicity : int
            Number of samples per input.
        steering_args : dict, optional
            Steering configuration (potentials, FK, guidance).  If provided,
            overrides the sampler config for FK/guidance settings.
        **network_condition_kwargs
            Passed through to ``decaf_forward`` (s_trunk, z_trunk, etc.).

        Returns
        -------
        dict with ``sample_atom_coords`` and ``diff_token_repr``.
        """
        num_sampling_steps = default(num_sampling_steps, decaf_module.num_sampling_steps)
        device = decaf_module.device

        logger.info(
            "%s.sample: steps=%d, multiplicity=%d, sde=%s, gamma=%.2f, fk=%s, guidance=%s",
            type(self).__name__, num_sampling_steps, multiplicity, self.cfg.use_sde,
            self.cfg.sde_gamma, self.cfg.fk_steering, self.cfg.physical_guidance_update,
        )

        # Resolve steering
        use_fk = self.cfg.fk_steering
        use_guidance = self.cfg.physical_guidance_update
        potentials = []
        num_particles = self.cfg.num_particles if use_fk else 1
        fk_lambda = self.cfg.fk_lambda
        fk_interval = self.cfg.fk_resampling_interval
        num_gd_steps = self.cfg.num_gd_steps
        step_scale = self.cfg.guidance_step_scale

        if steering_args is not None:
            use_fk = steering_args.get("fk_steering", use_fk)
            use_guidance = steering_args.get("physical_guidance_update", use_guidance)
            num_particles = steering_args.get("num_particles", num_particles) if use_fk else 1
            fk_lambda = steering_args.get("fk_lambda", fk_lambda)
            fk_interval = steering_args.get("fk_resampling_interval", fk_interval)
            num_gd_steps = steering_args.get("num_gd_steps", num_gd_steps)
            step_scale = steering_args.get("guidance_step_scale", step_scale)
            if use_fk or use_guidance:
                potentials = get_potentials(steering_args, boltz2=False)

        effective_mult = multiplicity * num_particles
        atom_mask = atom_mask.repeat_interleave(effective_mult, 0)
        shape = (*atom_mask.shape, 3)

        sigmas = decaf_module.sample_schedule(num_sampling_steps)
        sigma_pairs = list(zip(sigmas[:-1], sigmas[1:]))
        logger.info("%s schedule: %d steps, sigma_max=%.4f -> sigma_min=%.4f",
                     type(self).__name__, len(sigma_pairs), sigmas[0].item(), sigmas[-1].item())

        # Initialise with noise scaled by the first sigma
        init_sigma = sigmas[0]
        atom_coords = init_sigma * torch.randn(shape, device=device)

        nck = dict(network_condition_kwargs, multiplicity=effective_mult)

        # FK steering state
        energy_traj = torch.empty((effective_mult, 0), device=device) if use_fk else None
        scaled_guidance_update = (
            torch.zeros(shape, device=device, dtype=torch.float32) if use_guidance else None
        )

        for step_idx, (sigma_tm, sigma_t) in enumerate(sigma_pairs):
            last_step = step_idx == len(sigma_pairs) - 1
            steering_t = 1.0 - (step_idx / num_sampling_steps)

            # (0) Random augmentation
            if self.cfg.random_augmentation:
                random_R, random_tr = compute_random_augmentation(
                    effective_mult, device=atom_coords.device, dtype=atom_coords.dtype
                )
                atom_coords = atom_coords - atom_coords.mean(dim=-2, keepdims=True)
                atom_coords = torch.einsum("bmd,bds->bms", atom_coords, random_R) + random_tr
                if scaled_guidance_update is not None:
                    scaled_guidance_update = torch.einsum(
                        "bmd,bds->bms", scaled_guidance_update, random_R
                    )

            sigma_tm_val = sigma_tm.item()
            sigma_t_val = sigma_t.item()
            sigma_t_vec = torch.tensor([sigma_tm_val], device=device)
            sigma_r_vec = torch.tensor([sigma_t_val], device=device)

            with torch.no_grad():
                # -------------------------------------------------------
                # Fast path: no steering / guidance
                # -------------------------------------------------------
                if not (use_fk or use_guidance):
                    atom_coords = self._step_fast(
                        atom_coords, sigma_t_vec, sigma_r_vec, decaf_module, nck,
                    )
                    logger.info(
                        f"step {step_idx}: sigma_t={sigma_tm_val:.4f} -> sigma_r={sigma_t_val:.4f}, "
                        f"sde={self.cfg.use_sde}"
                    )
                    continue

                # -------------------------------------------------------
                # Guided path: x0 → FK weights → guidance → step
                # -------------------------------------------------------

                # (1) Get x0 prediction
                x0 = self._recover_x0_for_guidance(
                    atom_coords, sigma_t_vec, sigma_r_vec, decaf_module, nck,
                )

                # (2) FK steering: compute energy and resampling weights
                if use_fk and (
                    (step_idx % fk_interval == 0) or last_step
                ):
                    energy = torch.zeros(effective_mult, device=device)
                    for potential in potentials:
                        parameters = potential.compute_parameters(steering_t)
                        if parameters["resampling_weight"] > 0:
                            component_energy = potential.compute(
                                x0,
                                network_condition_kwargs["feats"],
                                parameters,
                            )
                            energy += parameters["resampling_weight"] * component_energy
                    energy_traj = torch.cat((energy_traj, energy.unsqueeze(1)), dim=1)

                    if step_idx == 0:
                        log_G = -1 * energy
                    else:
                        log_G = energy_traj[:, -2] - energy_traj[:, -1]

                    ll_difference = torch.zeros_like(energy)

                    resample_weights = F.softmax(
                        (ll_difference + fk_lambda * log_G).reshape(
                            -1, num_particles
                        ),
                        dim=1,
                    )

                # (3) Guidance: gradient-based updates on x0
                if use_guidance and not last_step:
                    guidance_update = torch.zeros_like(x0)
                    for _ in range(num_gd_steps):
                        energy_gradient = torch.zeros_like(x0)
                        for potential in potentials:
                            parameters = potential.compute_parameters(steering_t)
                            if parameters.get("guidance_weight", 0) > 0:
                                energy_gradient += parameters[
                                    "guidance_weight"
                                ] * potential.compute_gradient(
                                    x0 + guidance_update,
                                    network_condition_kwargs["feats"],
                                    parameters,
                                ) 
                        guidance_update -= energy_gradient
                    x0 = x0 + guidance_update
                    scaled_guidance_update = (
                        guidance_update * -1 * step_scale
                        * (sigma_t_val - sigma_tm_val) / max(sigma_tm_val, 1e-8)
                    )

                # (4) FK resampling
                if use_fk and (
                    (step_idx % fk_interval == 0) or last_step 
                ):
                    n_resample = 1 if last_step else num_particles
                    resample_indices = torch.multinomial(
                        resample_weights,
                        num_samples=n_resample,
                        replacement=True,
                    )
                    batch_offsets = (
                        torch.arange(
                            resample_weights.shape[0], device=device
                        )[:, None]
                        * num_particles
                    )
                    flat_indices = (resample_indices + batch_offsets).reshape(-1)
                    atom_coords = atom_coords[flat_indices]
                    x0 = x0[flat_indices]
                    energy_traj = energy_traj[flat_indices]
                    if scaled_guidance_update is not None:
                        scaled_guidance_update = scaled_guidance_update[flat_indices]

                # (5) Step from guided x0 using γ-sampling
                if self.cfg.use_sde:
                    gamma = self.cfg.sde_gamma
                    sigma_r_val = sigma_r_vec.item()
                    if sigma_r_val <= 0:
                        atom_coords = x0
                    elif gamma >= 1.0:
                        # Full SDE: x0 + σ_r · ε
                        atom_coords = x0 + sigma_r_vec[:, None, None] * torch.randn_like(x0)
                    else:
                        # γ-sampling: reconstruct x at ˜σ from guided x0, then diffuse
                        sigma_tilde = sqrt(1.0 - gamma**2) * sigma_r_val
                        atom_coords = (
                            x0
                            + sigma_tilde * (atom_coords - x0) / sigma_t_vec[:, None, None]
                            + gamma * sigma_r_vec[:, None, None] * torch.randn_like(x0)
                        )
                else:
                    guided_phi = (atom_coords - x0) / sigma_t_vec[:, None, None]
                    atom_coords = (
                        atom_coords
                        - (sigma_t_vec - sigma_r_vec)[:, None, None] * guided_phi
                    )

            logger.info(
                f"step {step_idx}: sigma_t={sigma_tm_val:.4f} -> sigma_r={sigma_t_val:.4f}, "
                f"sde={self.cfg.use_sde}, fk={use_fk}, guidance={use_guidance}"
            )

        return dict(sample_atom_coords=atom_coords, diff_token_repr=None)


# ---------------------------------------------------------------------------
# MCGradSampler
# ---------------------------------------------------------------------------


class MCGradSampler(DecafSampler):
    """MC-GRAD sampler (Algorithm 2) for Decaf models.

    Implements Algorithm 3 from
    "MC-GRAD: Efficient Reward Alignment via Stochastic Flow Maps"
    (Holderrieth et al., 2025), adapted for VE interpolation in sigma space.

    At each guided step, computes an importance-weighted gradient of the value
    function ∇V_t^r(x_t) and applies it as a correction to the base x0:

        x0_corrected = x0_base + sigma_t^2 * ∇V

    where ∇V = Σ_k softmax(v)[k] * (∇_k + δ_k) with:
        ∇_k  = ∇_{x_t} r(z_k)                        [reward gradient via VJP]
        δ_k  = s(x'_k, sigma') - s(x_t, sigma_t)     [score correction]
        v_k  = r(z_k) - ||x_t - z_k||^2/(2*sigma_t^2) + gamma_k + 0.5*||eps_k||^2

    When no FK/guidance is configured, falls back to plain DecafSampler.
    Outside the mc_grad sigma window, also falls back to the parent step.
    """

    def __init__(self, cfg: MCGradSamplerConfig | None = None):
        super().__init__(cfg if cfg is not None else MCGradSamplerConfig())
        # Per-step context set during the sample loop so that
        # _recover_x0_for_guidance can evaluate the reward.
        self._mc_grad_potentials: list[Any] = []
        self._mc_grad_feats: Any = None
        self._mc_grad_steering_t: float = 1.0
        self._in_guided_mode: bool = False

        # MC-GRAD calls the model K times per step; raise dynamo cache limit.
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = max(
            torch._dynamo.config.cache_size_limit, 64
        )

    @property
    def _mc_grad_cfg(self) -> MCGradSamplerConfig:
        return self.cfg  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Reward evaluation (FK potentials negated → higher = better)
    # ------------------------------------------------------------------

    def _evaluate_mc_grad_reward(self, x0: torch.Tensor) -> torch.Tensor:
        """Evaluate FK potentials on x0 as MC-GRAD reward.

        Potentials are cost functions (lower = better), so reward = -energy.
        """
        total = torch.zeros(x0.shape[0], device=x0.device)
        for potential in self._mc_grad_potentials:
            parameters = potential.compute_parameters(self._mc_grad_steering_t)
            if parameters.get("resampling_weight", 0) > 0:
                energy = potential.compute(x0, self._mc_grad_feats, parameters)
                total = total - parameters["resampling_weight"] * energy
        if self._mc_grad_cfg.invert_reward:
            total = -total
        return total

    # ------------------------------------------------------------------
    # Flow map helper
    # ------------------------------------------------------------------

    def _decaf_x0(
        self,
        z: torch.Tensor,
        sigma: float,
        decaf_module,
        nck: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict x0 via flow map at given sigma. Returns (x0, phi)."""
        sigma_t = torch.tensor([sigma], device=z.device, dtype=z.dtype)
        sigma_zero = torch.zeros_like(sigma_t)
        phi = decaf_module.decaf_forward(
            z, sigma_t, sigma_zero,
            training=False,
            network_condition_kwargs=nck,
        )
        return z - sigma * phi, phi

    # ------------------------------------------------------------------
    # Algorithm 2: importance-weighted value gradient
    # ------------------------------------------------------------------

    def _compute_reward_grad_vjp(
        self,
        x_prime_k: torch.Tensor,
        z_k_detached: torch.Tensor,
        z_k_with_grad: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ∇_{x'_k} r(z_k) via backward-mode VJP through the flow map.

        Uses potential.compute_gradient(z_k) as the vector in the VJP:
          dr/dx'_k = Jac(z_k / x'_k)^T * (dr/dz_k)
        where dr/dz_k = -Σ_pot w_pot * dE/dz_k  (reward = -energy).

        Uses .backward() (not torch.autograd.grad) because the flow map may use
        activation checkpointing, which is incompatible with .grad().

        Must be called inside a torch.inference_mode(False) + torch.enable_grad()
        context, with z_k_with_grad having been computed in the same context.
        """
        dr_dz_k = torch.zeros_like(z_k_detached)
        for potential in self._mc_grad_potentials:
            params = potential.compute_parameters(self._mc_grad_steering_t)
            if params.get("resampling_weight", 0) > 0:
                # compute_gradient returns dE/dz; reward = -energy → dr/dz = -dE/dz
                dr_dz_k = dr_dz_k - params["resampling_weight"] * potential.compute_gradient(
                    z_k_detached, self._mc_grad_feats, params
                )

        if dr_dz_k.abs().max() < 1e-12:
            return torch.zeros_like(z_k_detached)

        # VJP via .backward() — compatible with activation checkpointing
        x_prime_k.grad = None
        z_k_with_grad.backward(dr_dz_k)
        grad = x_prime_k.grad
        x_prime_k.grad = None  # free memory
        return grad if grad is not None else torch.zeros_like(z_k_detached)

    def _alg3_value_gradient(
        self,
        z: torch.Tensor,
        sigma_t_val: float,
        decaf_module,
        nck: dict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Algorithm 2 (MC-GRAD): importance-weighted ∇V_t^r(x_t).

        For VE schedule (α=1):
          sigma' = sqrt(lambda) * sigma_t
          for k in 1..K:
            eps_k ~ N(0, I)
            x'_k  = x_t + sqrt(sigma'^2 - sigma_t^2) * eps_k    [renoise]
            z_k   = X_{sigma',0}(x'_k)                           [flow map → x0]
            r_k   = r(z_k) - ||x_t - z_k||^2 / (2*sigma_t^2)    [local reward]

            grad_k    = ∇_{x_t} r(z_k)                          [reward grad via VJP]

            s_k   = -phi(x'_k, sigma') / sigma'                  [score at particle]
            delta_k = s_k - s_t                                   [score correction]
            gamma_k = 0.5 * (s_k + s_{sigma'->x_t})^T (x'_k - x_t)
            v_k   = r_k + gamma_k + 0.5 * ||eps_k||^2            [logit]
          ∇V = sum_k softmax(v)[k] * (grad_k + delta_k)

        Note: grad_r_k is computed via VJP through the flow map using
        potential.compute_gradient as the grad-output vector. This is the
        key term missing in the naive implementation and is required for
        correct value function gradient estimation per Proposition 5.1.
        """
        cfg = self._mc_grad_cfg
        K = cfg.num_mc_grad_particles
        sigma_prime = sqrt(cfg.snr_ratio) * sigma_t_val
        noise_std = sqrt(sigma_prime**2 - sigma_t_val**2)
        device = z.device
        z_det = z.detach()

        # Escape inference mode once: clone z and nck tensors so that downstream
        # autograd operations can save them as intermediates.  Lightning's predict
        # loop runs under torch.inference_mode(); torch.enable_grad() alone does
        # NOT override inference mode, so we must explicitly exit it.
        with torch.inference_mode(mode=False):
            z_ng = z_det.clone()  # non-inference copy of x_t
            nck_ng = {
                k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
                for k, v in nck.items()
            }

        # Scores at x_t (no grad needed)
        with torch.no_grad():
            _, phi_at_sigma_t = self._decaf_x0(z_ng, sigma_t_val, decaf_module, nck_ng)
            s_t = -phi_at_sigma_t / sigma_t_val

            s_tp_at_xt: torch.Tensor | None = None
            if cfg.use_score_correction:
                _, phi_at_sp = self._decaf_x0(z_ng, sigma_prime, decaf_module, nck_ng)
                s_tp_at_xt = -phi_at_sp / sigma_prime

        grad_r_locals: list[torch.Tensor] = []
        delta_scores: list[torch.Tensor] = []
        logits: list[torch.Tensor] = []

        for _ in range(K):
            eps_k = torch.randn_like(z_ng)

            # Run flow map with grad tracking and compute reward VJP inside a single
            # inference_mode(False)+enable_grad() context.  We must call .backward()
            # here (not torch.autograd.grad) because the flow map may use activation
            # checkpointing, which is incompatible with the functional .grad() API.
            with torch.inference_mode(mode=False):
                with torch.enable_grad():
                    x_prime_k = (z_ng + noise_std * eps_k).requires_grad_(True)
                    z_k_grad, phi_k_g = self._decaf_x0(
                        x_prime_k, sigma_prime, decaf_module, nck_ng
                    )
                    z_k = z_k_grad.detach()
                    # Reward gradient through flow map via .backward()
                    grad_reward_k = self._compute_reward_grad_vjp(x_prime_k, z_k, z_k_grad)

            phi_k = phi_k_g.detach()

            # Local reward (for logit; no grad needed)
            with torch.no_grad():
                reward_k = self._evaluate_mc_grad_reward(z_k)
            recovery_k = -(z_ng - z_k).pow(2).flatten(1).sum(-1) / (2 * sigma_t_val**2)
            r_local_k = reward_k + recovery_k

            # Gradient of r_local w.r.t. x_t (reward only):
            #   grad_k = ∇_{x_t} r(z_k)
            grad_k = grad_reward_k

            # Score correction and gamma
            s_k = -phi_k / sigma_prime
            if cfg.use_score_correction and s_tp_at_xt is not None:
                delta_scores.append(s_k - s_t)
                diff_k = noise_std * eps_k
                gamma_k = 0.5 * ((s_k + s_tp_at_xt) * diff_k).flatten(1).sum(-1)
            else:
                delta_scores.append(torch.zeros_like(s_t))
                gamma_k = torch.zeros(z_ng.shape[0], device=device)

            grad_r_locals.append(grad_k)
            noise_logit_k = 0.5 * eps_k.pow(2).flatten(1).sum(-1)
            logits.append(r_local_k.detach() + gamma_k.detach() + noise_logit_k)

        weights = torch.softmax(torch.stack(logits, dim=0), dim=0)  # [K, B]

        grad_V = torch.zeros_like(z)
        for k in range(K):
            w_k = weights[k].unsqueeze(-1).unsqueeze(-1)
            grad_V = grad_V + w_k * (grad_r_locals[k] + delta_scores[k])

        grad_norm = grad_V.norm(p=2, dim=(-1, -2), keepdim=True)

        ess = (1.0 / (weights.pow(2).sum(0) + 1e-8)).mean().item()
        return grad_V, {
            "mc_grad_ess": ess,
            "mc_grad_max_weight": weights.max().item(),
            "mc_grad_min_weight": weights.min().item(),
            "mc_grad_grad_V_norm": grad_norm.mean().item(),
        }

    # ------------------------------------------------------------------
    # Override: use mc_grad x0 during guided steps
    # ------------------------------------------------------------------

    def _recover_x0_for_guidance(
        self, atom_coords, sigma_t_vec, sigma_r_vec, decaf_module, nck,
    ):
        """Use importance-weighted mc_grad x0 when inside the sigma window.

        Also advances the per-step steering_t counter (guided path only).
        """
        # Update steering_t and step counter (guided path; _step_fast not called)
        num_steps = getattr(self, "_mc_grad_num_steps", 1) or 1
        self._mc_grad_steering_t = 1.0 - (getattr(self, "_mc_grad_step", 0) / num_steps)
        self._mc_grad_step = getattr(self, "_mc_grad_step", 0) + 1

        sigma_t_val = sigma_t_vec.item()
        cfg = self._mc_grad_cfg

        in_window = (
            self._in_guided_mode
            and cfg.mc_grad_sigma_min <= sigma_t_val <= cfg.mc_grad_sigma_max
        )
        if not in_window:
            return super()._recover_x0_for_guidance(
                atom_coords, sigma_t_vec, sigma_r_vec, decaf_module, nck
            )

        # Compute x0 base prediction
        x0_base, _ = self._decaf_x0(atom_coords, sigma_t_val, decaf_module, nck)

        # MC-GRAD correction: (σ_t/σ_data)² · ∇V from Algorithm 3 / Prop 5.1
        # The schedule multiplies raw sigmas by sigma_data, so σ_t in code
        # is σ_raw * σ_data.  The paper's formula uses raw σ².
        sigma_data = decaf_module.sigma_data
        grad_V, diag = self._alg3_value_gradient(
            atom_coords, sigma_t_val, decaf_module, nck
        )
        raw_sigma_sq = (sigma_t_vec[:, None, None] / sigma_data) ** 2
        correction = raw_sigma_sq * grad_V

        # Clip correction: per-atom max ~1.5 Å
        n_atoms = correction.shape[-2]
        max_corr_norm = 1.5 * sqrt(n_atoms)
        corr_norm = correction.norm(p=2, dim=(-1, -2), keepdim=True)
        clip_scale = torch.clamp(max_corr_norm / (corr_norm + 1e-8), max=1.0)
        correction = correction * clip_scale

        mc_grad_corr = correction
        clip_ratio = clip_scale.mean().item()
        logger.info(
            f"[MCGrad Alg3] sigma_t={sigma_t_val:.4f} K={cfg.num_mc_grad_particles} "
            f"snr_ratio={cfg.snr_ratio} ESS={diag['mc_grad_ess']:.2f} "
            f"grad_V_norm={diag['mc_grad_grad_V_norm']:.6f} "
            f"corr_raw={corr_norm.mean().item():.4f} "
            f"corr_clip={mc_grad_corr.norm(p=2, dim=(-1,-2)).mean().item():.4f} "
            f"x0_norm={x0_base.norm(p=2, dim=(-1,-2)).mean().item():.4f} "
            f"clip_ratio={clip_ratio:.4f}"
        )

        x0_corrected = x0_base + mc_grad_corr

        # Optional: iterative GD steps on x0 using potential gradients
        if cfg.mc_grad_use_gd_guidance:
            num_gd_steps = cfg.num_mc_grad_particles
            x0_guided = x0_corrected.clone()
            for _ in range(num_gd_steps):
                energy_gradient = torch.zeros_like(x0_guided)
                for potential in self._mc_grad_potentials:
                    params = potential.compute_parameters(self._mc_grad_steering_t)
                    if params.get("guidance_weight", 0) > 0:
                        energy_gradient = energy_gradient + params[
                            "guidance_weight"
                        ] * potential.compute_gradient(
                            x0_guided, self._mc_grad_feats, params
                        )
                x0_guided = x0_guided - energy_gradient
            gd_corr = (x0_guided - x0_corrected).norm(p=2, dim=(-1, -2)).mean().item()
            logger.info(f"  [GD guidance] steps={num_gd_steps} gd_corr_norm={gd_corr:.4f}")
            return x0_guided

        return x0_corrected

    # ------------------------------------------------------------------
    # Override sample(): wire in per-step reward context
    # ------------------------------------------------------------------

    def sample(
        self,
        decaf_module,
        atom_mask,
        num_sampling_steps=None,
        multiplicity=1,
        steering_args=None,
        **network_condition_kwargs,
    ):
        """Full sampling loop with MC-GRAD importance-weighted x0.

        Overrides the base sample() to populate per-step reward context
        (potentials, feats, steering_t) so that _recover_x0_for_guidance
        can evaluate the mc_grad reward.  steering_t is updated from the
        step counter (_mc_grad_step / _mc_grad_num_steps) maintained as
        instance state and incremented inside _recover_x0_for_guidance.
        """
        use_fk = self.cfg.fk_steering
        use_guidance = self.cfg.physical_guidance_update
        if steering_args is not None:
            use_fk = steering_args.get("fk_steering", use_fk)
            use_guidance = steering_args.get("physical_guidance_update", use_guidance)

        self._in_guided_mode = use_fk or use_guidance
        self._mc_grad_feats = network_condition_kwargs.get("feats")
        self._mc_grad_step = 0
        self._mc_grad_num_steps = default(num_sampling_steps, decaf_module.num_sampling_steps)

        if self._in_guided_mode and steering_args is not None:
            self._mc_grad_potentials = get_potentials(steering_args, boltz2=False)
        else:
            self._mc_grad_potentials = []

        # MC-GRAD incorporates the reward signal into x0 via
        # importance-weighted ∇V.  We run our own loop instead of
        # delegating to the parent, so that we apply the mc_grad
        # correction without FK resampling or gradient guidance.

        device = decaf_module.device
        num_sampling_steps = default(num_sampling_steps, decaf_module.num_sampling_steps)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
        shape = (*atom_mask.shape, 3)

        sigmas = decaf_module.sample_schedule(num_sampling_steps)
        sigma_pairs = list(zip(sigmas[:-1], sigmas[1:]))

        atom_coords = sigmas[0] * torch.randn(shape, device=device)
        nck = dict(network_condition_kwargs, multiplicity=multiplicity)

        for step_idx, (sigma_tm, sigma_t) in enumerate(sigma_pairs):
            sigma_tm_val = sigma_tm.item()
            sigma_t_val = sigma_t.item()
            sigma_t_vec = torch.tensor([sigma_tm_val], device=device)
            sigma_r_vec = torch.tensor([sigma_t_val], device=device)

            # Random augmentation
            if self.cfg.random_augmentation:
                random_R, random_tr = compute_random_augmentation(
                    multiplicity, device=atom_coords.device, dtype=atom_coords.dtype
                )
                atom_coords = atom_coords - atom_coords.mean(dim=-2, keepdims=True)
                atom_coords = torch.einsum("bmd,bds->bms", atom_coords, random_R) + random_tr

            with torch.no_grad():
                # MC-GRAD-corrected x0
                x0 = self._recover_x0_for_guidance(
                    atom_coords, sigma_t_vec, sigma_r_vec, decaf_module, nck
                )

                # Step from mc_grad-corrected x0 (γ-sampling)
                if self.cfg.use_sde:
                    gamma = self.cfg.sde_gamma
                    sigma_r_val = sigma_r_vec.item()
                    if sigma_r_val <= 0:
                        atom_coords = x0
                    elif gamma >= 1.0:
                        atom_coords = x0 + sigma_r_vec[:, None, None] * torch.randn_like(x0)
                    else:
                        sigma_tilde = sqrt(1.0 - gamma**2) * sigma_r_val
                        atom_coords = (
                            x0
                            + sigma_tilde * (atom_coords - x0) / sigma_t_vec[:, None, None]
                            + gamma * sigma_r_vec[:, None, None] * torch.randn_like(x0)
                        )
                else:
                    guided_phi = (atom_coords - x0) / sigma_t_vec[:, None, None]
                    atom_coords = (
                        atom_coords
                        - (sigma_t_vec - sigma_r_vec)[:, None, None] * guided_phi
                    )

        return dict(sample_atom_coords=atom_coords, diff_token_repr=None)


# ---------------------------------------------------------------------------
# MCTSFullRolloutDecafSampler
# ---------------------------------------------------------------------------


class MCTSFullRolloutDecafSampler(DecafSampler):
    """MCTS Full Rollout sampler for Decaf models.

    Implements Monte Carlo Tree Search over the denoising trajectory.  At each
    MCTS iteration:

      1. **Select** — traverse tree from root using UCT (or DTS) to a leaf
      2. **Expand** — take one SDE step forward to create child nodes
      3. **Simulate** — roll out each child to σ=0 via the Decaf SDE
      4. **Reward** — evaluate FK potentials on the terminal x0
      5. **Backpropagate** — soft backup values up each path

    The best trajectory is extracted after all iterations and, if it has not
    reached σ=0, completed with a standard Decaf SDE pass.

    Uses the boltz-decaf API: decaf_forward instead of a denoised
    estimator, FK potentials as the reward, and the simple sigma schedule
    from sample_schedule() instead of sigmas_and_gammas.
    """

    def __init__(self, cfg: MCTSFullRolloutDecafSamplerConfig | None = None):
        super().__init__(cfg if cfg is not None else MCTSFullRolloutDecafSamplerConfig())

        c = self._mcts_cfg
        self._total_denoiser_calls: int = 0
        self._total_reward_evals: int = 0
        self._timing: dict[str, float] = defaultdict(float)
        self._timing_counts: dict[str, int] = defaultdict(int)

        # Active branching timesteps resolved per sample() call
        self._active_branching_timesteps: list[int] = []
        self._num_sampling_steps: int = 0

        # Per-call reward context (set in sample())
        self._mcts_potentials: list[Any] = []
        self._mcts_feats: Any = None
        self._mcts_steering_t: float = 1.0

    @property
    def _mcts_cfg(self) -> MCTSFullRolloutDecafSamplerConfig:
        return self.cfg  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Flow map helpers
    # ------------------------------------------------------------------

    def _decaf_x0_batch(
        self,
        z: torch.Tensor,
        sigma: float,
        decaf_module,
        nck: dict,
    ) -> torch.Tensor:
        """Predict x0 for a batch of z at scalar sigma."""
        sigma_t = torch.tensor([sigma], device=z.device, dtype=z.dtype)
        sigma_zero = torch.zeros_like(sigma_t)
        phi = decaf_module.decaf_forward(
            z, sigma_t, sigma_zero,
            training=False,
            network_condition_kwargs=nck,
        )
        return z - sigma_t[:, None, None] * phi

    def _sde_step(self, x0: torch.Tensor, sigma_next: float) -> torch.Tensor:
        """SDE step: x_next = x0 + sigma_next * eps."""
        if sigma_next > 0:
            return x0 + sigma_next * torch.randn_like(x0)
        return x0

    # ------------------------------------------------------------------
    # Reward evaluation (FK potentials, negated → higher = better)
    # ------------------------------------------------------------------

    def _evaluate_reward(self, x0: torch.Tensor) -> torch.Tensor:
        """Evaluate FK potentials on x0. Returns reward [B] (higher = better)."""
        total = torch.zeros(x0.shape[0], device=x0.device)
        for potential in self._mcts_potentials:
            parameters = potential.compute_parameters(self._mcts_steering_t)
            if parameters.get("resampling_weight", 0) > 0:
                energy = potential.compute(x0, self._mcts_feats, parameters)
                total = total - parameters["resampling_weight"] * energy
        return total

    # ------------------------------------------------------------------
    # Schedule helpers
    # ------------------------------------------------------------------

    def _get_branching_timesteps(self, num_steps: int) -> list[int]:
        cfg = self._mcts_cfg
        if cfg.branching_timesteps:
            out = [t for t in cfg.branching_timesteps if not (0 <= t < num_steps)]
            if out:
                raise ValueError(
                    f"branching_timesteps out of range [0, {num_steps}): {out}"
                )
            return sorted(cfg.branching_timesteps)
        return list(range(num_steps))

    def _get_next_branching_step(self, step: int) -> int | None:
        steps = self._active_branching_timesteps
        if not steps:
            return step + 1
        idx = bisect_right(steps, step)
        return steps[idx] if idx < len(steps) else None

    def _num_children_allowed(self, visits: int) -> int:
        c = self._mcts_cfg
        return int(c.pw_k * (visits ** c.pw_alpha))

    # ------------------------------------------------------------------
    # MCTS selection
    # ------------------------------------------------------------------

    def _select(self, tree, root_id: int) -> int:
        """Traverse from root to an expandable leaf (UCT or DTS)."""
        from boltz.model.modules.mcts_utils import ParticleTree
        current = root_id
        terminal_step = (
            self._active_branching_timesteps[-1]
            if self._active_branching_timesteps else None
        )
        while True:
            children = tree.get_children(current)
            if not children:
                return current
            if tree.is_terminal(current, self._num_sampling_steps, terminal_step=terminal_step):
                return current
            visits = tree.graph.nodes[current]["visits"]
            step_idx = tree.graph.nodes[current]["step_idx"]
            n_allowed = self._num_children_allowed(visits)
            branching = (
                step_idx in self._active_branching_timesteps
                if self._active_branching_timesteps
                else True
            )
            if len(children) < n_allowed and branching:
                return current
            current = tree.select_child_uct(current)

    # ------------------------------------------------------------------
    # MCTS expansion
    # ------------------------------------------------------------------

    def _expand_batch(
        self,
        tree,
        node_ids: list[int],
        sigma_pairs: list[tuple[float, float]],
        decaf_module,
        nck: dict,
        atom_mask: torch.Tensor,
    ) -> list[int]:
        """Expand: take one SDE step per node, create expansion_children children each."""
        from boltz.model.modules.mcts_utils import ParticleTree

        if not node_ids:
            return []

        terminal_step = (
            self._active_branching_timesteps[-1]
            if self._active_branching_timesteps else None
        )
        terminal_nodes = [
            n for n in node_ids
            if tree.is_terminal(n, self._num_sampling_steps, terminal_step=terminal_step)
        ]
        expandable = [
            n for n in node_ids
            if not tree.is_terminal(n, self._num_sampling_steps, terminal_step=terminal_step)
        ]
        if not expandable:
            return node_ids

        coords = tree.batch_get_coords(expandable)
        device = coords.device

        # Expand atom_mask to node batch size
        if atom_mask.shape[0] != len(expandable):
            batch_mask = atom_mask[0:1].expand(len(expandable), -1)
        else:
            batch_mask = atom_mask

        all_children: list[int] = []

        for _ in range(self._mcts_cfg.expansion_children):
            node_coords = coords.clone()
            new_child_ids: list[int] = []

            for i, parent_id in enumerate(expandable):
                step_idx = tree.graph.nodes[parent_id]["step_idx"]
                next_step = self._get_next_branching_step(step_idx)
                if next_step is None or next_step >= len(sigma_pairs):
                    # Already at terminal
                    new_child_ids.append(parent_id)
                    continue

                sigma_t, sigma_next = sigma_pairs[step_idx][0], sigma_pairs[step_idx][1]

                # Random augmentation for single sample
                z_i = node_coords[i : i + 1]
                mask_i = batch_mask[i : i + 1]
                random_R, random_tr = compute_random_augmentation(
                    1, device=device, dtype=z_i.dtype
                )
                z_i = z_i - z_i.mean(dim=-2, keepdim=True)
                z_i = torch.einsum("bmd,bds->bms", z_i, random_R) + random_tr

                x0_i = self._decaf_x0_batch(z_i, sigma_t, decaf_module, nck)
                z_next_i = self._sde_step(x0_i, sigma_next)

                if not torch.all(torch.isfinite(z_next_i)):
                    logger.warning("[MCTS] Non-finite coords in expansion, skipping child")
                    new_child_ids.append(parent_id)
                    continue

                child_id = tree.add_node(
                    coords=z_next_i[0].clone(),
                    step_idx=next_step,
                    sigma=sigma_next,
                    parent_id=parent_id,
                    value=0.0,
                    visits=1,
                )
                new_child_ids.append(child_id)

            all_children.extend(new_child_ids)

        return all_children + terminal_nodes

    # ------------------------------------------------------------------
    # MCTS simulation (full rollout to sigma=0)
    # ------------------------------------------------------------------

    def _simulate_batch_full(
        self,
        tree,
        leaf_ids: list[int],
        sigma_pairs: list[tuple[float, float]],
        decaf_module,
        nck: dict,
        atom_mask: torch.Tensor,
    ) -> tuple[list[list[int]], torch.Tensor]:
        """Full rollout: simulate each leaf to σ=0, storing intermediate nodes."""
        if not leaf_ids:
            return [], torch.tensor([], device=atom_mask.device)

        coords = tree.batch_get_coords(leaf_ids)
        device = coords.device
        num_leaves = len(leaf_ids)

        if atom_mask.shape[0] != num_leaves:
            batch_mask = atom_mask[0:1].expand(num_leaves, -1)
        else:
            batch_mask = atom_mask

        current_steps = [tree.graph.nodes[n]["step_idx"] for n in leaf_ids]
        current_parent_ids = list(leaf_ids)
        num_pairs = len(sigma_pairs)

        # Step all leaves to sigma=0
        max_step = max(current_steps)
        max_iterations = num_pairs - max_step

        for iteration in range(max(0, max_iterations)):
            # Advance each sample individually (they may be at different steps)
            new_parent_ids: list[int] = []
            new_coords_list: list[torch.Tensor] = []

            for i, parent_id in enumerate(current_parent_ids):
                step = current_steps[i]
                if step >= num_pairs:
                    new_parent_ids.append(parent_id)
                    new_coords_list.append(coords[i])
                    continue

                sigma_t, sigma_next = sigma_pairs[step][0], sigma_pairs[step][1]

                z_i = coords[i : i + 1]
                random_R, random_tr = compute_random_augmentation(
                    1, device=device, dtype=z_i.dtype
                )
                z_i = z_i - z_i.mean(dim=-2, keepdim=True)
                z_i = torch.einsum("bmd,bds->bms", z_i, random_R) + random_tr

                x0_i = self._decaf_x0_batch(z_i, sigma_t, decaf_module, nck)
                z_next_i = self._sde_step(x0_i, sigma_next)

                if not torch.all(torch.isfinite(z_next_i)):
                    new_parent_ids.append(parent_id)
                    new_coords_list.append(coords[i])
                    current_steps[i] = num_pairs  # mark done
                    continue

                next_step = step + 1
                child_id = tree.add_node(
                    coords=z_next_i[0].clone(),
                    step_idx=next_step,
                    sigma=sigma_next,
                    parent_id=parent_id,
                    value=0.0,
                    visits=1,
                )
                new_parent_ids.append(child_id)
                new_coords_list.append(z_next_i[0])
                current_steps[i] = next_step

            current_parent_ids = new_parent_ids
            coords = torch.stack(new_coords_list, dim=0)

        # Collect terminal x0 predictions for reward
        terminal_x0_list: list[torch.Tensor] = []
        for i, parent_id in enumerate(current_parent_ids):
            step = current_steps[i]
            if step < num_pairs:
                sigma_t = sigma_pairs[min(step, num_pairs - 1)][0]
                z_i = coords[i : i + 1]
                x0_i = self._decaf_x0_batch(z_i, sigma_t, decaf_module, nck)
                terminal_x0_list.append(x0_i[0])
            else:
                terminal_x0_list.append(coords[i])

        terminal_x0 = torch.stack(terminal_x0_list, dim=0)

        # Evaluate rewards
        self._total_reward_evals += num_leaves
        rewards = self._evaluate_reward(terminal_x0)

        paths = [tree.get_path(nid) for nid in current_parent_ids]
        return paths, rewards

    # ------------------------------------------------------------------
    # Backpropagation + extraction
    # ------------------------------------------------------------------

    def _backpropagate(self, tree, paths: list[list[int]], rewards: torch.Tensor) -> None:
        tree.backup_paths(paths, rewards)

    def _extract_best_trajectory(
        self, tree, root_id: int
    ) -> tuple[list[torch.Tensor], int]:
        """Return (coord_list, deepest_step_idx) following the best path."""
        method = self._mcts_cfg.output_selection_method
        if method == "leaves":
            leaf = tree.get_best_leaf(root_id)
            path = tree.get_path(leaf)
        else:  # "root"
            path = tree.get_best_path(root_id)

        coords = [tree.graph.nodes[n]["coords"] for n in path]
        deepest_step = tree.graph.nodes[path[-1]]["step_idx"] if path else 0
        return coords, deepest_step

    # ------------------------------------------------------------------
    # Completion denoising (if tree hasn't reached sigma=0)
    # ------------------------------------------------------------------

    def _complete_denoising(
        self,
        atom_coords: torch.Tensor,
        start_steps: list[int],
        sigma_pairs: list[tuple[float, float]],
        decaf_module,
        nck: dict,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run remaining SDE steps from start_steps to sigma=0."""
        device = atom_coords.device
        num_pairs = len(sigma_pairs)
        coords = atom_coords.clone()
        current_steps = list(start_steps)

        max_remaining = max(num_pairs - s for s in current_steps)
        for _ in range(max_remaining):
            new_coords_list: list[torch.Tensor] = []
            for i in range(coords.shape[0]):
                step = current_steps[i]
                if step >= num_pairs:
                    new_coords_list.append(coords[i])
                    continue
                sigma_t, sigma_next = sigma_pairs[step]
                z_i = coords[i : i + 1]

                if atom_mask.shape[0] > 1:
                    mask_i = atom_mask[i : i + 1]
                else:
                    mask_i = atom_mask

                random_R, random_tr = compute_random_augmentation(
                    1, device=device, dtype=z_i.dtype
                )
                z_i = z_i - z_i.mean(dim=-2, keepdim=True)
                z_i = torch.einsum("bmd,bds->bms", z_i, random_R) + random_tr

                x0_i = self._decaf_x0_batch(z_i, sigma_t, decaf_module, nck)
                z_next_i = self._sde_step(x0_i, sigma_next)
                new_coords_list.append(z_next_i[0])
                current_steps[i] = step + 1

            coords = torch.stack(new_coords_list, dim=0)

        return coords

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def sample(
        self,
        decaf_module,
        atom_mask,
        num_sampling_steps=None,
        multiplicity=1,
        steering_args=None,
        **network_condition_kwargs,
    ) -> dict:
        """Sample structures using MCTS over the Decaf denoising trajectory.

        Overrides the parent sample() entirely — the MCTS loop is structurally
        incompatible with the parent's step-by-step loop.

        Falls back to plain DecafSampler when no FK/guidance is configured.
        """
        from boltz.model.modules.mcts_utils import ParticleTree

        # Resolve steering
        use_fk = self.cfg.fk_steering
        use_guidance = self.cfg.physical_guidance_update
        if steering_args is not None:
            use_fk = steering_args.get("fk_steering", use_fk)
            use_guidance = steering_args.get("physical_guidance_update", use_guidance)

        # Fall back to standard DecafSampler when no guidance / FK
        if not (use_fk or use_guidance):
            logger.info("[MCTS] No FK/guidance configured — falling back to DecafSampler")
            return DecafSampler.sample(
                self, decaf_module, atom_mask,
                num_sampling_steps=num_sampling_steps,
                multiplicity=multiplicity,
                steering_args=steering_args,
                **network_condition_kwargs,
            )

        # Set up reward context
        self._mcts_feats = network_condition_kwargs.get("feats")
        self._mcts_steering_t = 1.0
        if steering_args is not None:
            self._mcts_potentials = get_potentials(steering_args, boltz2=False)
        else:
            self._mcts_potentials = []

        # Reset counters
        self._total_denoiser_calls = 0
        self._total_reward_evals = 0
        self._timing = defaultdict(float)
        self._timing_counts = defaultdict(int)

        num_sampling_steps = default(num_sampling_steps, decaf_module.num_sampling_steps)
        self._num_sampling_steps = num_sampling_steps
        self._active_branching_timesteps = self._get_branching_timesteps(num_sampling_steps)

        device = decaf_module.device
        effective_num_roots = max(self._mcts_cfg.num_roots, multiplicity)
        atom_mask = atom_mask.repeat_interleave(effective_num_roots, 0)

        sigmas = decaf_module.sample_schedule(num_sampling_steps)
        sigma_pairs: list[tuple[float, float]] = [
            (sigmas[i].item(), sigmas[i + 1].item())
            for i in range(len(sigmas) - 1)
        ]
        init_sigma = sigmas[0].item()

        nck = dict(network_condition_kwargs, multiplicity=effective_num_roots)
        shape = (*atom_mask.shape, 3)

        logger.info(
            "[MCTS] steps=%d roots=%d sims=%d children=%d",
            num_sampling_steps, effective_num_roots,
            self._mcts_cfg.num_simulations, self._mcts_cfg.expansion_children,
        )

        # Initialize at sigma_max
        atom_coords = init_sigma * torch.randn(shape, device=device)

        # Build tree with one root per sample in the batch
        tree = ParticleTree(c_uct=self._mcts_cfg.c_uct, inv_temp=self._mcts_cfg.inv_temp)
        root_ids: list[int] = []
        for i in range(effective_num_roots):
            rid = tree.add_node(
                coords=atom_coords[i].clone(),
                step_idx=0,
                sigma=init_sigma,
                parent_id=None,
                value=0.0,
                visits=1,
            )
            root_ids.append(rid)

        # MCTS iterations
        for iteration in range(max(1, self._mcts_cfg.num_simulations)):
            # Update steering_t proportionally
            self._mcts_steering_t = 1.0 - (iteration / self._mcts_cfg.num_simulations)

            t0 = time.perf_counter()
            # Selection: one leaf per root
            selected = [self._select(tree, rid) for rid in root_ids]
            self._timing["select"] += time.perf_counter() - t0

            # Expansion
            t0 = time.perf_counter()
            expanded = self._expand_batch(
                tree, selected, sigma_pairs, decaf_module, nck, atom_mask
            )
            self._timing["expand"] += time.perf_counter() - t0

            # Separate terminal from non-terminal for simulation
            reward_terminal_step = (
                self._active_branching_timesteps[-1]
                if self._active_branching_timesteps else None
            )
            non_terminal = [
                n for n in expanded
                if not tree.is_terminal(n, num_sampling_steps, terminal_step=reward_terminal_step)
            ]
            terminal = [
                n for n in expanded
                if tree.is_terminal(n, num_sampling_steps, terminal_step=reward_terminal_step)
            ]

            # Simulation (full rollout for non-terminal)
            t0 = time.perf_counter()
            paths, rewards = self._simulate_batch_full(
                tree, non_terminal, sigma_pairs, decaf_module, nck, atom_mask
            )

            # Direct reward for terminal nodes
            if terminal:
                t_coords = tree.batch_get_coords(terminal)
                t_rewards = self._evaluate_reward(t_coords)
                for nid, rv in zip(terminal, t_rewards):
                    tree.graph.nodes[nid]["value"] = rv.item()
                    paths.append(tree.get_path(nid))
                rewards = torch.cat([rewards, t_rewards]) if rewards.numel() else t_rewards
            self._timing["simulate"] += time.perf_counter() - t0

            # Backpropagation
            t0 = time.perf_counter()
            if paths and rewards.numel():
                self._backpropagate(tree, paths, rewards)
            self._timing["backpropagate"] += time.perf_counter() - t0

            self._timing_counts["iterations"] += 1

        # Extract best trajectories
        best_roots = sorted(root_ids, key=lambda x: tree.graph.nodes[x]["value"], reverse=True)
        final_coords_list: list[torch.Tensor] = []
        deepest_steps: list[int] = []

        for rid in best_roots[:multiplicity]:
            traj, deepest_step = self._extract_best_trajectory(tree, rid)
            deepest_steps.append(deepest_step)
            if traj:
                final_coord = traj[-1]
                if not torch.all(torch.isfinite(final_coord)):
                    logger.warning("[MCTS] Non-finite final coords, falling back to root")
                    final_coord = tree.graph.nodes[rid]["coords"]
                final_coords_list.append(final_coord)
            else:
                final_coords_list.append(tree.graph.nodes[rid]["coords"])

        atom_coords = torch.stack(final_coords_list, dim=0)

        # Complete denoising if tree hasn't reached terminal depth
        num_pairs = len(sigma_pairs)
        if any(s < num_pairs for s in deepest_steps):
            remaining = [num_pairs - s for s in deepest_steps]
            logger.warning(
                "[MCTS] Tree did not reach terminal depth for %d/%d samples; "
                "completing with %s remaining steps",
                sum(1 for r in remaining if r > 0), len(remaining), remaining,
            )
            completion_mask = atom_mask[:multiplicity]
            atom_coords = self._complete_denoising(
                atom_coords, deepest_steps, sigma_pairs,
                decaf_module, nck, completion_mask,
            )

        if not torch.all(torch.isfinite(atom_coords)):
            logger.warning("[MCTS] Non-finite coordinates detected, replacing with zeros")
            atom_coords = torch.nan_to_num(atom_coords, nan=0.0, posinf=0.0, neginf=0.0)

        # Log statistics
        self._last_tree = tree
        stats = tree.get_statistics()
        logger.info("[MCTS] tree stats: %s", stats)
        logger.info(
            "[MCTS] compute: denoiser_calls=%d reward_evals=%d",
            self._total_denoiser_calls, self._total_reward_evals,
        )
        t = self._timing
        n = self._timing_counts
        total = sum(t.values())
        if total > 0:
            logger.info(
                "[MCTS] timing: select=%.1fs expand=%.1fs simulate=%.1fs backprop=%.1fs",
                t["select"], t["expand"], t["simulate"], t["backpropagate"],
            )

        return dict(sample_atom_coords=atom_coords, diff_token_repr=None)
