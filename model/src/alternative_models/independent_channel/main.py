"""Train and simulate the independent per-node channel comparison model."""

from __future__ import annotations

import hashlib

import numpy as np
import tensorflow as tf

from ...main_vae import config
from ...main_vae.simulate import run_simulation
from ...main_vae.train import train_model
from .model import IndependentNodeChannelModel


ARCHITECTURE_TAG = "independent_channels"
MAX_ARTIFACT_STEM_BYTES = 240


def generate_independent_channel_batch(
    batch_size: int,
    time_steps: int,
) -> tf.Tensor:
    """Generate a complete training batch with one vectorized RNG draw."""
    reward_support = np.asarray(
        (0, 1)
        if time_steps == 2
        else (-4, -3, -2, -1, 1, 2, 3, 4),
        dtype=np.float32,
    )
    reward_indices = np.random.randint(
        0,
        reward_support.size,
        size=(batch_size, time_steps),
    )
    reward_values = reward_support[reward_indices]
    return tf.convert_to_tensor(
        reward_values[:, :, np.newaxis],
        dtype=tf.float32,
    )


def _compact_artifact_stem(full_stem: str) -> str:
    """Bound artifact stems while retaining a deterministic identity."""
    if not full_stem.isascii():
        raise ValueError("Independent-channel artifact stems must be ASCII.")
    if len(full_stem.encode("ascii")) <= MAX_ARTIFACT_STEM_BYTES:
        return full_stem

    configuration_hash = hashlib.sha256(
        full_stem.encode("ascii")
    ).hexdigest()[:12]
    separator = f"_cfg_{configuration_hash}_"
    retained_length = MAX_ARTIFACT_STEM_BYTES - len(separator)
    prefix_length = (retained_length + 1) // 2
    suffix_length = retained_length - prefix_length
    return (
        full_stem[:prefix_length]
        + separator
        + full_stem[-suffix_length:]
    )


def independent_channel_artifact_name(
    lambda_value: float,
    alpha_value: float,
    model_seed: int | None = None,
) -> str:
    """Return an artifact stem that cannot collide with sequential models."""
    selected_seed = config.seed if model_seed is None else model_seed
    base_name = (
        f"{ARCHITECTURE_TAG}_lambda_{lambda_value}"
        f"_alpha_{alpha_value}_seed_{selected_seed}"
        f"_tree_{config.tree_size}"
    )
    if config.tree_size == 31:
        base_name += f"_{config.tree_type}"
    capacity_suffix = ""
    if (
        config.independent_channel_rnn_units != 64
        or config.independent_channel_latent_dim != 16
    ):
        capacity_suffix = (
            f"_rnn_units_{config.independent_channel_rnn_units}"
            f"_latent_dim_{config.independent_channel_latent_dim}"
        )
    full_stem = (
        base_name
        + config.reconstruction_head_artifact_suffix()
        + capacity_suffix
        + config.reconstruction_range_artifact_suffix()
        + f"_recon_units_{config.reconstruction_hidden_units}"
        + "_recon_scope_all_channels"
        + f"_kl_{config.channel_kl_aggregation}"
        + f"_time_{config.channel_kl_time_weighting}"
        + config.epoch_artifact_suffix()
        + config.critic_artifact_suffix()
        + config.gradient_scale_artifact_suffix()
        + config.final_learning_rate_artifact_suffix()
        + config.learning_rate_schedule_artifact_suffix()
        + config.parameter_multiplier_artifact_suffix()
        + config.action_entropy_artifact_suffix()
        + config.reward_normalization_artifact_suffix()
    )
    return _compact_artifact_stem(full_stem)


def build_independent_channel_model(
    lambda_value: float,
    alpha_value: float,
) -> IndependentNodeChannelModel:
    """Build one configured independent-channel comparison model."""
    return IndependentNodeChannelModel(
        rnn_units=config.independent_channel_rnn_units,
        latent_dim=config.independent_channel_latent_dim,
        time_steps=config.time_steps,
        num_paths=config.num_paths,
        index_path_map=config.index_path_map,
        alpha=alpha_value,
        lambda_=lambda_value,
        reward_normalization_constant=config.reward_normalization_constant,
        max_action_gradient_scale=config.max_action_gradient_scale,
        reconstruction_head_type=config.reconstruction_head_type,
        reconstruction_hidden_units=config.reconstruction_hidden_units,
        kl_aggregation=config.channel_kl_aggregation,
        kl_time_weighting=config.channel_kl_time_weighting,
        parameter_multiplier=config.parameter_multiplier,
        action_entropy_bonus=config.action_entropy_bonus,
    )


def main() -> None:
    """Train requested grid cells and run separately tagged simulations."""
    if config.critic_loss_enabled:
        raise ValueError(
            "The independent-channel model has no critic. Invoke this entry "
            "point with --no-critic-loss."
        )

    if config.train_mode == "train":
        for lambda_value in config.lambda_values:
            for alpha_value in config.alpha_values:
                model = build_independent_channel_model(
                    lambda_value,
                    alpha_value,
                )
                train_model(
                    model=model,
                    epochs=config.epochs,
                    trials_per_epoch=config.trials_per_epoch,
                    batch_size=config.batch_size,
                    time_steps=config.time_steps,
                    weight_dir=config.weight_dir,
                    checkpoint_dir=config.checkpoint_dir,
                    training_log_dir=config.training_log_dir,
                    model_name=independent_channel_artifact_name(
                        lambda_value,
                        alpha_value,
                    ),
                    final_learning_rate=config.final_learning_rate,
                    late_lr_plateau_start_epoch=(
                        config.late_lr_plateau_start_epoch
                    ),
                    late_lr_plateau_end_epoch=(
                        config.late_lr_plateau_end_epoch
                    ),
                    batch_data_generator=generate_independent_channel_batch,
                )

    run_simulation(
        config,
        model_builder=build_independent_channel_model,
        model_name_builder=independent_channel_artifact_name,
        simulation_metadata={
            "model_architecture": "independent_node_channels",
            "critic_loss_enabled": False,
            "reconstruction_loss_scope": "all_channels",
            "channel_kl_aggregation": config.channel_kl_aggregation,
            "channel_kl_time_weighting": (
                config.channel_kl_time_weighting
            ),
            "channel_kl_time_weights": ",".join(
                str(weight)
                for weight in (
                    range(config.time_steps, 0, -1)
                    if config.channel_kl_time_weighting == "remaining_steps"
                    else (1 for _ in range(config.time_steps))
                )
            ),
            "independent_channel_rnn_units": (
                config.independent_channel_rnn_units
            ),
            "independent_channel_latent_dim": (
                config.independent_channel_latent_dim
            ),
            "channel_forward_implementation": "vectorized_stacked_weights",
            "simulation_forward_pass": "single_full_batch",
            "simulation_random_draw_grouping": "batched",
        },
        reuse_full_batch_outputs=True,
    )


if __name__ == "__main__":
    main()
