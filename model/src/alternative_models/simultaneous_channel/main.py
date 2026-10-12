"""Train and simulate the single simultaneous-observation channel model."""

from __future__ import annotations

from datetime import datetime
import hashlib
import time


def runtime_log(message: str) -> None:
    """Print one timestamped stage message immediately to Slurm stdout."""
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[{timestamp}] simultaneous-channel: {message}", flush=True)


if __name__ == "__main__":
    runtime_log("Python entry point started; importing NumPy and TensorFlow")

import numpy as np
import tensorflow as tf

from ...main_vae import config
from ...main_vae.simulate import run_simulation
from ...main_vae.train import train_model
from .model import SimultaneousObservationChannelModel

if __name__ == "__main__":
    runtime_log("Python imports and project configuration completed")


ARCHITECTURE_TAG = "simultaneous_channel"
MAX_ARTIFACT_STEM_BYTES = 240


def generate_simultaneous_channel_batch(
    batch_size: int,
    time_steps: int,
) -> tf.Tensor:
    """Generate complete trials for the one simultaneous model input."""
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
    return tf.convert_to_tensor(
        reward_support[reward_indices, np.newaxis],
        dtype=tf.float32,
    )


def _compact_artifact_stem(full_stem: str) -> str:
    """Bound artifact stems while retaining deterministic identity."""
    if not full_stem.isascii():
        raise ValueError("Simultaneous-channel artifact stems must be ASCII.")
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


def simultaneous_channel_artifact_name(
    lambda_value: float,
    alpha_value: float,
    model_seed: int | None = None,
) -> str:
    """Return a unique artifact stem for one simultaneous-channel run."""
    selected_seed = config.seed if model_seed is None else model_seed
    base_name = (
        f"{ARCHITECTURE_TAG}_lambda_{lambda_value}"
        f"_alpha_{alpha_value}_seed_{selected_seed}"
        f"_tree_{config.tree_size}"
    )
    if config.tree_size == 31:
        base_name += f"_{config.tree_type}"
    full_stem = (
        base_name
        + config.reconstruction_head_artifact_suffix()
        + config.reconstruction_range_artifact_suffix()
        + f"_recon_units_{config.reconstruction_hidden_units}"
        + "_recon_scope_all_nodes"
        + f"_rnn_units_{config.rnn_units}"
        + f"_latent_dim_{config.latent_dim}"
        + "_kl_once"
        + config.epoch_artifact_suffix()
        + config.critic_artifact_suffix()
        + config.gradient_scale_artifact_suffix()
        + config.final_learning_rate_artifact_suffix()
        + config.learning_rate_schedule_artifact_suffix()
        + config.parameter_multiplier_artifact_suffix()
        + (
            config.initial_critic_coefficient_artifact_suffix()
            if config.critic_loss_enabled
            else ""
        )
        + config.action_entropy_artifact_suffix()
        + config.reward_normalization_artifact_suffix()
    )
    return _compact_artifact_stem(full_stem)


def build_simultaneous_channel_model(
    lambda_value: float,
    alpha_value: float,
) -> SimultaneousObservationChannelModel:
    """Build the configured one-channel simultaneous model."""
    return SimultaneousObservationChannelModel(
        rnn_units=config.rnn_units,
        latent_dim=config.latent_dim,
        time_steps=config.time_steps,
        num_paths=config.num_paths,
        index_path_map=config.index_path_map,
        alpha=alpha_value,
        lambda_=lambda_value,
        reward_normalization_constant=config.reward_normalization_constant,
        max_action_gradient_scale=config.max_action_gradient_scale,
        reconstruction_head_type=config.reconstruction_head_type,
        reconstruction_hidden_units=config.reconstruction_hidden_units,
        critic_loss_enabled=config.critic_loss_enabled,
        parameter_multiplier=config.parameter_multiplier,
        action_entropy_bonus=config.action_entropy_bonus,
    )


def main() -> None:
    """Train requested parameter cells and simulate their final weights."""
    process_start = time.perf_counter()
    runtime_log(
        "configuration: "
        f"tree={config.tree_size}:{config.tree_type}, "
        f"mode={config.train_mode}, epochs={config.epochs}, "
        f"batches_per_epoch={config.trials_per_epoch}, "
        f"batch_size={config.batch_size}, simulation_trials="
        f"{config.num_trials}, rnn_units={config.rnn_units}, "
        f"latent_dim={config.latent_dim}, critic="
        f"{config.critic_loss_enabled}"
    )

    if config.train_mode == "train":
        for lambda_value in config.lambda_values:
            for alpha_value in config.alpha_values:
                cell_label = (
                    f"lambda={lambda_value},alpha={alpha_value},"
                    f"seed={config.seed},tree={config.tree_size}:"
                    f"{config.tree_type}"
                )
                build_start = time.perf_counter()
                runtime_log(f"building model for {cell_label}")
                model = build_simultaneous_channel_model(
                    lambda_value,
                    alpha_value,
                )
                runtime_log(
                    f"model object built for {cell_label} in "
                    f"{time.perf_counter() - build_start:.2f}s"
                )
                training_start = time.perf_counter()
                runtime_log(f"training started for {cell_label}")
                train_model(
                    model=model,
                    epochs=config.epochs,
                    trials_per_epoch=config.trials_per_epoch,
                    batch_size=config.batch_size,
                    time_steps=config.time_steps,
                    weight_dir=config.weight_dir,
                    checkpoint_dir=config.checkpoint_dir,
                    training_log_dir=config.training_log_dir,
                    model_name=simultaneous_channel_artifact_name(
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
                    batch_data_generator=generate_simultaneous_channel_batch,
                    initial_critic_coefficient=(
                        config.initial_critic_coefficient
                    ),
                    progress_timing_label=(
                        f"simultaneous-channel {cell_label}"
                    ),
                )
                runtime_log(
                    f"training returned for {cell_label} after "
                    f"{time.perf_counter() - training_start:.2f}s"
                )

    simulation_start = time.perf_counter()
    runtime_log("post-training simulation started")
    run_simulation(
        config,
        model_builder=build_simultaneous_channel_model,
        model_name_builder=simultaneous_channel_artifact_name,
        simulation_metadata={
            "model_architecture": "single_simultaneous_channel",
            "critic_loss_enabled": config.critic_loss_enabled,
            "initial_critic_coefficient": (
                config.initial_critic_coefficient
                if config.initial_critic_coefficient is not None
                else "auto"
            ),
            "observation_schedule": "all_rewards_simultaneous",
            "channel_count": 1,
            "kl_terms_per_trial": 1,
            "rnn_units": config.rnn_units,
            "latent_dim": config.latent_dim,
            "reconstruction_loss_scope": "all_nodes",
            "simulation_forward_pass": "single_full_batch",
            "simulation_random_draw_grouping": "batched",
        },
        reuse_full_batch_outputs=True,
        progress_timing_label="simultaneous-channel simulation",
    )
    runtime_log(
        "post-training simulation completed in "
        f"{time.perf_counter() - simulation_start:.2f}s; "
        f"total Python runtime={time.perf_counter() - process_start:.2f}s"
    )


if __name__ == "__main__":
    main()
