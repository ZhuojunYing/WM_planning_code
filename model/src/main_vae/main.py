"""Run model training and simulation from command-line configuration."""

from datetime import datetime
import os
import time


_RUNTIME_DIAGNOSTICS = os.environ.get(
    "WM_PLANNING_COMPARISON_RUNTIME_DIAGNOSTICS",
    "0",
) == "1"


def runtime_log(message: str) -> None:
    """Print a timestamped main-VAE runtime milestone when requested."""
    if not _RUNTIME_DIAGNOSTICS:
        return
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[{timestamp}] main-VAE: {message}", flush=True)


runtime_log("Python entry point started; importing model configuration")

from . import config
from .model import VariationalRNN, build_decoder, build_encoder
from .simulate import run_simulation
from .train import train_model

_RUNTIME_DIAGNOSTICS = _RUNTIME_DIAGNOSTICS or config.tree_size == 31
runtime_log("Python imports and project configuration completed")


def main() -> None:
    """Create configured models and run training followed by simulation.

    Args:
        None.

    Returns:
        None.
    """
    process_start = time.perf_counter()
    runtime_log(
        "configuration: "
        f"tree={config.tree_size}:{config.tree_type}, "
        f"condition={config.model_condition}, "
        f"mode={config.train_mode}, epochs={config.epochs}, "
        f"batches_per_epoch={config.trials_per_epoch}, "
        f"batch_size={config.batch_size}, simulation_trials="
        f"{config.num_trials}, rnn_units={config.rnn_units}, "
        f"latent_dim={config.latent_dim}"
    )
    if config.train_mode == "train":
        for lambda_ in config.lambda_values:
            for alpha in config.alpha_values:
                cell_label = (
                    f"lambda={lambda_},alpha={alpha},seed={config.seed},"
                    f"tree={config.tree_size}:{config.tree_type}"
                )
                model_name = config.model_artifact_name(lambda_, alpha)

                build_start = time.perf_counter()
                runtime_log(f"building training model for {cell_label}")
                encoder = build_encoder(
                    config.rnn_units * 2,
                    config.latent_dim,
                    config.rnn_units,
                )
                decoder = build_decoder(
                    config.latent_dim,
                    2 * config.rnn_units,
                    config.rnn_units,
                )

                vrnn_model = VariationalRNN(
                    encoder=encoder,
                    decoder=decoder,
                    rnn_units=config.rnn_units,
                    latent_dim=config.latent_dim,
                    time_steps=config.time_steps,
                    num_paths=config.num_paths,
                    index_path_map=config.index_path_map,
                    path_map=config.path_map,
                    path_cov_mat=config.path_cov_mat,
                    alpha=alpha,
                    lambda_=lambda_,
                    tree_type=config.tree_type,
                    reward_normalization_constant=(
                        config.reward_normalization_constant
                    ),
                    max_action_gradient_scale=(
                        config.max_action_gradient_scale
                    ),
                    reconstruction_head_type=(
                        config.reconstruction_head_type
                    ),
                    reconstruction_hidden_units=(
                        config.reconstruction_hidden_units
                    ),
                    reconstruction_loss_scope=(
                        config.reconstruction_loss_scope
                    ),
                    critic_loss_enabled=config.critic_loss_enabled,
                    parameter_multiplier=config.parameter_multiplier,
                    action_entropy_bonus=config.action_entropy_bonus,
                    model_condition=config.model_condition,
                )
                runtime_log(
                    f"training model object built for {cell_label} in "
                    f"{time.perf_counter() - build_start:.2f}s"
                )

                training_start = time.perf_counter()
                runtime_log(f"training started for {cell_label}")
                train_model(
                    model=vrnn_model,
                    epochs=config.epochs,
                    trials_per_epoch=config.trials_per_epoch,
                    batch_size=config.batch_size,
                    time_steps=config.time_steps,
                    weight_dir=config.weight_dir,
                    checkpoint_dir=config.checkpoint_dir,
                    training_log_dir=config.training_log_dir,
                    model_name=model_name,
                    final_learning_rate=config.final_learning_rate,
                    late_lr_plateau_start_epoch=(
                        config.late_lr_plateau_start_epoch
                    ),
                    late_lr_plateau_end_epoch=(
                        config.late_lr_plateau_end_epoch
                    ),
                    initial_critic_coefficient=(
                        config.initial_critic_coefficient
                    ),
                    progress_timing_label=(
                        f"main-VAE {cell_label}"
                        if _RUNTIME_DIAGNOSTICS
                        else None
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
        simulation_metadata={
            "simulation_forward_pass": "single_full_batch",
            "simulation_random_draw_grouping": "batched",
            "model_condition": config.model_condition,
            "action_loss_enabled": config.action_loss_enabled,
        },
        reuse_full_batch_outputs=True,
        progress_timing_label=(
            "main-VAE simulation"
            if _RUNTIME_DIAGNOSTICS
            else None
        ),
    )
    runtime_log(
        "post-training simulation completed in "
        f"{time.perf_counter() - simulation_start:.2f}s; "
        f"total Python runtime={time.perf_counter() - process_start:.2f}s"
    )


if __name__ == "__main__":
    main()
