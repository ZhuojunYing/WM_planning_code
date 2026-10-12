"""Simulation and CSV export utilities for trained or analytic inference."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from types import ModuleType
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from .model import VariationalRNN, build_decoder, build_encoder
from . import helper
from project_paths import job_temporary_path


def calculate_path_rewards_sim(
    index_path_map: dict[int, list[int]],
    trial_rewards: list[int],
) -> np.ndarray:
    """Calculate path rewards for one simulated trial.

    Args:
        index_path_map: Path identifiers mapped to one-based node indices.
        trial_rewards: Reward values for each node in the trial.

    Returns:
        One-dimensional array of summed rewards for each path.
    """
    path_rewards = []
    for _, node_indices in index_path_map.items():
        path_sum = sum(trial_rewards[node - 1] for node in node_indices)
        path_rewards.append(path_sum)
    return np.array(path_rewards, dtype=float)


def _slice_batched_model_outputs(
    outputs: tuple[object, ...] | list[object],
    trial_index: int,
    batch_size: int,
) -> tuple[object, ...]:
    """Return one trial while preserving a leading batch dimension."""
    trial_outputs = []
    for output in outputs:
        output_shape = getattr(output, "shape", None)
        if (
            output_shape is not None
            and len(output_shape) > 0
            and output_shape[0] is not None
            and int(output_shape[0]) == batch_size
        ):
            trial_outputs.append(output[trial_index : trial_index + 1])
        else:
            trial_outputs.append(output)
    return tuple(trial_outputs)


def _validate_batched_trial_metadata(
    metadata: dict[str, object],
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Require one scalar metadata value per trial and output column."""
    validated = {}
    for column_name, values in metadata.items():
        column_values = np.asarray(values)
        if column_values.ndim != 1 or column_values.shape[0] != batch_size:
            raise ValueError(
                "Batched trial metadata column "
                f"'{column_name}' must have shape [{batch_size}]."
            )
        validated[column_name] = column_values
    return validated


def run_simulation(
    config: ModuleType,
    model_builder: Callable[[float, float], tf.keras.Model] | None = None,
    model_name_builder: Callable[[float, float], str] | None = None,
    simulation_metadata: dict[str, object] | None = None,
    node_output_metadata_builder: (
        Callable[[tuple[object, ...] | list[object], int], dict[str, object]]
        | None
    ) = None,
    trial_output_metadata_builder: (
        Callable[
            [tuple[object, ...] | list[object], list[int], np.ndarray],
            dict[str, object],
        ]
        | None
    ) = None,
    batch_trial_output_metadata_builder: (
        Callable[
            [
                tuple[object, ...] | list[object],
                list[list[int]],
                np.ndarray,
            ],
            dict[str, object],
        ]
        | None
    ) = None,
    reuse_full_batch_outputs: bool = False,
    require_saved_weights: bool = True,
    progress_timing_label: str | None = None,
) -> None:
    """Run models over simulated trials and export node-level results.

    Args:
        config: Imported configuration module containing model, tree, and path
            settings.
        model_builder: Optional comparison-model factory. The ordinary
            sequential model is built when omitted.
        model_name_builder: Optional artifact-name factory paired with
            ``model_builder``.
        simulation_metadata: Optional constant columns added to every row.
        node_output_metadata_builder: Optional callback that derives extra
            node-level columns from one model prediction and zero-based node
            index. Ordinary simulations leave this unset.
        trial_output_metadata_builder: Optional callback that derives metadata
            once per trial from model outputs, true node rewards, and true path
            rewards. The resulting columns are repeated across that trial's
            node rows.
        batch_trial_output_metadata_builder: Optional callback that derives
            scalar metadata columns for every trial from the full-batch model
            outputs. Each returned value must have shape ``[num_trials]``.
        reuse_full_batch_outputs: Whether row generation should slice the
            already-computed full-batch inference rather than invoke Keras
            ``predict`` separately for every trial. Model families opting in
            must either be deterministic or accept batched regrouping of
            independent stochastic draws.
        require_saved_weights: Whether a matching saved-weight file must be
            present and loaded. Analytic models with no learned parameters set
            this to ``False``.
        progress_timing_label: Optional label enabling timestamped, flushed
            timing messages for each major simulation stage.

    Returns:
        None.
    """
    num_trials = int(config.num_trials)
    time_steps = config.time_steps
    timing_enabled = bool(progress_timing_label)

    def progress_log(message: str) -> None:
        if not timing_enabled:
            return
        timestamp = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )
        print(
            f"[{timestamp}] {progress_timing_label}: {message}",
            flush=True,
        )

    simulation_start = time.perf_counter()
    progress_log(
        f"simulation setup started; trials={num_trials}, "
        f"nodes_per_trial={time_steps}, "
        f"reuse_full_batch_outputs={reuse_full_batch_outputs}"
    )
    if (
        batch_trial_output_metadata_builder is not None
        and not reuse_full_batch_outputs
    ):
        raise ValueError(
            "Batched trial metadata requires reuse_full_batch_outputs=True."
        )
    constant_metadata = (
        {} if simulation_metadata is None else dict(simulation_metadata)
    )

    stage_start = time.perf_counter()
    rewards_list = helper.generate_reward_values(
        num_trials,
        time_steps,
    ).tolist()
    progress_log(
        "simulation rewards generated in "
        f"{time.perf_counter() - stage_start:.2f}s"
    )

    for lambda_ in config.lambda_values:
        for alpha in config.alpha_values:
            cell_start = time.perf_counter()
            cell_label = f"lambda={lambda_},alpha={alpha}"
            progress_log(f"building simulation model for {cell_label}")
            model_name = (
                config.model_artifact_name(lambda_, alpha)
                if model_name_builder is None
                else model_name_builder(lambda_, alpha)
            )

            if model_builder is None:
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
            else:
                vrnn_model = model_builder(lambda_, alpha)
            reward_normalization_constant = getattr(
                vrnn_model,
                "reward_normalization_constant",
                config.reward_normalization_constant,
            )
            reward_normalization_scheme = getattr(
                vrnn_model,
                "reward_normalization_scheme",
                config.reward_normalization_scheme,
            )
            progress_log(
                f"simulation model object built for {cell_label} in "
                f"{time.perf_counter() - cell_start:.2f}s"
            )

            if require_saved_weights:
                model_name_candidates = (
                    config.model_artifact_name_candidates(lambda_, alpha)
                    if model_name_builder is None
                    and hasattr(config, "model_artifact_name_candidates")
                    else (model_name,)
                )
                weights_file_path = next(
                    (
                        config.weight_dir / (candidate + ".weights.h5")
                        for candidate in model_name_candidates
                        if (
                            config.weight_dir
                            / (candidate + ".weights.h5")
                        ).exists()
                    ),
                    config.weight_dir / (model_name + ".weights.h5"),
                )
                if not weights_file_path.exists():
                    progress_log(
                        f"matching weights missing for {cell_label}; skipping"
                    )
                    continue

            stage_start = time.perf_counter()
            dummy_input = tf.zeros(
                (1, config.time_steps, 1),
                dtype=tf.float32,
            )
            _ = vrnn_model(dummy_input, training=False)
            if require_saved_weights:
                try:
                    vrnn_model.load_weights(str(weights_file_path))
                except Exception:
                    progress_log(
                        f"weight loading failed for {cell_label}; skipping"
                    )
                    continue
            progress_log(
                f"model variables initialized and weights loaded for "
                f"{cell_label} in {time.perf_counter() - stage_start:.2f}s"
            )

            feature_dim = 1
            values = np.array(rewards_list)
            input_data = tf.constant(values, dtype=tf.float32)
            input_data = tf.reshape(
                input_data,
                [num_trials, time_steps, feature_dim],
            )

            stage_start = time.perf_counter()
            full_batch_outputs = vrnn_model(input_data, training=False)
            mutual_information_cost = float(full_batch_outputs[10])
            progress_log(
                f"full-batch inference completed for {cell_label} in "
                f"{time.perf_counter() - stage_start:.2f}s"
            )

            stage_start = time.perf_counter()
            true_path_rewards_by_trial = np.stack(
                [
                    calculate_path_rewards_sim(
                        config.index_path_map,
                        rewards,
                    )
                    for rewards in rewards_list
                ],
                axis=0,
            )
            batched_trial_metadata = (
                {}
                if batch_trial_output_metadata_builder is None
                else _validate_batched_trial_metadata(
                    batch_trial_output_metadata_builder(
                        full_batch_outputs,
                        rewards_list,
                        true_path_rewards_by_trial,
                    ),
                    num_trials,
                )
            )
            progress_log(
                f"path rewards and batched metadata completed for "
                f"{cell_label} in {time.perf_counter() - stage_start:.2f}s"
            )

            simulation_rows = []
            row_assembly_start = time.perf_counter()

            for trial_index, rewards in enumerate(rewards_list[:num_trials]):
                if reuse_full_batch_outputs:
                    outputs = _slice_batched_model_outputs(
                        full_batch_outputs,
                        trial_index,
                        num_trials,
                    )
                else:
                    rewards_tensor = tf.constant(rewards, dtype=tf.float32)
                    rewards_tensor = tf.reshape(
                        rewards_tensor,
                        [1, -1, 1],
                    )
                    outputs = vrnn_model.predict(rewards_tensor, verbose=0)

                category_outputs = outputs[0]

                all_category_probs = category_outputs[0]
                final_category_probs = all_category_probs[-1, :, :]
                log_probs = tf.math.log(
                    tf.constant(final_category_probs, dtype=tf.float32)
                    + 1e-8
                )
                sampled_indices = tf.random.categorical(
                    log_probs,
                    num_samples=1,
                )
                category_indices = tf.cast(
                    tf.squeeze(sampled_indices, axis=-1),
                    tf.int32,
                )
                category_values_tf = tf.constant(
                    config.reconstruction_values,
                    dtype=tf.float32,
                )
                scalar_reconstructions = tf.gather(
                    category_values_tf,
                    category_indices,
                )
                modal_category_indices = np.argmax(
                    final_category_probs,
                    axis=-1,
                )
                category_values = np.asarray(
                    category_values_tf.numpy(),
                    dtype=float,
                )
                modal_reconstructions = category_values[
                    modal_category_indices
                ]
                expected_reconstructions = np.sum(
                    np.asarray(final_category_probs, dtype=float)
                    * category_values[None, :],
                    axis=-1,
                )

                path_rewards = true_path_rewards_by_trial[trial_index]
                max_path_reward = float(np.max(path_rewards))
                min_path_reward = float(np.min(path_rewards))
                action_outputs = outputs[1]
                action_policy = np.asarray(
                    action_outputs,
                    dtype=float,
                ).reshape(-1)
                action_policy = action_policy / np.sum(action_policy)
                path_rewards_arr = path_rewards
                trial_output_metadata = (
                    {
                        column_name: column_values[trial_index].item()
                        for column_name, column_values in (
                            batched_trial_metadata.items()
                        )
                    }
                    if batched_trial_metadata
                    else (
                        {}
                        if trial_output_metadata_builder is None
                        else trial_output_metadata_builder(
                            outputs,
                            rewards,
                            path_rewards_arr,
                        )
                    )
                )
                best_idx = int(
                    np.random.choice(config.num_paths, p=action_policy)
                )
                V = float(path_rewards_arr[best_idx])
                for node_number in range(1, config.time_steps + 1):
                    node_idx = node_number - 1
                    node_output_metadata = (
                        {}
                        if node_output_metadata_builder is None
                        else node_output_metadata_builder(outputs, node_idx)
                    )
                    true_category_index = int(
                        np.clip(
                            round(
                                config.reconstruction_max_reward
                                - rewards[node_idx]
                            ),
                            0,
                            config.num_categories - 1,
                        )
                    )

                    node_paths = config.node_path_map[str(node_number)]
                    node_path_rewards = [
                        path_rewards[path_index]
                        for path_index in node_paths
                    ]

                    is_leaf = (
                        node_number in list(config.path_leaf_dict.values())
                    )

                    simulation_rows.append(
                        {
                            "graph": int(trial_index),
                            "node": int(node_number),
                            "actual_reward": float(rewards[node_idx]),
                            "estimated_reward": float(
                                scalar_reconstructions[node_idx].numpy()
                            ),
                            "modal_estimated_reward": float(
                                modal_reconstructions[node_idx]
                            ),
                            "expected_estimated_reward": float(
                                expected_reconstructions[node_idx]
                            ),
                            "true_reward_probability": float(
                                final_category_probs[
                                    node_idx,
                                    true_category_index,
                                ]
                            ),
                            "max_path_reward": max_path_reward,
                            "min_path_reward": min_path_reward,
                            "path_reward": float(np.mean(node_path_rewards)),
                            "V": V,
                            "MI_cost": mutual_information_cost,
                            "is_leaf": bool(is_leaf),
                            "reconstruction_hidden_units": int(
                                config.reconstruction_hidden_units
                            ),
                            "rnn_units": int(
                                getattr(
                                    vrnn_model,
                                    "rnn_units",
                                    config.rnn_units,
                                )
                            ),
                            "latent_dim": int(
                                getattr(
                                    vrnn_model,
                                    "latent_dim",
                                    config.latent_dim,
                                )
                            ),
                            "reconstruction_loss_scope": (
                                config.reconstruction_loss_scope
                            ),
                            "reconstruction_min_reward": int(
                                config.reconstruction_min_reward
                            ),
                            "reconstruction_max_reward": int(
                                config.reconstruction_max_reward
                            ),
                            "reconstruction_category_count": int(
                                config.num_categories
                            ),
                            "parameter_multiplier": float(
                                getattr(
                                    vrnn_model,
                                    "parameter_multiplier",
                                    1.0,
                                )
                            ),
                            "parameter_multiplier_explicit": bool(
                                getattr(
                                    config,
                                    "parameter_multiplier_explicit",
                                    False,
                                )
                            ),
                            "action_entropy_bonus": float(
                                getattr(
                                    vrnn_model,
                                    "action_entropy_bonus",
                                    0.0,
                                )
                            ),
                            "model_condition": getattr(
                                vrnn_model,
                                "model_condition",
                                "planning",
                            ),
                            "action_loss_enabled": bool(
                                getattr(
                                    vrnn_model,
                                    "action_loss_enabled",
                                    True,
                                )
                            ),
                            "reward_normalization_constant": float(
                                reward_normalization_constant
                            ),
                            "reward_normalization_scheme": (
                                reward_normalization_scheme
                            ),
                            **node_output_metadata,
                            **trial_output_metadata,
                            **constant_metadata,
                        }
                    )

                completed_trials = trial_index + 1
                if (
                    timing_enabled
                    and (
                        completed_trials % 500 == 0
                        or completed_trials == num_trials
                    )
                ):
                    row_elapsed = time.perf_counter() - row_assembly_start
                    progress_log(
                        f"assembled {completed_trials}/{num_trials} trials "
                        f"for {cell_label} in {row_elapsed:.2f}s "
                        f"({len(simulation_rows)} node rows)"
                    )

            stage_start = time.perf_counter()
            simulation_frame = pd.DataFrame(simulation_rows)
            progress_log(
                f"simulation DataFrame built for {cell_label} in "
                f"{time.perf_counter() - stage_start:.2f}s; "
                f"rows={len(simulation_frame)}"
            )

            output_file = config.simulation_dir / (model_name + ".csv")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            temporary_output_file = job_temporary_path(
                output_file,
                label="simulation",
                suffix=".csv.in_progress",
            )
            temporary_output_file.unlink(missing_ok=True)
            stage_start = time.perf_counter()
            simulation_frame.to_csv(temporary_output_file, index=False)
            temporary_output_file.replace(output_file)
            progress_log(
                f"simulation CSV written for {cell_label} in "
                f"{time.perf_counter() - stage_start:.2f}s; "
                f"cell_total={time.perf_counter() - cell_start:.2f}s; "
                f"output={output_file}"
            )

    progress_log(
        "all requested simulations finished in "
        f"{time.perf_counter() - simulation_start:.2f}s"
    )
