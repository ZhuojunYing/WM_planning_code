"""Simulation and CSV export utilities for trained model inference."""

from __future__ import annotations

import os
import random
from types import ModuleType

import numpy as np
import pandas as pd
import tensorflow as tf

from model import VariationalRNN, build_decoder, build_encoder


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


def run_simulation(config: ModuleType) -> None:
    """Run trained models over simulated trials and export node-level results.

    Args:
        config: Imported configuration module containing model, tree, and path
            settings.

    Returns:
        None.
    """
    num_trials = int(config.num_trials)
    time_steps = config.time_steps

    if config.time_steps != 2:
        rewards_list = [
            [
                random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                for _ in range(time_steps)
            ]
            for _ in range(num_trials)
        ]
    else:
        rewards_list = [
            [random.choice([0, 1]) for _ in range(time_steps)]
            for _ in range(num_trials)
        ]

    for lambda_ in config.lambda_values:
        for alpha in config.alpha_values:
            if config.time_steps == 30:
                model_name = (
                    f"lambda_{lambda_}_alpha_{alpha}_"
                    f"seed_{config.seed}_"
                    f"{config.tree_type}"
                )
            else:
                model_name = (
                    f"lambda_{lambda_}_alpha_{alpha}_"
                    f"seed_{config.seed}"
                )

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
            )

            weights_file_path = os.path.join(
                config.model_dir_name,
                model_name + '.weights.h5',
            )

            if os.path.exists(weights_file_path):
                dummy_input = tf.zeros(
                    (1, config.time_steps, 1),
                    dtype=tf.float32,
                )
                _ = vrnn_model(dummy_input, training=False)

                try:
                    vrnn_model.load_weights(weights_file_path)
                except Exception:
                    continue
            else:
                continue

            feature_dim = 1
            values = np.array(rewards_list)
            input_data = tf.constant(values, dtype=tf.float32)
            input_data = tf.reshape(
                input_data,
                [num_trials, time_steps, feature_dim],
            )

            full_batch_outputs = vrnn_model(input_data, training=False)
            mutual_information_cost = float(full_batch_outputs[10])

            simulation_rows = []

            for trial_index, rewards in enumerate(rewards_list[:num_trials]):
                rewards_tensor = tf.constant(rewards, dtype=tf.float32)
                rewards_tensor = tf.reshape(rewards_tensor, [1, -1, 1])

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
                    [4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0],
                    dtype=tf.float32,
                )
                scalar_reconstructions = tf.gather(
                    category_values_tf,
                    category_indices,
                )

                path_rewards = calculate_path_rewards_sim(
                    config.index_path_map,
                    rewards,
                )
                max_path_reward = float(np.max(path_rewards))
                min_path_reward = float(np.min(path_rewards))
                action_outputs = outputs[1]
                action_policy = np.asarray(
                    action_outputs,
                    dtype=float,
                ).reshape(-1)
                action_policy = action_policy / np.sum(action_policy)
                path_rewards_arr = calculate_path_rewards_sim(
                    config.index_path_map,
                    rewards,
                )
                best_idx = int(
                    np.random.choice(config.num_paths, p=action_policy)
                )
                V = float(path_rewards_arr[best_idx])
                for node_number in range(1, config.time_steps + 1):
                    node_idx = node_number - 1

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
                            "max_path_reward": max_path_reward,
                            "min_path_reward": min_path_reward,
                            "path_reward": float(np.mean(node_path_rewards)),
                            "V": V,
                            "MI_cost": mutual_information_cost,
                            "is_leaf": bool(is_leaf),
                        }
                    )

            simulation_frame = pd.DataFrame(simulation_rows)

            output_file = os.path.join(
                config.sim_dir_name,
                model_name + ".csv",
            )
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            simulation_frame.to_csv(output_file, index=False)
