"""Utilities for tree analysis, reward encoding, and batch generation."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import tensorflow as tf
from scipy.stats import truncnorm


DecisionTree = dict[str, dict[str, list[Any]]]


def analyze_tree_paths(
    decision_tree: DecisionTree,
) -> tuple[
    list[str],
    dict[int, int],
    dict[str, str],
    dict[str, list[int]],
    dict[str, str],
    list[int],
    list[list[int]],
    dict[str, int],
    dict[int, list[int]],
]:
    """Enumerate root-to-leaf paths and associated node/path lookup tables.

    Args:
        decision_tree: Nested mapping from node IDs to available actions, where
            each action stores the next node as its second entry.

    Returns:
        A tuple containing path names, leaf-node mappings, sibling mappings,
        node-to-path mappings, node path names, path leaf indices, path node
        indices, estimated-best-path mappings, and path-to-node mappings.
    """

    def find_siblings_for_node(parent_node: str, child_node: str) -> None:
        child_nodes = [
            transition[1] for transition in decision_tree[parent_node].values()
        ]
        for child_position, current_child in enumerate(child_nodes):
            if current_child == child_node:
                if child_position + 1 < len(child_nodes):
                    sibling_map[child_node] = child_nodes[child_position + 1]
                elif len(child_nodes) > 1:
                    sibling_map[child_node] = child_nodes[0]

    def depth_first_search(node_id: str, current_path: list[str]) -> None:
        path_index = len(path_names)
        for path_node in current_path:
            if path_node not in node_path_map:
                node_path_map[path_node] = []
            if path_index not in node_path_map[path_node]:
                node_path_map[path_node].append(path_index)

        if not decision_tree[node_id]:
            path_name = ", ".join(current_path)
            path_names.append(path_name)
            path_leaf_dict[len(path_names) - 1] = int(node_id)

            for path_node in current_path:
                node_path_name[path_node] = path_name

            path_indices.append(int(node_id))
            current_node_indices = [
                int(path_node)
                for path_node in current_path
                if path_node != "0"
            ]
            node_indices.append(current_node_indices)

            current_path_index = len(path_names) - 1
            est_best_path_map[path_name] = current_path_index
            path_node_map[current_path_index] = current_node_indices
            return

        for _, next_transition in decision_tree[node_id].items():
            next_node = next_transition[1]
            find_siblings_for_node(node_id, next_node)
            depth_first_search(next_node, current_path + [next_node])

    path_names: list[str] = []
    path_leaf_dict: dict[int, int] = {}
    sibling_map: dict[str, str] = {}
    node_path_map: dict[str, list[int]] = {}
    node_path_name: dict[str, str] = {}
    path_indices: list[int] = []
    node_indices: list[list[int]] = []
    est_best_path_map: dict[str, int] = {}
    path_node_map: dict[int, list[int]] = {}

    depth_first_search("0", ["0"])
    return (
        path_names,
        path_leaf_dict,
        sibling_map,
        node_path_map,
        node_path_name,
        path_indices,
        node_indices,
        est_best_path_map,
        path_node_map,
    )


def pad_vector(
    vector: list[Any],
    target_length: int,
    placeholder: Any,
) -> list[Any]:
    """Pad a list to a target length using a placeholder value.

    Args:
        vector: Input list to pad.
        target_length: Desired output length.
        placeholder: Value used for padding.

    Returns:
        The padded list, or the original list when no padding is required.
    """
    padding_length = target_length - len(vector)

    if padding_length > 0:
        padded_vector = [placeholder] * target_length
        padded_vector[: len(vector)] = vector
    else:
        padded_vector = vector

    return padded_vector


def calculate_posterior_mean_variance(
    prior_std: tf.Tensor,
    likelihood_mean: tf.Tensor,
    likelihood_std: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Compute posterior mean and variance terms for Gaussian updates.

    Args:
        prior_std: Prior standard deviation tensor.
        likelihood_mean: Likelihood mean tensor.
        likelihood_std: Likelihood standard deviation tensor.

    Returns:
        Posterior mean and posterior mean-variance tensors.
    """
    prior_precision = 1 / tf.square(prior_std)
    likelihood_precision = 1 / tf.square(likelihood_std)

    posterior_variance = 1 / (prior_precision + likelihood_precision)
    posterior_mean = posterior_variance * (
        likelihood_mean * likelihood_precision
    )
    posterior_mean_variance = tf.square(
        posterior_variance * likelihood_precision
    )

    return posterior_mean, posterior_mean_variance


def get_truncated_normal_samples(
    size: int = 6,
    mean: float = 0,
    sd: float = 1,
    low: float = 0,
    upp: float = 10,
) -> np.ndarray:
    """Draw samples from a truncated normal distribution.

    Args:
        size: Number of samples to draw.
        mean: Distribution mean before truncation.
        sd: Distribution standard deviation before truncation.
        low: Lower truncation bound.
        upp: Upper truncation bound.

    Returns:
        Array of sampled values.
    """
    distribution = truncnorm(
        (low - mean) / sd,
        (upp - mean) / sd,
        loc=mean,
        scale=sd,
    )
    return distribution.rvs(size)


def scalar_to_categorical(
    scalar_values: tf.Tensor,
    num_classes: int = 9,
) -> tf.Tensor:
    """Convert scalar rewards into reversed one-hot category encodings.

    Args:
        scalar_values: Tensor of scalar rewards.
        num_classes: Number of reward categories.

    Returns:
        One-hot encoded tensor with the same leading dimensions as the input.
    """
    shifted_values = 4.0 - scalar_values
    category_indices = tf.floor(shifted_values + 0.5)
    category_indices = tf.clip_by_value(category_indices, 0, num_classes - 1)
    category_indices = tf.cast(category_indices, tf.int32)
    return tf.one_hot(category_indices, num_classes, dtype=tf.float32)


def categorical_to_scalar(category_probs: tf.Tensor) -> tf.Tensor:
    """Convert category probabilities to scalar rewards using hard argmax.

    Args:
        category_probs: Tensor whose last dimension contains category
            probabilities in descending reward order.

    Returns:
        Tensor of scalar reward predictions with a final singleton dimension.
    """
    category_values = tf.constant(
        [4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0],
        dtype=tf.float32,
    )
    predicted_indices = tf.argmax(category_probs, axis=-1)
    predicted_scalars = tf.gather(category_values, predicted_indices)
    return tf.expand_dims(predicted_scalars, axis=-1)


def random_argmax(vector: tf.Tensor) -> tf.Tensor:
    """Return argmax indices along the first axis.

    Args:
        vector: Tensor to evaluate.

    Returns:
        Tensor of argmax indices.
    """
    return tf.argmax(vector, axis=0)


def generate_batch_data(
    batch_size: int,
    time_steps: int,
) -> tf.Tensor:
    """Generate a random reward batch for model training.

    Args:
        batch_size: Number of trials in the batch.
        time_steps: Number of reward nodes per trial.

    Returns:
        Tensor with shape ``[batch_size, time_steps, 1]``.
    """
    feature_dim = 1

    if time_steps == 2:
        reward_values = np.array(
            [
                [random.choice([0, 1]) for _ in range(time_steps)]
                for _ in range(batch_size)
            ]
        )
    else:
        reward_values = np.array(
            [
                [
                    random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                    for _ in range(time_steps)
                ]
                for _ in range(batch_size)
            ]
        )

    input_data = tf.constant(reward_values, dtype=tf.float32)
    return tf.reshape(input_data, [batch_size, time_steps, feature_dim])


def random_argmax_per_row(tensor: tf.Tensor) -> tf.Tensor:
    """Return an equal-weight mask over maxima along the first tensor axis.

    Args:
        tensor: Input tensor.

    Returns:
        Tensor with nonzero entries only where values equal the column maxima.
    """
    max_values = tf.reduce_max(tensor, axis=0)
    max_mask = tf.equal(tensor, max_values)
    max_mask_float = tf.cast(max_mask, tf.float32)
    max_count = tf.reduce_sum(max_mask_float, axis=0)
    return max_mask_float / max_count


def policy(estimated_path_rewards: tf.Tensor) -> tf.Tensor:
    """Normalize estimated path rewards into a probability distribution.

    Args:
        estimated_path_rewards: Tensor with shape ``[batch_size, num_paths]``.

    Returns:
        Normalized policy tensor with shape ``[batch_size, num_paths]``.
    """
    min_rewards = tf.reduce_min(estimated_path_rewards, axis=1, keepdims=True)
    max_rewards = tf.reduce_max(estimated_path_rewards, axis=1, keepdims=True)

    normalized_rewards = (estimated_path_rewards - min_rewards) / (
        max_rewards - min_rewards + 1e-8
    )
    return normalized_rewards / tf.reduce_sum(
        normalized_rewards,
        axis=1,
        keepdims=True,
    )
