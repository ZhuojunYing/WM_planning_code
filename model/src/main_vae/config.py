"""
config.py
Handles system arguments, CPU-only setup, hyperparameter definition,
and builds the decision tree matrices used across the project.
"""

import os
import sys
import random
import math
from pathlib import Path

import numpy as np
import tensorflow as tf

# Import your existing helper script
from . import helper

# Add the repository root before loading the repository-wide path
# configuration when this package is invoked from an external working path.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project_paths import (  # noqa: E402
    CHECKPOINT_DIR,
    SIMULATION_DIR,
    TRAINING_LOG_DIR,
    WEIGHT_DIR,
    resolve_directory_override,
)

# ---------------------------------------------------------
# 1. COMMAND LINE ARGUMENTS
# ---------------------------------------------------------
try:
    lambda_string = sys.argv[1]
    lambda_values = [float(x) for x in lambda_string.split(',')]

    alpha_string = sys.argv[2]
    alpha_values = [float(x) for x in alpha_string.split(',')]

    model_dir_arg = sys.argv[3]
    simulation_dir_arg = sys.argv[4]
    weight_dir = resolve_directory_override(model_dir_arg, WEIGHT_DIR)
    checkpoint_dir = CHECKPOINT_DIR
    training_log_dir = TRAINING_LOG_DIR
    simulation_dir = resolve_directory_override(
        simulation_dir_arg,
        SIMULATION_DIR,
    )
    epochs_string = sys.argv[5]
    epochs = int(epochs_string)
    epochs_count = 0

    seed = int(sys.argv[6])

    tree_size = int(sys.argv[7])
    # Renamed from 'train' to avoid variable shadowing
    train_mode = sys.argv[8]
    tree_type = str(sys.argv[9])
    optional_arguments = list(sys.argv[10:])
    if optional_arguments and not optional_arguments[0].startswith("--"):
        num_trials = int(optional_arguments.pop(0))
    else:
        num_trials = 2000

    max_action_gradient_scale = 1.0
    reconstruction_head_type = "nonlinear_location_scale"
    reconstruction_hidden_units = 128
    reconstruction_loss_scope = "final_timestep"
    reconstruction_architecture_tagged = False
    channel_kl_aggregation = "sum"
    channel_kl_time_weighting = "remaining_steps"
    main_vae_rnn_units = 64
    independent_channel_rnn_units = 64
    independent_channel_latent_dim = 16
    critic_loss_enabled = True
    final_learning_rate = 3e-4
    late_lr_plateau_start_epoch = None
    late_lr_plateau_end_epoch = None
    parameter_multiplier = None
    parameter_multiplier_explicit = False
    initial_critic_coefficient = None
    initial_critic_coefficient_explicit = False
    action_entropy_bonus = 0.0
    model_condition = "planning"
    while optional_arguments:
        option = optional_arguments.pop(0)
        if option == "--max-action-gradient-scale":
            if not optional_arguments:
                raise ValueError(
                    "--max-action-gradient-scale requires a value."
                )
            max_action_gradient_scale = float(optional_arguments.pop(0))
        elif option.startswith("--max-action-gradient-scale="):
            max_action_gradient_scale = float(option.split("=", 1)[1])
        elif option == "--action-entropy-bonus":
            if not optional_arguments:
                raise ValueError(
                    "--action-entropy-bonus requires a value."
                )
            action_entropy_bonus = float(optional_arguments.pop(0))
        elif option.startswith("--action-entropy-bonus="):
            action_entropy_bonus = float(option.split("=", 1)[1])
        elif option == "--condition":
            if not optional_arguments:
                raise ValueError(
                    "--condition requires planning or exposure."
                )
            model_condition = optional_arguments.pop(0)
        elif option.startswith("--condition="):
            model_condition = option.split("=", 1)[1]
        elif option == "--nonlinear-scale":
            reconstruction_head_type = "nonlinear_location_scale"
        elif option == "--nonlinear-categorical":
            reconstruction_head_type = "nonlinear_categorical"
        elif option == "--reconstruction-hidden-units":
            if not optional_arguments:
                raise ValueError(
                    "--reconstruction-hidden-units requires a value."
                )
            reconstruction_hidden_units = int(optional_arguments.pop(0))
            reconstruction_architecture_tagged = True
        elif option.startswith("--reconstruction-hidden-units="):
            reconstruction_hidden_units = int(option.split("=", 1)[1])
            reconstruction_architecture_tagged = True
        elif option == "--reconstruction-loss-scope":
            if not optional_arguments:
                raise ValueError(
                    "--reconstruction-loss-scope requires a value."
                )
            reconstruction_loss_scope = optional_arguments.pop(0)
            reconstruction_architecture_tagged = True
        elif option.startswith("--reconstruction-loss-scope="):
            reconstruction_loss_scope = option.split("=", 1)[1]
            reconstruction_architecture_tagged = True
        elif option == "--channel-kl-aggregation":
            if not optional_arguments:
                raise ValueError(
                    "--channel-kl-aggregation requires sum or mean."
                )
            channel_kl_aggregation = optional_arguments.pop(0)
        elif option.startswith("--channel-kl-aggregation="):
            channel_kl_aggregation = option.split("=", 1)[1]
        elif option == "--sum-channel-kl":
            channel_kl_aggregation = "sum"
        elif option == "--mean-channel-kl":
            channel_kl_aggregation = "mean"
        elif option == "--channel-kl-time-weighting":
            if not optional_arguments:
                raise ValueError(
                    "--channel-kl-time-weighting requires remaining_steps "
                    "or uniform."
                )
            channel_kl_time_weighting = optional_arguments.pop(0)
        elif option.startswith("--channel-kl-time-weighting="):
            channel_kl_time_weighting = option.split("=", 1)[1]
        elif option == "--rnn-units":
            if not optional_arguments:
                raise ValueError("--rnn-units requires a value.")
            main_vae_rnn_units = int(optional_arguments.pop(0))
            independent_channel_rnn_units = main_vae_rnn_units
        elif option.startswith("--rnn-units="):
            main_vae_rnn_units = int(option.split("=", 1)[1])
            independent_channel_rnn_units = main_vae_rnn_units
        elif option == "--latent-dim":
            if not optional_arguments:
                raise ValueError("--latent-dim requires a value.")
            independent_channel_latent_dim = int(
                optional_arguments.pop(0)
            )
        elif option.startswith("--latent-dim="):
            independent_channel_latent_dim = int(option.split("=", 1)[1])
        elif option == "--final-learning-rate":
            if not optional_arguments:
                raise ValueError(
                    "--final-learning-rate requires a value."
                )
            final_learning_rate = float(optional_arguments.pop(0))
        elif option.startswith("--final-learning-rate="):
            final_learning_rate = float(option.split("=", 1)[1])
        elif option == "--late-lr-plateau-start-epoch":
            if not optional_arguments:
                raise ValueError(
                    "--late-lr-plateau-start-epoch requires a value."
                )
            late_lr_plateau_start_epoch = int(optional_arguments.pop(0))
        elif option.startswith("--late-lr-plateau-start-epoch="):
            late_lr_plateau_start_epoch = int(option.split("=", 1)[1])
        elif option == "--late-lr-plateau-end-epoch":
            if not optional_arguments:
                raise ValueError(
                    "--late-lr-plateau-end-epoch requires a value."
                )
            late_lr_plateau_end_epoch = int(optional_arguments.pop(0))
        elif option.startswith("--late-lr-plateau-end-epoch="):
            late_lr_plateau_end_epoch = int(option.split("=", 1)[1])
        elif option == "--no-critic-loss":
            critic_loss_enabled = False
        elif option == "--parameter-multiplier":
            if not optional_arguments:
                raise ValueError(
                    "--parameter-multiplier requires a value."
                )
            parameter_multiplier = float(optional_arguments.pop(0))
            parameter_multiplier_explicit = True
        elif option.startswith("--parameter-multiplier="):
            parameter_multiplier = float(option.split("=", 1)[1])
            parameter_multiplier_explicit = True
        elif option == "--initial-critic-coefficient":
            if not optional_arguments:
                raise ValueError(
                    "--initial-critic-coefficient requires a value."
                )
            initial_critic_coefficient = float(
                optional_arguments.pop(0)
            )
            initial_critic_coefficient_explicit = True
        elif option.startswith("--initial-critic-coefficient="):
            initial_critic_coefficient = float(option.split("=", 1)[1])
            initial_critic_coefficient_explicit = True
        else:
            raise ValueError(f"Unknown command-line option: {option}")

    # The current 13-node specification uses unit-scaled objectives, a unit
    # initial critic coefficient, and no actor entropy bonus. Materialize the
    # two loss-scale defaults so their artifacts cannot collide with the
    # historical 13-node defaults (multiplier 10 and critic coefficient 0.1).
    if model_condition not in {"planning", "exposure"}:
        raise ValueError("--condition must be planning or exposure.")

    action_loss_enabled = model_condition == "planning"
    if model_condition == "exposure":
        if action_entropy_bonus != 0.0:
            raise ValueError(
                "--action-entropy-bonus must be 0 for exposure models."
            )
        if initial_critic_coefficient_explicit:
            raise ValueError(
                "--initial-critic-coefficient is not applicable to exposure "
                "models."
            )
        critic_loss_enabled = False

    if tree_size == 13:
        if parameter_multiplier is None:
            parameter_multiplier = 1.0
            parameter_multiplier_explicit = True
        if (
            model_condition == "planning"
            and initial_critic_coefficient is None
        ):
            initial_critic_coefficient = 1.0
            initial_critic_coefficient_explicit = True

    if (
        not math.isfinite(max_action_gradient_scale)
        or max_action_gradient_scale < 1.0
    ):
        raise ValueError(
            "--max-action-gradient-scale must be finite and at least 1."
        )
    if (
        not math.isfinite(action_entropy_bonus)
        or action_entropy_bonus < 0.0
    ):
        raise ValueError(
            "--action-entropy-bonus must be finite and non-negative."
        )
    if reconstruction_hidden_units < 1:
        raise ValueError(
            "--reconstruction-hidden-units must be a positive integer."
        )
    if main_vae_rnn_units < 1:
        raise ValueError("--rnn-units must be a positive integer.")
    if independent_channel_rnn_units < 1:
        raise ValueError("--rnn-units must be a positive integer.")
    if independent_channel_latent_dim < 1:
        raise ValueError("--latent-dim must be a positive integer.")
    if parameter_multiplier is not None and (
        not math.isfinite(parameter_multiplier)
        or parameter_multiplier <= 0.0
    ):
        raise ValueError(
            "--parameter-multiplier must be finite and positive."
        )
    if initial_critic_coefficient is not None and (
        not math.isfinite(initial_critic_coefficient)
        or initial_critic_coefficient < 0.0
    ):
        raise ValueError(
            "--initial-critic-coefficient must be finite and non-negative."
        )
    if reconstruction_loss_scope not in {
        "all_timesteps",
        "final_timestep",
    }:
        raise ValueError(
            "--reconstruction-loss-scope must be all_timesteps or "
            "final_timestep."
        )
    if channel_kl_aggregation not in {"sum", "mean"}:
        raise ValueError(
            "--channel-kl-aggregation must be sum or mean."
        )
    if channel_kl_time_weighting not in {"remaining_steps", "uniform"}:
        raise ValueError(
            "--channel-kl-time-weighting must be remaining_steps or uniform."
        )
    if (
        not math.isfinite(final_learning_rate)
        or final_learning_rate <= 0.0
        or final_learning_rate > 3e-4
    ):
        raise ValueError(
            "--final-learning-rate must be finite, positive, and no greater "
            "than the peak learning rate (3e-4)."
        )
    if (late_lr_plateau_start_epoch is None) != (
        late_lr_plateau_end_epoch is None
    ):
        raise ValueError(
            "Both late learning-rate plateau epochs must be supplied."
        )
    if late_lr_plateau_start_epoch is not None and not (
        0
        < late_lr_plateau_start_epoch
        < late_lr_plateau_end_epoch
        < epochs
    ):
        raise ValueError(
            "Late learning-rate plateau epochs must satisfy "
            "0 < start < end < epochs."
        )
except (IndexError, ValueError) as error:
    print(f"Error: {error}")
    print(
        "Usage: python -m model.src.main_vae.main <lambda> <alpha> "
        "<weight_dir|default> <simulation_dir|default> "
        "<epochs> <seed> <tree_size: 3|7|13|31|40> "
        "<train/simulate> <tree_type> [num_trials] "
        "[--condition planning|exposure] "
        "[--max-action-gradient-scale VALUE] "
        "[--action-entropy-bonus VALUE] "
        "[--nonlinear-scale|--nonlinear-categorical] "
        "[--reconstruction-hidden-units VALUE] "
        "[--reconstruction-loss-scope all_timesteps|final_timestep] "
        "[--channel-kl-aggregation sum|mean] "
        "[--channel-kl-time-weighting remaining_steps|uniform] "
        "[--rnn-units VALUE] "
        "[--latent-dim VALUE] "
        "[--final-learning-rate VALUE] "
        "[--late-lr-plateau-start-epoch VALUE] "
        "[--late-lr-plateau-end-epoch VALUE] "
        "[--parameter-multiplier VALUE] "
        "[--initial-critic-coefficient VALUE] "
        "[--no-critic-loss]"
    )
    sys.exit(1)


def gradient_scale_artifact_suffix() -> str:
    """Tag scales that differ from the historical untagged scale-10 run."""
    if max_action_gradient_scale == 10.0:
        return ""
    formatted_scale = format(max_action_gradient_scale, ".15g")
    return f"_gradient_scale_{formatted_scale}"


def reconstruction_head_artifact_suffix() -> str:
    """Return the active reconstruction architecture's artifact suffix."""
    return f"_head_{reconstruction_head_type}"


def reconstruction_range_artifact_suffix() -> str:
    """Identify weights using the current inclusive reconstruction range."""
    return "_recon_range_minus5_plus5"


def reconstruction_architecture_artifact_suffix() -> str:
    """Return a suffix for an explicitly selected architecture variant."""
    if (
        not reconstruction_architecture_tagged
        and reconstruction_hidden_units == 64
        and reconstruction_loss_scope == "all_timesteps"
    ):
        return ""
    return (
        f"_recon_units_{reconstruction_hidden_units}"
        f"_recon_scope_{reconstruction_loss_scope}"
    )


def rnn_architecture_artifact_suffix() -> str:
    """Identify a nondefault main recurrent hidden-state width."""
    if rnn_units == 64:
        return ""
    return f"_rnn_units_{rnn_units}"


def epoch_artifact_suffix() -> str:
    """Return the requested training duration's artifact suffix."""
    return f"_epochs_{epochs}"


def critic_artifact_suffix() -> str:
    """Return a suffix when the auxiliary critic loss is disabled."""
    return "" if critic_loss_enabled else "_critic_off"


def final_learning_rate_artifact_suffix() -> str:
    """Return a suffix for a nondefault terminal learning rate."""
    if final_learning_rate == 3e-6:
        return ""
    return f"_final_lr_{format(final_learning_rate, '.15g')}"


def learning_rate_schedule_artifact_suffix() -> str:
    """Return a suffix for the optional late learning-rate plateau."""
    if late_lr_plateau_start_epoch is None:
        return ""
    return (
        f"_lr_plateau_{late_lr_plateau_start_epoch}"
        f"_{late_lr_plateau_end_epoch}"
    )


def tree_artifact_suffix() -> str:
    """Identify the tree topology in every original-model artifact name."""
    suffix = f"_tree_{tree_size}"
    if tree_size == 31:
        suffix += f"_{tree_type}"
    return suffix


def condition_artifact_suffix() -> str:
    """Keep planning names stable and separate exposure-only weights."""
    return "" if model_condition == "planning" else "_exposure"


def parameter_multiplier_artifact_suffix() -> str:
    """Identify an explicitly selected common loss-weight multiplier."""
    if not parameter_multiplier_explicit:
        return ""
    formatted_multiplier = format(parameter_multiplier, ".15g")
    return f"_pm_{formatted_multiplier}"


def initial_critic_coefficient_artifact_suffix() -> str:
    """Identify an explicitly selected initial critic coefficient."""
    if not initial_critic_coefficient_explicit:
        return ""
    formatted_coefficient = format(initial_critic_coefficient, ".15g")
    return f"_critic_coef_{formatted_coefficient}"


def action_entropy_artifact_suffix() -> str:
    """Identify nonzero entropy runs and the zero-entropy 13-node spec."""
    if tree_size != 13 and action_entropy_bonus == 0.0:
        return ""
    formatted_bonus = format(action_entropy_bonus, ".15g")
    return f"_eb_{formatted_bonus}"


def reward_normalization_artifact_suffix() -> str:
    """Identify 31-node weights using the shared reward normalizer."""
    if tree_size != 31:
        return ""
    # Keep this marker short: the fully tagged 31-node weight names are close
    # to the 255-byte per-component filesystem limit on Torch.
    return "_rns"


def model_artifact_name(
    lambda_value: float,
    alpha_value: float,
    model_seed: int | None = None,
) -> str:
    """Return the shared artifact stem for one trained model."""
    selected_seed = seed if model_seed is None else model_seed
    base_name = (
        f"lambda_{lambda_value}_alpha_{alpha_value}_seed_{selected_seed}"
    )
    return (
        base_name
        + tree_artifact_suffix()
        + condition_artifact_suffix()
        + reconstruction_head_artifact_suffix()
        + reconstruction_range_artifact_suffix()
        + reconstruction_architecture_artifact_suffix()
        + rnn_architecture_artifact_suffix()
        + epoch_artifact_suffix()
        + critic_artifact_suffix()
        + gradient_scale_artifact_suffix()
        + final_learning_rate_artifact_suffix()
        + learning_rate_schedule_artifact_suffix()
        + parameter_multiplier_artifact_suffix()
        + initial_critic_coefficient_artifact_suffix()
        + action_entropy_artifact_suffix()
        + reward_normalization_artifact_suffix()
    )


def model_artifact_name_candidates(
    lambda_value: float,
    alpha_value: float,
    model_seed: int | None = None,
) -> tuple[str, ...]:
    """Return current and recognized legacy stems for loading artifacts."""
    current_name = model_artifact_name(lambda_value, alpha_value, model_seed)
    candidates = [current_name]
    if parameter_multiplier_explicit:
        candidates.append(
            current_name.replace("_pm_", "_parameter_multiplier_")
        )
    if initial_critic_coefficient_explicit:
        candidates.extend(
            candidate.replace(
                "_critic_coef_",
                "_initial_critic_coefficient_",
            )
            for candidate in tuple(candidates)
        )
    return tuple(
        candidate
        for index, candidate in enumerate(candidates)
        if candidate not in candidates[:index]
        and len((candidate + ".weights.h5").encode("utf-8")) <= 255
    )


# ---------------------------------------------------------
# 2. SEED & REPRODUCIBILITY
# ---------------------------------------------------------
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# ---------------------------------------------------------
# 3. CPU-ONLY CONFIGURATION
# ---------------------------------------------------------
# Torch jobs for this project are CPU-only. Keep TensorFlow from selecting a
# GPU even if this source is run in an environment where one is visible.
tf.config.set_visible_devices([], "GPU")
method = "exponential"
# ---------------------------------------------------------
# 4. MODEL HYPERPARAMETERS
# ---------------------------------------------------------
tree_steps = tree_size
reward_nodes_by_tree_steps = {
    3: 2,
    7: 6,
    13: 12,
    31: 30,
    40: 39,
}
if tree_steps not in reward_nodes_by_tree_steps:
    raise ValueError(
        "tree_size should be the root-inclusive number of tree nodes: "
        "3, 7, 13, 31, or 40."
    )

latent_dim = 16
reward_output_dim = reward_nodes_by_tree_steps[tree_steps]
rnn_units = main_vae_rnn_units
output_dim = rnn_units
time_steps = reward_output_dim
input_dim = 1
reconstruction_min_reward = helper.RECONSTRUCTION_MIN_REWARD
reconstruction_max_reward = helper.RECONSTRUCTION_MAX_REWARD
reconstruction_values = helper.RECONSTRUCTION_VALUES
num_categories = helper.NUM_RECONSTRUCTION_CATEGORIES

trials_per_epoch = 200
batch_size = 200
kl_scaler = 5

# Expected maximum path-reward normalizers used by the actor loss. The
# standard trees retain their established constants. All four 31-node
# topologies deliberately use the same historical deep-tree constant so the
# actor objective has one common reward divisor in the wide/deep comparison.
reward_normalization_by_time_steps = {
    2: 0.75,
    6: 3.58,
    12: 5.11,
    39: 8.0682,
}
if time_steps == 30:
    configured_31_tree_types = {
        "deep_breadth",
        "deep_depth",
        "wide_breadth",
        "wide_depth",
    }
    if tree_type not in configured_31_tree_types:
        raise ValueError(
            "A 31-node model needs one of the configured tree types: "
            "deep_breadth, deep_depth, wide_breadth, or wide_depth."
        )
    reward_normalization_constant = 8.157443089523166
    reward_normalization_scheme = "shared_31_expected_max"
else:
    reward_normalization_constant = (
        reward_normalization_by_time_steps[time_steps]
    )
    reward_normalization_scheme = "legacy_time_step_expected_max"

# ---------------------------------------------------------
# 5. DECISION TREE SETUP
# ---------------------------------------------------------
if time_steps == 2:
    decision_tree = {
        '0': {'right': [-1, '1'], 'up': [-1, '2']},
        '1': {},
        '2': {}
    }


elif time_steps == 6:
    decision_tree = {
        '0': {'right': [-1, '1'], 'up': [-1, '4']},
        '1': {'right': [-1, '2'], 'up': [-1, '3']},
        '2': {},
        '3': {},
        '4': {'right': [-1, '5'], 'up': [-1, '6']},
        '5': {},
        '6': {}
    }

elif time_steps == 12:
    decision_tree = {
        '0': {'right': [-1, '1'], 'up': [-1, '5'], 'left': [-1, "9"]},
        '1': {'right': [-1, '2'], 'up': [-1, '3'], 'left': [-1, "4"]},
        '2': {},
        '3': {},
        '4': {},
        '5': {'right': [-1, '6'], 'up': [-1, '7'], 'left': [-1, "8"]},
        '6': {},
        '7': {},
        '8': {},
        '9': {'right': [-1, '10'], 'up': [-1, '11'], 'left': [-1, "12"]},
        '10': {},
        '11': {},
        '12': {}
    }
elif time_steps == 39:
    decision_tree = {
        '0': {'right': [-1, '1'], 'up': [-1, '14'], 'left': [-1, '27']},
        '1': {'right': [-1, '2'], 'up': [-1, '6'], 'left': [-1, '10']},
        '2': {'right': [-1, '3'], 'up': [-1, '4'], 'left': [-1, '5']},
        '3': {},
        '4': {},
        '5': {},
        '6': {'right': [-1, '7'], 'up': [-1, '8'], 'left': [-1, '9']},
        '7': {},
        '8': {},
        '9': {},
        '10': {'right': [-1, '11'], 'up': [-1, '12'], 'left': [-1, '13']},
        '11': {},
        '12': {},
        '13': {},
        '14': {'right': [-1, '15'], 'up': [-1, '19'], 'left': [-1, '23']},
        '15': {'right': [-1, '16'], 'up': [-1, '17'], 'left': [-1, '18']},
        '16': {},
        '17': {},
        '18': {},
        '19': {'right': [-1, '20'], 'up': [-1, '21'], 'left': [-1, '22']},
        '20': {},
        '21': {},
        '22': {},
        '23': {'right': [-1, '24'], 'up': [-1, '25'], 'left': [-1, '26']},
        '24': {},
        '25': {},
        '26': {},
        '27': {'right': [-1, '28'], 'up': [-1, '32'], 'left': [-1, '36']},
        '28': {'right': [-1, '29'], 'up': [-1, '30'], 'left': [-1, '31']},
        '29': {},
        '30': {},
        '31': {},
        '32': {'right': [-1, '33'], 'up': [-1, '34'], 'left': [-1, '35']},
        '33': {},
        '34': {},
        '35': {},
        '36': {'right': [-1, '37'], 'up': [-1, '38'], 'left': [-1, '39']},
        '37': {},
        '38': {},
        '39': {}
    }
else:
    if (tree_type == "deep_breadth"):

        tree = "deep"
        # Your decision tree dictionary
        decision_tree = {
            '0': {'right': [-1, '1'], 'left': [3, '2']},
            '1': {'up': [-1, '3'], 'down': [2, '4']},
            '2': {'up': [-1, '5'], 'down': [2, '6']},
            '3': {'up': [-1, '7'], 'down': [2, '8']},
            '4': {'up': [-1, '9'], 'down': [2, '10']},
            '5': {'up': [-1, '11'], 'down': [2, '12']},
            '6': {'up': [-1, '13'], 'down': [2, '14']},
            '7': {'up': [-1, '15'], 'down': [2, '16']},
            '8': {'up': [-1, '17'], 'down': [2, '18']},
            '9': {'up': [-1, '19'], 'down': [2, '20']},
            '10': {'right': [-1, '21'], 'left': [3, '22']},
            '11': {'up': [-1, '23'], 'down': [2, '24']},
            '12': {'up': [-1, '25'], 'down': [2, '26']},
            '13': {'up': [-1, '27'], 'down': [2, '28']},
            '14': {'up': [-1, '29'], 'down': [2, '30']},
            '15': {},
            '16': {},
            '17': {},
            '18': {},
            '19': {},
            '20': {},
            '21': {},
            '22': {},
            '23': {},
            '24': {},
            '25': {},
            '26': {},
            '27': {},
            '28': {},
            '29': {},
            '30': {}
        }

    elif tree_type == "deep_depth":

        tree = "deep"
        # Your decision tree dictionary
        decision_tree = {
            '0': {'right': [-1, '1'], 'left': [3, '16']},
            '1': {'up': [-1, '2'], 'down': [2, '9']},
            '2': {'up': [-1, '3'], 'down': [2, '6']},
            '3': {'up': [-1, '4'], 'down': [2, '5']},
            '4': {},
            '5': {},
            '6': {'up': [-1, '7'], 'down': [2, '8']},
            '7': {},
            '8': {},
            '9': {'up': [-1, '10'], 'down': [2, '13']},
            '10': {'right': [-1, '11'], 'left': [3, '12']},
            '11': {},
            '12': {},
            '13': {'up': [-1, '14'], 'down': [2, '15']},
            '14': {},
            '15': {},
            '16': {'up': [-1, '17'], 'down': [2, '24']},
            '17': {'up': [-1, '18'], 'down': [2, '21']},
            '18': {'up': [-1, '19'], 'down': [2, '20']},
            '19': {},
            '20': {},
            '21': {'up': [-1, '22'], 'down': [2, '23']},
            '22': {},
            '23': {},
            '24': {'up': [-1, '25'], 'down': [2, '28']},
            '25': {'up': [-1, '26'], 'down': [2, '27']},
            '26': {},
            '27': {},
            '28': {'up': [-1, '29'], 'down': [2, '30']},
            '29': {},
            '30': {}
        }

    elif (tree_type == "wide_breadth"):
        tree = "wide"
        # Your decision tree dictionary
        decision_tree = {
            '0': {
                'right': [-1, '1'], 'left': [3, '2'],
                'up': [-1, '3'], 'down': [2, '4'], 'up1': [-1, '5']
            },
            '1': {
                'up': [-1, '6'], 'down': [2, '7'],
                'right': [-1, '8'], 'left': [3, '9'], 'up1': [-1, '10']
            },
            '2': {
                'up': [-1, '11'], 'down': [2, '12'],
                'right': [-1, '13'], 'left': [3, '14'],
                'up1': [-1, '15']
            },
            '3': {
                'up': [-1, '16'], 'down': [2, '17'],
                'right': [-1, '18'], 'left': [3, '19'],
                'up1': [-1, '20']
            },
            '4': {
                'up': [-1, '21'], 'down': [2, '22'],
                'right': [-1, '23'], 'left': [3, '24'],
                'up1': [-1, '25']
            },
            '5': {
                'up': [-1, '26'], 'down': [2, '27'],
                'right': [-1, '28'], 'left': [3, '29'],
                'up1': [-1, '30']
            },
            '6': {},
            '7': {},
            '8': {},
            '9': {},
            '10': {},
            '11': {},
            '12': {},
            '13': {},
            '14': {},
            '15': {},
            '16': {},
            '17': {},
            '18': {},
            '19': {},
            '20': {},
            '21': {},
            '22': {},
            '23': {},
            '24': {},
            '25': {},
            '26': {},
            '27': {},
            '28': {},
            '29': {},
            '30': {}
        }

    else:

        tree = "wide"
        # Your decision tree dictionary
        decision_tree = {
            '0': {
                'right': [-1, '1'], 'left': [3, '7'],
                'up': [-1, '13'], 'down': [2, '19'],
                'up1': [-1, '25']
            },
            '1': {
                'up': [-1, '2'], 'down': [2, '3'],
                'right': [-1, '4'], 'left': [3, '5'], 'up1': [-1, '6']
            },
            '2': {},
            '3': {},
            '4': {},
            '5': {},
            '6': {},
            '7': {
                'up': [-1, '8'], 'down': [2, '9'],
                'right': [-1, '10'], 'left': [3, '11'],
                'up1': [-1, '12']
            },
            '8': {},
            '9': {},
            '10': {},
            '11': {},
            '12': {},
            '13': {
                'up': [-1, '14'], 'down': [2, '15'],
                'right': [-1, '16'], 'left': [3, '17'],
                'up1': [-1, '18']
            },
            '14': {},
            '15': {},
            '16': {},
            '17': {},
            '18': {},
            '19': {
                'up': [-1, '20'], 'down': [2, '21'],
                'right': [-1, '22'], 'left': [3, '23'],
                'up1': [-1, '24']
            },
            '20': {},
            '21': {},
            '22': {},
            '23': {},
            '24': {},
            '25': {
                'up': [-1, '26'], 'down': [2, '27'],
                'right': [-1, '28'], 'left': [3, '29'],
                'up1': [-1, '30']
            },
            '26': {},
            '27': {},
            '28': {},
            '29': {},
            '30': {}
        }


# ---------------------------------------------------------
# 6. PATH ANALYSIS & TENSOR MATRICES
# ---------------------------------------------------------
results = helper.analyze_tree_paths(decision_tree)

(path_names, path_leaf_dict, sibling_map, node_path_map, node_path_name,
 path_indices, node_indices, est_best_path_map, path_node_map) = results

num_paths = len(path_names)


# Create path map
path_map_np = np.zeros((num_paths, time_steps), dtype=int)
for i in range(num_paths):
    path_map_np[i, :] = [
        1 if f"{n+1}" in path_names[i] else 0 for n in range(time_steps)]
path_map = tf.convert_to_tensor(path_map_np, dtype=tf.float32)

# Create path covariance matrix
path_cov_mat_np = np.zeros((num_paths, num_paths, time_steps), dtype=int)
for i in range(num_paths):
    for j in range(num_paths):
        path_cov_mat_np[i, j, :] = path_map_np[i, :] * path_map_np[j, :]
path_cov_mat = tf.convert_to_tensor(path_cov_mat_np, dtype=tf.float32)

# Create index path map
index_path_map = {path_indices[i]: node_indices[i]
                  for i in range(len(path_indices))}
