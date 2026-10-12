"""Fit model lambda and alpha values to participant recall behavior.

For planning trials, this script combines two negative log-likelihood
components for each participant and parameter-grid point:

1. The participant-chosen path under the model's final path policy.
2. Each participant-selected node reward under the model's final reward
   reconstruction policy.

The recalled-reward NLL receives a default planning weight of 0.5, balancing
the two recall observations against the one path choice on each trial. For
exposure trials, the fit uses the unscaled recalled-reward NLL alone because
there is no participant path choice. Each
participant is assigned one trained-model seed, held fixed across the entire
parameter grid. The minimum weighted total negative log-likelihood is the grid
fit. After selection, the fitted parameter pair is simulated under every
requested trained-model seed. For the 7-node task, the post-fit plotting step
also visualizes summed chosen-path NLL and summed recalled-reward NLL
separately over the complete lambda/alpha grid.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import tensorflow as tf


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_paths import (
    behavior_path as behavior_input_path,
    REPO_ROOT,  # noqa: E402
    DATA_DIR,
    SIMULATION_DIR,
    SUMMARY_DIR,
    WEIGHT_DIR,
    ensure_directory,
    print_local_scp_command,
    require_existing_path,
)


MAX_PARTICIPANT_FIT_RUN_SUFFIX_BYTES = 160


@dataclass(frozen=True)
class TreeSpecification:
    """Command-line tree specification resolved for the model configuration."""

    size: int
    tree_type: str
    label: str


@dataclass
class BehavioralTrials:
    """Model inputs and participant selections aligned by trial index."""

    trials: pd.DataFrame
    recalls: pd.DataFrame
    reward_inputs: np.ndarray


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Grid-fit lambda and alpha (or rho) to participant planning or "
            "exposure behavior."
        )
    )
    parser.add_argument(
        "tree",
        help=(
            "3, 7, 13, 40, a named 31-node tree, or 31:TREE_TYPE. "
            "The centralized default behavioral file is currently available "
            "only for the 7-node task."
        ),
    )
    parser.add_argument("lambda_values", help="Comma-separated lambda values.")
    parser.add_argument(
        "alpha_values",
        help=(
            "Comma-separated alpha values, or rho = alpha / lambda values "
            "when --rho is supplied."
        ),
    )
    parser.add_argument(
        "--rho",
        dest="rho_grid",
        action="store_true",
        help=(
            "Interpret the third positional list as rho = alpha / lambda "
            "and fit paired cells with alpha = lambda * rho."
        ),
    )
    parser.add_argument(
        "--condition",
        choices=("planning", "exposure"),
        default="planning",
        help=(
            "Behavioral condition to fit. Planning uses chosen-path NLL plus "
            "recalled-reward NLL; exposure uses recalled-reward NLL only "
            "(default: planning)."
        ),
    )
    model_seed_group = parser.add_mutually_exclusive_group()
    model_seed_group.add_argument(
        "--seed-values",
        default=None,
        help=(
            "Comma-separated trained-model seeds. Each participant is "
            "randomly assigned one seed for NLL evaluation, and every seed is "
            "used for post-fit simulations (default: 1,2,3,4,5)."
        ),
    )
    model_seed_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Legacy single trained-model seed. Equivalent to supplying one "
            "value with --seed-values."
        ),
    )
    parser.add_argument(
        "--seed-selection-seed",
        type=int,
        default=2026,
        help=(
            "Random seed controlling the participant-to-model-seed "
            "assignment (default: 2026)."
        ),
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=32,
        help=(
            "Latent-state samples averaged for each model policy "
            "estimate (default: 32)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Participant graphs evaluated per model call (default: 128).",
    )
    parser.add_argument(
        "--evaluation-seed",
        type=int,
        default=2026,
        help="Random seed for Monte Carlo policy evaluation (default: 2026).",
    )
    parser.add_argument(
        "--reward-nll-scale",
        type=float,
        default=None,
        help=(
            "Multiplier applied to recalled-reward NLL when computing the "
            "fit objective. Defaults to 0.5 for planning and 1 for exposure."
        ),
    )
    parser.add_argument(
        "--simulation-trials-per-seed",
        type=int,
        default=100,
        help=(
            "Post-fit random trials generated for every participant and each "
            "trained-model seed (default: 100)."
        ),
    )
    parser.add_argument(
        "--simulation-random-seed",
        type=int,
        default=2027,
        help="Random seed for post-fit trials and responses (default: 2027).",
    )
    parser.add_argument(
        "--skip-fitted-plots",
        action="store_true",
        help="Write fit and simulation CSVs without running the fitted R plots.",
    )
    parser.add_argument(
        "--max-action-gradient-scale",
        type=float,
        default=1.0,
        help=(
            "Maximum backward-only action-gradient multiplier identifying "
            "the trained model artifacts (default: 1)."
        ),
    )
    parser.add_argument(
        "--parameter-multiplier",
        type=float,
        default=None,
        help=(
            "Explicit common loss-weight multiplier identifying the trained "
            "model artifacts. Omit to use the tree-dependent default "
            "(1 for the 13-node tree)."
        ),
    )
    parser.add_argument(
        "--initial-critic-coefficient",
        type=float,
        default=None,
        help=(
            "Explicit initial critic coefficient identifying the trained "
            "model artifacts. Omit for the tree-dependent default "
            "(1 for the 13-node tree)."
        ),
    )
    parser.add_argument(
        "--model-epochs",
        type=int,
        default=320,
        help=(
            "Requested training duration recorded with the fit outputs "
            "(default: 320)."
        ),
    )
    reconstruction_head_group = parser.add_mutually_exclusive_group()
    reconstruction_head_group.add_argument(
        "--nonlinear-scale",
        dest="nonlinear_scale",
        action="store_true",
        help=(
            "Load models trained with the default nonlinear location-scale "
            "head."
        ),
    )
    reconstruction_head_group.add_argument(
        "--nonlinear-categorical",
        dest="nonlinear_scale",
        action="store_false",
        help="Load models trained with the nonlinear categorical head.",
    )
    parser.set_defaults(nonlinear_scale=True)
    parser.add_argument(
        "--reconstruction-hidden-units",
        type=int,
        default=128,
        help=(
            "Reconstruction hidden width identifying a tagged architecture "
            "artifact (default: 128)."
        ),
    )
    parser.add_argument(
        "--rnn-units",
        type=int,
        default=64,
        help=(
            "Recurrent hidden-state width identifying the trained model "
            "artifact (default: 64)."
        ),
    )
    parser.add_argument(
        "--reconstruction-loss-scope",
        choices=("all_timesteps", "final_timestep"),
        default="final_timestep",
        help=(
            "Reconstruction supervision scope identifying a tagged "
            "architecture artifact (default: final_timestep)."
        ),
    )
    parser.add_argument(
        "--final-learning-rate",
        type=float,
        default=3e-4,
        help=(
            "Terminal learning rate identifying the trained model artifact "
            "(default: 3e-4)."
        ),
    )
    parser.add_argument(
        "--late-lr-plateau-start-epoch",
        type=int,
        default=200,
        help="Late-LR plateau start identifying the model (default: 200).",
    )
    parser.add_argument(
        "--late-lr-plateau-end-epoch",
        type=int,
        default=300,
        help="Late-LR plateau end identifying the model (default: 300).",
    )
    parser.add_argument(
        "--no-late-lr-plateau",
        action="store_true",
        help="Load a model artifact without late-LR plateau tags.",
    )
    parser.add_argument(
        "--no-critic-loss",
        action="store_true",
        help="Load model artifacts trained without the auxiliary critic loss.",
    )
    parser.add_argument(
        "--behavior-csv",
        type=Path,
        default=None,
        help=(
            "Optional behavioral CSV override. Centralized defaults are "
            "data/analysis/processed_data/exp1_beh.csv for tree 7 and "
            "data/analysis/processed_data/exp2_beh.csv for tree 13."
        ),
    )
    parser.add_argument(
        "--subject",
        default=None,
        help=(
            "Fit only this participant. Intended for the parallel "
            "participant-shard launcher."
        ),
    )
    parser.add_argument(
        "--shard-directory",
        type=Path,
        default=None,
        help=(
            "Write participant-specific grid and best-fit shards here and "
            "skip fitted simulations and plots."
        ),
    )
    args = parser.parse_args()

    if args.reward_nll_scale is None:
        args.reward_nll_scale = 0.5 if args.condition == "planning" else 1.0

    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be non-negative.")
    if args.seed_selection_seed < 0:
        parser.error("--seed-selection-seed must be non-negative.")
    if args.mc_samples < 1:
        parser.error("--mc-samples must be positive.")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    if args.evaluation_seed < 0:
        parser.error("--evaluation-seed must be non-negative.")
    if (
        not np.isfinite(args.reward_nll_scale)
        or args.reward_nll_scale <= 0.0
    ):
        parser.error("--reward-nll-scale must be finite and positive.")
    if args.simulation_trials_per_seed < 1:
        parser.error("--simulation-trials-per-seed must be positive.")
    if args.simulation_random_seed < 0:
        parser.error("--simulation-random-seed must be non-negative.")
    if (
        not np.isfinite(args.max_action_gradient_scale)
        or args.max_action_gradient_scale < 1.0
    ):
        parser.error(
            "--max-action-gradient-scale must be finite and at least 1."
        )
    if args.parameter_multiplier is not None and (
        not np.isfinite(args.parameter_multiplier)
        or args.parameter_multiplier <= 0.0
    ):
        parser.error("--parameter-multiplier must be finite and positive.")
    if args.initial_critic_coefficient is not None and (
        not np.isfinite(args.initial_critic_coefficient)
        or args.initial_critic_coefficient < 0.0
    ):
        parser.error(
            "--initial-critic-coefficient must be finite and non-negative."
        )
    if (
        args.condition == "exposure"
        and args.initial_critic_coefficient is not None
    ):
        parser.error(
            "--initial-critic-coefficient is not applicable when "
            "--condition exposure."
        )
    if args.condition == "exposure":
        args.no_critic_loss = True
    if args.model_epochs < 1:
        parser.error("--model-epochs must be positive.")
    if args.reconstruction_hidden_units < 1:
        parser.error("--reconstruction-hidden-units must be positive.")
    if args.rnn_units < 1:
        parser.error("--rnn-units must be positive.")
    if (
        not np.isfinite(args.final_learning_rate)
        or args.final_learning_rate <= 0.0
        or args.final_learning_rate > 3e-4
    ):
        parser.error("--final-learning-rate must be in (0, 3e-4].")
    if args.no_late_lr_plateau:
        args.late_lr_plateau_start_epoch = None
        args.late_lr_plateau_end_epoch = None
    elif not (
        0
        < args.late_lr_plateau_start_epoch
        < args.late_lr_plateau_end_epoch
        < args.model_epochs
    ):
        parser.error(
            "plateau epochs must satisfy 0 < start < end < model epochs"
        )

    return args


def parse_number_list(text: str, label: str) -> list[float]:
    """Parse a comma-separated list of unique finite numbers."""
    tokens = [token.strip() for token in text.split(",")]
    if not tokens or any(not token for token in tokens):
        raise ValueError(f"{label} must be a non-empty comma-separated list.")

    try:
        values = [float(token) for token in tokens]
    except ValueError as error:
        raise ValueError(f"{label} contains a non-numeric value.") from error

    if not all(np.isfinite(value) for value in values):
        raise ValueError(f"{label} values must all be finite.")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} contains duplicate numeric values.")
    return values


def alpha_from_lambda_rho(lambda_value: float, rho_value: float) -> float:
    """Return the same stable raw-alpha value used by the grid launcher."""
    alpha_value = lambda_value * rho_value
    if not np.isfinite(alpha_value):
        raise ValueError(
            "Derived alpha is not finite for "
            f"lambda={lambda_value:g}, rho={rho_value:g}."
        )
    return float(f"{alpha_value:.12g}")


def parse_seed_list(text: str) -> list[int]:
    """Parse a comma-separated list of unique non-negative model seeds."""
    tokens = [token.strip() for token in text.split(",")]
    if not tokens or any(not token for token in tokens):
        raise ValueError(
            "seed_values must be a non-empty comma-separated list."
        )
    if any(not token.isdigit() for token in tokens):
        raise ValueError("seed_values must contain non-negative integers.")

    seeds = [int(token) for token in tokens]
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed_values contains duplicate values.")
    return seeds


def canonical_subject(value: object) -> str:
    """Return a stable textual participant identifier."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def safe_subject_token(value: object) -> str:
    """Return a filename-safe participant identifier."""
    token = canonical_subject(value)
    safe = "".join(
        character if character.isalnum() or character in "_-" else "_"
        for character in token
    )
    if not safe:
        raise ValueError("Participant identifier has no filename-safe text.")
    return safe


def resolve_tree(tree: str) -> TreeSpecification:
    """Resolve the launcher-compatible tree notation."""
    named_types = {
        "deep_breadth",
        "deep_depth",
        "wide_breadth",
        "wide_depth",
    }
    if tree in {"3", "7", "13", "40"}:
        return TreeSpecification(int(tree), "standard", tree)
    if tree in named_types:
        return TreeSpecification(31, tree, tree)
    if tree.startswith("31:") and tree[3:] in named_types:
        tree_type = tree[3:]
        return TreeSpecification(31, tree_type, tree_type)
    if tree == "31":
        raise ValueError(
            "A 31-node tree needs its type, for example deep_depth."
        )
    raise ValueError(f"Invalid tree specification: {tree}")


def load_model_config(
    tree: TreeSpecification,
    lambda_values: list[float],
    alpha_values: list[float],
    seed: int,
    max_action_gradient_scale: float,
    parameter_multiplier: float | None,
    initial_critic_coefficient: float | None,
    model_epochs: int,
    nonlinear_scale: bool,
    reconstruction_hidden_units: int,
    reconstruction_loss_scope: str,
    no_critic_loss: bool,
    final_learning_rate: float,
    late_lr_plateau_start_epoch: int | None,
    late_lr_plateau_end_epoch: int | None,
    model_condition: str,
    action_entropy_bonus: float = 0.0,
    rnn_units: int = 64,
) -> ModuleType:
    """Import the existing command-line configuration for a fitting run."""
    original_argv = sys.argv[:]
    sys.argv = [
        "fit_participant_parameters.py",
        ",".join(str(value) for value in lambda_values),
        ",".join(str(value) for value in alpha_values),
        "default",
        "default",
        str(model_epochs),
        str(seed),
        str(tree.size),
        "simulate",
        tree.tree_type,
        "1",
        "--max-action-gradient-scale",
        str(max_action_gradient_scale),
        "--final-learning-rate",
        str(final_learning_rate),
        "--reconstruction-hidden-units",
        str(reconstruction_hidden_units),
        "--reconstruction-loss-scope",
        reconstruction_loss_scope,
        "--condition",
        model_condition,
        "--rnn-units",
        str(rnn_units),
    ]
    if parameter_multiplier is not None:
        sys.argv.extend(
            ["--parameter-multiplier", str(parameter_multiplier)]
        )
    if initial_critic_coefficient is not None:
        sys.argv.extend(
            [
                "--initial-critic-coefficient",
                str(initial_critic_coefficient),
            ]
        )
    if action_entropy_bonus != 0.0:
        sys.argv.extend(
            ["--action-entropy-bonus", str(action_entropy_bonus)]
        )
    if nonlinear_scale:
        sys.argv.append("--nonlinear-scale")
    else:
        sys.argv.append("--nonlinear-categorical")
    if no_critic_loss:
        sys.argv.append("--no-critic-loss")
    if late_lr_plateau_start_epoch is not None:
        sys.argv.extend(
            [
                "--late-lr-plateau-start-epoch",
                str(late_lr_plateau_start_epoch),
                "--late-lr-plateau-end-epoch",
                str(late_lr_plateau_end_epoch),
            ]
        )
    try:
        from . import config  # pylint: disable=import-outside-toplevel
    finally:
        sys.argv = original_argv
    return config


def model_configuration_artifact_suffix(
    config: ModuleType,
    tree: TreeSpecification,
    lambda_value: float,
    alpha_value: float,
    seed: int,
) -> str:
    """Return the centralized model tags following lambda/alpha/seed."""
    artifact_name = config.model_artifact_name(
        lambda_value,
        alpha_value,
        seed,
    )
    artifact_prefix = (
        f"lambda_{lambda_value}_alpha_{alpha_value}_seed_{seed}"
        f"_tree_{tree.size}"
    )
    if tree.size == 31:
        artifact_prefix += f"_{tree.tree_type}"
    if not artifact_name.startswith(artifact_prefix):
        raise RuntimeError(
            "Centralized model artifact name does not start with its "
            f"expected parameter prefix: {artifact_name}"
        )
    return artifact_name[len(artifact_prefix):]


def compact_participant_fit_run_suffix(full_suffix: str) -> tuple[str, str]:
    """Bound a run suffix and return it with its full-configuration hash."""
    if not full_suffix.isascii():
        raise ValueError("Participant-fit artifact suffix must be ASCII.")

    configuration_hash = hashlib.sha256(
        full_suffix.encode("ascii")
    ).hexdigest()[:12]
    if len(full_suffix.encode("ascii")) <= MAX_PARTICIPANT_FIT_RUN_SUFFIX_BYTES:
        return full_suffix, configuration_hash

    separator = f"_cfg_{configuration_hash}_"
    retained_length = MAX_PARTICIPANT_FIT_RUN_SUFFIX_BYTES - len(separator)
    prefix_length = (retained_length + 1) // 2
    suffix_length = retained_length - prefix_length
    compact_suffix = (
        full_suffix[:prefix_length]
        + separator
        + full_suffix[-suffix_length:]
    )
    if len(compact_suffix.encode("ascii")) > MAX_PARTICIPANT_FIT_RUN_SUFFIX_BYTES:
        raise RuntimeError("Compacted participant-fit suffix is still too long.")
    return compact_suffix, configuration_hash


def default_behavior_path(tree: TreeSpecification) -> Path:
    """Return the centralized behavioral input supported by default."""
    default_file_by_tree_size = {
        7: "exp1_beh.csv",
        13: "exp2_beh.csv",
    }
    default_file = default_file_by_tree_size.get(tree.size)
    if default_file is None:
        raise ValueError(
            "No default exact-path behavioral file is configured for tree "
            f"'{tree.label}'. Supply --behavior-csv with subject, graph, "
            "est_best_path, node, estimated_reward, condition, and one true "
            "reward column per non-root node."
        )
    return behavior_input_path("exp1" if tree.size == 7 else "exp2")


def normalize_path_name(value: object) -> str:
    """Normalize a participant path string to the model's path-name format."""
    pieces = [piece.strip() for piece in str(value).split(",")]
    if not pieces or any(not piece for piece in pieces):
        raise ValueError(f"Invalid participant path: {value!r}")
    try:
        node_ids = [str(int(float(piece))) for piece in pieces]
    except ValueError as error:
        raise ValueError(f"Invalid participant path: {value!r}") from error
    if node_ids[0] != "0":
        node_ids.insert(0, "0")
    return ", ".join(node_ids)


def one_unique_numeric(
    group: pd.DataFrame,
    column: str,
    subject: object,
    graph: object,
) -> float:
    """Return one consistent numeric value from a participant graph."""
    values = pd.to_numeric(group[column], errors="coerce").dropna().unique()
    if len(values) != 1:
        raise ValueError(
            f"Expected one {column} value for subject={subject}, "
            f"graph={graph}; found {len(values)}."
        )
    return float(values[0])


def reward_category_index(
    response: float,
    minimum_reward: int,
    maximum_reward: int,
) -> int:
    """Map a selected reward to the model's descending category index."""
    category_index = int(round(float(maximum_reward) - response))
    maximum_index = maximum_reward - minimum_reward
    if category_index < 0 or category_index > maximum_index:
        raise ValueError(
            f"Reward response is outside [{minimum_reward}, "
            f"{maximum_reward}]: {response}"
        )
    represented_reward = float(maximum_reward) - category_index
    if not np.isclose(response, represented_reward, atol=1e-6):
        raise ValueError(
            "Reward responses must be integer categories from "
            f"{minimum_reward} through {maximum_reward}; "
            f"received {response}."
        )
    return category_index


def prepare_behavioral_trials(
    behavior: pd.DataFrame,
    config: ModuleType,
    condition: str = "planning",
) -> BehavioralTrials:
    """Build model inputs and choices from one behavioral condition."""
    if condition not in {"planning", "exposure"}:
        raise ValueError(
            "Behavioral condition must be planning or exposure; "
            f"received {condition!r}."
        )
    reward_columns = [f"n{node}_reward" for node in range(1, config.time_steps + 1)]
    required_columns = {
        "subject",
        "graph",
        "condition",
        "node",
        "estimated_reward",
        *reward_columns,
    }
    if condition == "planning":
        required_columns.add("est_best_path")
    missing_columns = sorted(required_columns.difference(behavior.columns))
    if missing_columns:
        raise ValueError(
            "Behavioral CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    selected_condition = behavior.loc[
        behavior["condition"].astype(str).str.lower().eq(condition)
    ].copy()
    selected_condition = selected_condition.dropna(subset=["subject", "graph"])
    if condition == "exposure":
        # The processed Exp. 1 data can contain non-recall task rows whose
        # subject/graph identifiers overlap exposure recall rows while their
        # reward configurations differ.  Those rows provide neither a reward
        # response nor any likelihood term for an exposure-only fit.  Remove
        # them before defining trials so they cannot merge an unrelated
        # stimulus into the participant's recall trial.
        selected_condition = selected_condition.dropna(
            subset=["node", "estimated_reward"]
        )
    if selected_condition.empty:
        raise ValueError(
            f"Behavioral CSV contains no {condition}-condition rows."
        )

    trial_records: list[dict[str, object]] = []
    recall_records: list[dict[str, object]] = []
    reward_inputs: list[list[float]] = []

    trial_group_columns = ["subject", "graph"]
    if condition == "exposure":
        # Exposure graph numbers can restart within a participant.  Include
        # the complete presented stimulus in the key so distinct exposure
        # trials that reuse a graph number remain separate model inputs.
        trial_group_columns.extend(reward_columns)
    grouped = selected_condition.groupby(
        trial_group_columns,
        sort=True,
        dropna=False,
    )
    for group_key, group in grouped:
        subject, graph = group_key[:2]
        graph_rewards = [
            one_unique_numeric(group, column, subject, graph)
            for column in reward_columns
        ]
        trial_index = len(trial_records)
        trial_record: dict[str, object] = {
            "trial_index": trial_index,
            "subject": subject,
            "graph": graph,
            "condition": condition,
        }
        if condition == "planning":
            path_names = {
                normalize_path_name(value)
                for value in group["est_best_path"].dropna().unique()
            }
            if len(path_names) != 1:
                raise ValueError(
                    "Expected exactly one chosen path for "
                    f"subject={subject}, graph={graph}; found {len(path_names)}."
                )
            chosen_path = next(iter(path_names))
            if chosen_path not in config.est_best_path_map:
                raise ValueError(
                    f"Participant path '{chosen_path}' is invalid for the "
                    f"configured tree (subject={subject}, graph={graph})."
                )
            trial_record.update(
                {
                    "chosen_path": chosen_path,
                    "path_index": int(config.est_best_path_map[chosen_path]),
                }
            )
        trial_records.append(trial_record)
        reward_inputs.append(graph_rewards)

        recall_rows = group.dropna(subset=["node", "estimated_reward"])
        if (
            condition == "planning"
            and recall_rows.duplicated(subset=["node"]).any()
        ):
            raise ValueError(
                "Duplicate reward responses for "
                f"subject={subject}, graph={graph}."
            )
        for row in recall_rows.itertuples(index=False):
            node_index = int(float(row.node)) - 1
            if node_index < 0 or node_index >= config.time_steps:
                raise ValueError(
                    f"Invalid node {row.node} for subject={subject}, graph={graph}."
                )
            response = float(row.estimated_reward)
            recall_records.append(
                {
                    "trial_index": trial_index,
                    "subject": subject,
                    "graph": graph,
                    "condition": condition,
                    "node": node_index + 1,
                    "selected_reward": response,
                    "category_index": reward_category_index(
                        response,
                        config.reconstruction_min_reward,
                        config.reconstruction_max_reward,
                    ),
                }
            )

    trials = pd.DataFrame.from_records(trial_records)
    recalls = pd.DataFrame.from_records(recall_records)
    if trials.empty:
        raise ValueError(
            f"No complete {condition}-condition trials were available for fitting."
        )
    if recalls.empty:
        raise ValueError(
            f"No {condition}-condition reward responses were available for fitting."
        )

    reward_array = np.asarray(reward_inputs, dtype=np.float32)[..., np.newaxis]
    return BehavioralTrials(trials, recalls, reward_array)


def assign_nll_model_seeds(
    behavior: BehavioralTrials,
    model_seeds: list[int],
    selection_seed: int,
) -> pd.DataFrame:
    """Assign one reproducible trained-model seed to each participant."""
    subjects = behavior.trials["subject"].drop_duplicates().to_numpy()
    random_generator = np.random.default_rng(selection_seed)
    selected_seeds = random_generator.choice(
        np.asarray(model_seeds, dtype=int),
        size=len(subjects),
        replace=True,
    )
    return pd.DataFrame(
        {
            "subject": subjects,
            "nll_model_seed": selected_seeds.astype(int),
        }
    )


def subset_behavioral_trials(
    behavior: BehavioralTrials,
    subjects: np.ndarray,
) -> BehavioralTrials:
    """Subset participants and remap their model-input trial indices."""
    selected_trials = behavior.trials.loc[
        behavior.trials["subject"].isin(subjects)
    ].copy()
    if selected_trials.empty:
        raise ValueError("Cannot evaluate an empty participant subset.")

    original_trial_indices = selected_trials["trial_index"].to_numpy(dtype=int)
    trial_index_map = {
        original_index: new_index
        for new_index, original_index in enumerate(original_trial_indices)
    }
    selected_trials["trial_index"] = np.arange(
        len(selected_trials),
        dtype=int,
    )

    selected_recalls = behavior.recalls.loc[
        behavior.recalls["subject"].isin(subjects)
    ].copy()
    selected_recalls["trial_index"] = selected_recalls["trial_index"].map(
        trial_index_map
    )
    if selected_recalls["trial_index"].isna().any():
        raise RuntimeError("Failed to remap a participant recall trial index.")
    selected_recalls["trial_index"] = selected_recalls["trial_index"].astype(
        int
    )

    selected_inputs = behavior.reward_inputs[original_trial_indices]
    return BehavioralTrials(
        selected_trials,
        selected_recalls,
        selected_inputs,
    )


def build_and_load_model(
    config: ModuleType,
    lambda_value: float,
    alpha_value: float,
    seed: int,
    extra_weight_paths: tuple[Path, ...] = (),
) -> tuple[tf.keras.Model, Path]:
    """Construct one grid model and load its centralized weight file.

    ``extra_weight_paths`` lets post-fit simulation reuse the exact artifact
    convention recorded by an older fit table.  Config-derived current and
    compatibility names remain the fallback for ordinary fitting calls.
    """
    from .model import (  # pylint: disable=import-outside-toplevel
        VariationalRNN,
        build_decoder,
        build_encoder,
    )

    def construct_model() -> tf.keras.Model:
        """Build a fresh model so a failed load cannot taint the next try."""
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
        model = VariationalRNN(
            encoder=encoder,
            decoder=decoder,
            rnn_units=config.rnn_units,
            latent_dim=config.latent_dim,
            time_steps=config.time_steps,
            num_paths=config.num_paths,
            index_path_map=config.index_path_map,
            path_map=config.path_map,
            path_cov_mat=config.path_cov_mat,
            alpha=alpha_value,
            lambda_=lambda_value,
            tree_type=config.tree_type,
            reward_normalization_constant=(
                config.reward_normalization_constant
            ),
            max_action_gradient_scale=config.max_action_gradient_scale,
            reconstruction_head_type=config.reconstruction_head_type,
            reconstruction_hidden_units=config.reconstruction_hidden_units,
            reconstruction_loss_scope=config.reconstruction_loss_scope,
            critic_loss_enabled=config.critic_loss_enabled,
            parameter_multiplier=config.parameter_multiplier,
            action_entropy_bonus=config.action_entropy_bonus,
            model_condition=config.model_condition,
        )
        dummy_input = tf.zeros(
            (1, config.time_steps, 1),
            dtype=tf.float32,
        )
        model(dummy_input, training=False)
        return model

    artifact_candidates = (
        config.model_artifact_name_candidates(
            lambda_value,
            alpha_value,
            seed,
        )
        if hasattr(config, "model_artifact_name_candidates")
        else (
            config.model_artifact_name(lambda_value, alpha_value, seed),
        )
    )
    configured_weight_paths = tuple(
        WEIGHT_DIR / (artifact_name + ".weights.h5")
        for artifact_name in artifact_candidates
    )
    candidate_weight_paths: list[Path] = []
    seen_weight_paths: set[Path] = set()
    for candidate in (*extra_weight_paths, *configured_weight_paths):
        normalized_candidate = Path(candidate).expanduser()
        if normalized_candidate not in seen_weight_paths:
            candidate_weight_paths.append(normalized_candidate)
            seen_weight_paths.add(normalized_candidate)
    existing_weight_paths = [
        path for path in candidate_weight_paths if path.exists()
    ]
    if not existing_weight_paths:
        attempted_paths = "\n  - ".join(
            str(path) for path in candidate_weight_paths
        )
        raise FileNotFoundError(
            "Model weight file does not exist. Tried:\n  - "
            + attempted_paths
        )

    load_failures: list[tuple[Path, str]] = []
    for candidate_index, weight_path in enumerate(existing_weight_paths):
        if candidate_index > 0:
            tf.keras.backend.clear_session()
        fitted_model = construct_model()
        try:
            fitted_model.load_weights(str(weight_path))
        except (OSError, ValueError) as error:
            failure_message = str(error).splitlines()[0]
            load_failures.append((weight_path, failure_message))
            print(
                "Rejected incompatible model weight candidate | "
                f"tree={config.time_steps + 1} | "
                f"lambda={lambda_value:g} | alpha={alpha_value:g} | "
                f"seed={seed} | path={weight_path} | "
                f"reason={failure_message}",
                flush=True,
            )
            del fitted_model
            continue
        return fitted_model, weight_path

    failure_details = "\n".join(
        f"  - {path}: {message}" for path, message in load_failures
    )
    raise ValueError(
        "Existing model weight files are incompatible with the requested "
        f"{config.time_steps + 1}-node architecture. This commonly means a "
        "legacy filename without a tree tag was overwritten by another tree "
        "size. Rejected candidates:\n"
        + failure_details
    )


def estimate_policies(
    fitted_model: tf.keras.Model,
    reward_inputs: np.ndarray,
    time_steps: int,
    num_paths: int,
    num_categories: int,
    mc_samples: int,
    batch_size: int,
    evaluation_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate path and final reconstruction policies by Monte Carlo mean."""
    trial_count = reward_inputs.shape[0]
    action_probabilities = np.zeros((trial_count, num_paths), dtype=np.float64)
    reconstruction_probabilities = np.zeros(
        (trial_count, time_steps, num_categories),
        dtype=np.float64,
    )

    tf.random.set_seed(evaluation_seed)
    for start in range(0, trial_count, batch_size):
        stop = min(start + batch_size, trial_count)
        model_inputs = tf.convert_to_tensor(
            reward_inputs[start:stop],
            dtype=tf.float32,
        )
        action_sum = np.zeros((stop - start, num_paths), dtype=np.float64)
        reconstruction_sum = np.zeros(
            (stop - start, time_steps, num_categories),
            dtype=np.float64,
        )

        for _ in range(mc_samples):
            outputs = fitted_model(model_inputs, training=False)
            action_sum += np.asarray(outputs[1], dtype=np.float64)
            reconstruction_sum += np.asarray(
                outputs[0][:, -1, :, :],
                dtype=np.float64,
            )

        action_probabilities[start:stop] = action_sum / mc_samples
        reconstruction_probabilities[start:stop] = (
            reconstruction_sum / mc_samples
        )

    return action_probabilities, reconstruction_probabilities


def participant_nll(
    behavior: BehavioralTrials,
    action_probabilities: np.ndarray,
    reconstruction_probabilities: np.ndarray,
    lambda_value: float,
    alpha_value: float,
    model_seed: int,
    weight_path: Path,
    reward_nll_scale: float,
    condition: str = "planning",
) -> pd.DataFrame:
    """Calculate raw NLL components and their weighted fit objective."""
    if condition not in {"planning", "exposure"}:
        raise ValueError(
            "Fit condition must be planning or exposure; "
            f"received {condition!r}."
        )
    epsilon = np.finfo(np.float64).tiny

    trials = behavior.trials.copy()
    if condition == "planning":
        trial_indices = trials["trial_index"].to_numpy(dtype=int)
        path_indices = trials["path_index"].to_numpy(dtype=int)
        chosen_path_probabilities = action_probabilities[
            trial_indices,
            path_indices,
        ]
        trials["path_nll"] = -np.log(
            np.clip(chosen_path_probabilities, epsilon, 1.0)
        )
        path_summary = trials.groupby("subject", as_index=False).agg(
            path_nll=("path_nll", "sum"),
            n_path_choices=("path_nll", "size"),
        )
    else:
        path_summary = trials[["subject"]].drop_duplicates().copy()
        path_summary["path_nll"] = 0.0
        path_summary["n_path_choices"] = 0

    recalls = behavior.recalls.copy()
    recall_trial_indices = recalls["trial_index"].to_numpy(dtype=int)
    recall_node_indices = recalls["node"].to_numpy(dtype=int) - 1
    category_indices = recalls["category_index"].to_numpy(dtype=int)
    selected_reward_probabilities = reconstruction_probabilities[
        recall_trial_indices,
        recall_node_indices,
        category_indices,
    ]
    recalls["reconstruction_nll"] = -np.log(
        np.clip(selected_reward_probabilities, epsilon, 1.0)
    )
    reconstruction_summary = recalls.groupby("subject", as_index=False).agg(
        reconstruction_nll=("reconstruction_nll", "sum"),
        n_reward_choices=("reconstruction_nll", "size"),
    )

    summary = path_summary.merge(
        reconstruction_summary,
        on="subject",
        how="outer",
        validate="one_to_one",
    )
    summary[["path_nll", "reconstruction_nll"]] = summary[
        ["path_nll", "reconstruction_nll"]
    ].fillna(0.0)
    summary[["n_path_choices", "n_reward_choices"]] = summary[
        ["n_path_choices", "n_reward_choices"]
    ].fillna(0).astype(int)
    summary["reward_nll_scale"] = reward_nll_scale
    summary["scaled_reconstruction_nll"] = (
        reward_nll_scale * summary["reconstruction_nll"]
    )
    summary["total_nll"] = (
        summary["path_nll"] + summary["scaled_reconstruction_nll"]
    )
    summary["fit_condition"] = condition
    summary["fit_likelihood"] = (
        "path_plus_recalled_reward"
        if condition == "planning"
        else "recalled_reward_only"
    )
    summary["lambda"] = lambda_value
    summary["alpha"] = alpha_value
    summary["nll_model_seed"] = model_seed
    summary["weight_file"] = str(weight_path)
    return summary


def predict_probabilities_once(
    fitted_model: tf.keras.Model,
    reward_inputs: np.ndarray,
    time_steps: int,
    num_paths: int,
    num_categories: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one stochastic model draw for every post-fit simulation trial."""
    trial_count = reward_inputs.shape[0]
    reconstruction_probabilities = np.zeros(
        (trial_count, time_steps, num_categories),
        dtype=np.float64,
    )
    action_probabilities = np.zeros(
        (trial_count, num_paths),
        dtype=np.float64,
    )
    mutual_information_costs = np.zeros(trial_count, dtype=np.float64)

    for start in range(0, trial_count, batch_size):
        stop = min(start + batch_size, trial_count)
        model_inputs = tf.convert_to_tensor(
            reward_inputs[start:stop],
            dtype=tf.float32,
        )
        outputs = fitted_model(model_inputs, training=False)
        reconstruction_probabilities[start:stop] = np.asarray(
            outputs[0][:, -1, :, :],
            dtype=np.float64,
        )
        action_probabilities[start:stop] = np.asarray(
            outputs[1],
            dtype=np.float64,
        )
        mutual_information_costs[start:stop] = float(outputs[10])

    return (
        reconstruction_probabilities,
        action_probabilities,
        mutual_information_costs,
    )


def sample_probability_rows(
    probabilities: np.ndarray,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Draw one categorical sample from every row of a probability matrix."""
    probability_rows = np.asarray(probabilities, dtype=np.float64)
    if probability_rows.ndim != 2:
        raise ValueError("Categorical probabilities must be a two-dimensional array.")
    if not np.all(np.isfinite(probability_rows)):
        raise ValueError("Categorical probabilities contain non-finite values.")

    probability_rows = np.clip(probability_rows, 0.0, None)
    row_sums = probability_rows.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Categorical probabilities contain an empty row.")
    probability_rows = probability_rows / row_sums
    cumulative_probabilities = np.cumsum(probability_rows, axis=1)
    draws = random_generator.random((len(probability_rows), 1))
    sampled_indices = np.sum(
        draws > cumulative_probabilities,
        axis=1,
    )
    return np.minimum(sampled_indices, probability_rows.shape[1] - 1)


def calculate_path_reward_matrix(
    config: ModuleType,
    rewards: np.ndarray,
) -> np.ndarray:
    """Calculate every path reward for a matrix of simulated trials."""
    path_node_indices = list(config.index_path_map.values())
    if len(path_node_indices) != config.num_paths:
        raise ValueError(
            "Configured path count does not match index_path_map."
        )
    return np.column_stack(
        [
            rewards[
                :,
                np.asarray(node_indices, dtype=int) - 1,
            ].sum(axis=1)
            for node_indices in path_node_indices
        ]
    )


def simulate_fitted_participants(
    config: ModuleType,
    best_fits: pd.DataFrame,
    model_seeds: list[int],
    trials_per_seed: int,
    batch_size: int,
    simulation_random_seed: int,
    reward_nll_scale: float,
    fit_configuration_hash: str,
    condition: str = "planning",
    weight_path_candidates_by_parameter_seed: dict[
        tuple[float, float, int], tuple[Path, ...]
    ]
    | None = None,
) -> pd.DataFrame:
    """Simulate each fitted participant under every supplied model seed."""
    category_values = np.asarray(
        config.reconstruction_values,
        dtype=np.float64,
    )
    reward_values = (
        np.asarray([0, 1], dtype=int)
        if config.time_steps == 2
        else np.asarray([-4, -3, -2, -1, 1, 2, 3, 4], dtype=int)
    )
    leaf_nodes = set(config.path_leaf_dict.values())
    leaf_flags = np.asarray(
        [node in leaf_nodes for node in range(1, config.time_steps + 1)],
        dtype=bool,
    )
    simulation_frames: list[pd.DataFrame] = []
    next_graph_index = 0
    simulation_participant_order = (
        best_fits["subject"].drop_duplicates().tolist()
    )
    simulation_participant_positions = {
        subject: position
        for position, subject in enumerate(
            simulation_participant_order,
            start=1,
        )
    }
    print(
        "Starting fitted simulations: "
        f"{len(simulation_participant_order)} participants x "
        f"{len(model_seeds)} model seeds x "
        f"{trials_per_seed} trials.",
        flush=True,
    )

    fit_groups = best_fits.groupby(
        ["best_lambda", "best_alpha"],
        sort=False,
        dropna=False,
    )
    for parameter_index, ((lambda_value, alpha_value), fit_group) in enumerate(
        fit_groups
    ):
        participants = fit_group["subject"].to_numpy()
        nll_model_seeds = fit_group["nll_model_seed"].to_numpy(dtype=int)
        participant_by_trial = np.repeat(participants, trials_per_seed)
        nll_seed_by_trial = np.repeat(nll_model_seeds, trials_per_seed)
        trial_within_participant = np.tile(
            np.arange(trials_per_seed, dtype=int),
            len(participants),
        )
        trial_count = len(participant_by_trial)

        reward_seed = np.random.SeedSequence(
            [simulation_random_seed, parameter_index, 0]
        )
        reward_generator = np.random.default_rng(reward_seed)
        trial_rewards = reward_generator.choice(
            reward_values,
            size=(trial_count, config.time_steps),
            replace=True,
        )
        model_inputs = trial_rewards.astype(np.float32)[..., np.newaxis]
        path_rewards = calculate_path_reward_matrix(config, trial_rewards)
        maximum_path_rewards = np.max(path_rewards, axis=1)
        minimum_path_rewards = np.min(path_rewards, axis=1)
        mean_node_path_rewards = np.column_stack(
            [
                path_rewards[
                    :,
                    np.asarray(config.node_path_map[str(node)], dtype=int),
                ].mean(axis=1)
                for node in range(1, config.time_steps + 1)
            ]
        )

        for model_seed_index, model_seed in enumerate(model_seeds, start=1):
            for subject in participants:
                print(
                    "Fitted simulation participant starting | "
                    f"participant={simulation_participant_positions[subject]}/"
                    f"{len(simulation_participant_order)} | "
                    f"subject={subject} | "
                    f"tree_type={config.tree_type} | "
                    f"model seed {model_seed_index}/{len(model_seeds)} "
                    f"(seed={model_seed}) | "
                    f"trials={trials_per_seed} | "
                    f"lambda={float(lambda_value):g} | "
                    f"alpha={float(alpha_value):g}",
                    flush=True,
                )
            tf.keras.backend.clear_session()
            fitted_model, weight_path = build_and_load_model(
                config,
                float(lambda_value),
                float(alpha_value),
                model_seed,
                extra_weight_paths=(
                    weight_path_candidates_by_parameter_seed or {}
                ).get(
                    (
                        float(lambda_value),
                        float(alpha_value),
                        int(model_seed),
                    ),
                    (),
                ),
            )
            stochastic_seed = np.random.SeedSequence(
                [simulation_random_seed, parameter_index, model_seed, 1]
            )
            tensorflow_seed = int(
                stochastic_seed.generate_state(1, dtype=np.uint32)[0]
                % np.iinfo(np.int32).max
            )
            tf.random.set_seed(tensorflow_seed)
            (
                reconstruction_probabilities,
                action_probabilities,
                mutual_information_costs,
            ) = predict_probabilities_once(
                fitted_model,
                model_inputs,
                config.time_steps,
                config.num_paths,
                config.num_categories,
                batch_size,
            )

            response_generator = np.random.default_rng(
                np.random.SeedSequence(
                    [simulation_random_seed, parameter_index, model_seed, 2]
                )
            )
            sampled_category_indices = sample_probability_rows(
                reconstruction_probabilities.reshape(
                    -1,
                    config.num_categories,
                ),
                response_generator,
            ).reshape(trial_count, config.time_steps)
            sampled_rewards = category_values[sampled_category_indices]
            modal_rewards = category_values[
                np.argmax(reconstruction_probabilities, axis=-1)
            ]
            expected_rewards = np.sum(
                reconstruction_probabilities * category_values[None, None, :],
                axis=-1,
            )
            true_category_indices = np.clip(
                np.rint(
                    config.reconstruction_max_reward - trial_rewards
                ).astype(int),
                0,
                config.num_categories - 1,
            )
            true_reward_probabilities = np.take_along_axis(
                reconstruction_probabilities,
                true_category_indices[..., np.newaxis],
                axis=-1,
            )[..., 0]
            sampled_path_indices = sample_probability_rows(
                action_probabilities,
                response_generator,
            )
            chosen_path_rewards = path_rewards[
                np.arange(trial_count),
                sampled_path_indices,
            ]

            graph_indices = np.arange(
                next_graph_index,
                next_graph_index + trial_count,
                dtype=int,
            )
            next_graph_index += trial_count
            nodes = np.arange(1, config.time_steps + 1, dtype=int)
            simulation_frames.append(
                pd.DataFrame(
                    {
                        "subject": np.repeat(
                            participant_by_trial,
                            config.time_steps,
                        ),
                        "best_lambda": float(lambda_value),
                        "best_alpha": float(alpha_value),
                        "best_rho": (
                            float(alpha_value) / float(lambda_value)
                            if float(lambda_value) > 0.0
                            else np.nan
                        ),
                        "nll_model_seed": np.repeat(
                            nll_seed_by_trial,
                            config.time_steps,
                        ),
                        "simulation_model_seed": model_seed,
                        "simulation_random_seed": simulation_random_seed,
                        "simulation_trials_per_seed": trials_per_seed,
                        "reward_nll_scale": reward_nll_scale,
                        "condition": condition,
                        "fit_condition": condition,
                        "model_condition": config.model_condition,
                        "action_loss_enabled": (
                            config.action_loss_enabled
                        ),
                        "participant_fit_configuration_hash": (
                            fit_configuration_hash
                        ),
                        "simulation_trial": np.repeat(
                            trial_within_participant,
                            config.time_steps,
                        ),
                        "graph": np.repeat(graph_indices, config.time_steps),
                        "node": np.tile(nodes, trial_count),
                        "actual_reward": trial_rewards.reshape(-1),
                        "estimated_reward": sampled_rewards.reshape(-1),
                        "modal_estimated_reward": modal_rewards.reshape(-1),
                        "expected_estimated_reward": expected_rewards.reshape(-1),
                        "true_reward_probability": (
                            true_reward_probabilities.reshape(-1)
                        ),
                        "max_path_reward": np.repeat(
                            maximum_path_rewards,
                            config.time_steps,
                        ),
                        "min_path_reward": np.repeat(
                            minimum_path_rewards,
                            config.time_steps,
                        ),
                        "path_reward": mean_node_path_rewards.reshape(-1),
                        "V": np.repeat(chosen_path_rewards, config.time_steps),
                        "MI_cost": np.repeat(
                            mutual_information_costs,
                            config.time_steps,
                        ),
                        "is_leaf": np.tile(leaf_flags, trial_count),
                        "model_weight_file": str(weight_path),
                        "reconstruction_head_type": (
                            config.reconstruction_head_type
                        ),
                        "reconstruction_hidden_units": int(
                            config.reconstruction_hidden_units
                        ),
                        "rnn_units": int(config.rnn_units),
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
                        "reward_normalization_constant": float(
                            config.reward_normalization_constant
                        ),
                        "reward_normalization_scheme": (
                            config.reward_normalization_scheme
                        ),
                    }
                )
            )
            for subject in participants:
                print(
                    "Fitted simulation participant completed | "
                    f"participant={simulation_participant_positions[subject]}/"
                    f"{len(simulation_participant_order)} | "
                    f"subject={subject} | "
                    f"tree_type={config.tree_type} | "
                    f"model seed {model_seed_index}/{len(model_seeds)} "
                    f"(seed={model_seed}) | "
                    f"trials={trials_per_seed}",
                    flush=True,
                )
            del fitted_model
            gc.collect()

    if not simulation_frames:
        raise RuntimeError("No fitted participant simulations were generated.")
    return pd.concat(simulation_frames, ignore_index=True)


def write_csv_atomically(frame: pd.DataFrame, output_path: Path) -> None:
    """Write a CSV through a job-specific temporary file."""
    ensure_directory(output_path.parent)
    run_identifier = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    target_digest = hashlib.sha256(
        output_path.name.encode("utf-8")
    ).hexdigest()[:16]
    safe_run_identifier = "".join(
        character if character.isalnum() or character in "_-" else "_"
        for character in run_identifier
    )
    temporary_path = output_path.parent / (
        f".participant_fit_{target_digest}_{safe_run_identifier}"
        ".csv.in_progress"
    )
    temporary_path.unlink(missing_ok=True)
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)


def run_fitted_plots(
    tree: TreeSpecification,
    grid_path: Path,
    fit_path: Path,
    simulation_path: Path,
    condition: str = "planning",
) -> None:
    """Run fitted-simulation and aggregate-NLL plots after CSV creation."""
    if tree.size != 7:
        print(
            "Skipping fitted plots: the current fitted relationship plot is "
            "defined for the 7-node tree."
        )
        return

    configured_rscript = os.environ.get(
        "WM_PLANNING_COMPARISON_RSCRIPT", ""
    ).strip()
    rscript = configured_rscript or shutil.which("Rscript")
    if (
        rscript is None
        or not Path(rscript).is_file()
        or not os.access(rscript, os.X_OK)
    ):
        raise RuntimeError(
            "Rscript is unavailable. Create the separate R environment with "
            "setup_r_env.sh or pass --skip-fitted-plots."
        )
    plot_script = (
        REPO_ROOT / "analysis" / "tests" / "plot_fitted_participant_simulations.R"
    )
    require_existing_path(plot_script, "Fitted participant plot script")
    subprocess.run(
        [
            rscript,
            str(plot_script),
            tree.label,
            str(grid_path),
            str(fit_path),
            str(simulation_path),
            condition,
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def main() -> None:
    """Fit the requested parameter grid and write centralized summary CSVs."""
    args = parse_args()
    tree = resolve_tree(args.tree)
    lambda_values = parse_number_list(args.lambda_values, "lambda_values")
    parameter_two_name = "rho" if args.rho_grid else "alpha"
    parameter_two_input_values = parse_number_list(
        args.alpha_values,
        f"{parameter_two_name}_values",
    )
    if args.rho_grid:
        if any(value <= 0.0 for value in lambda_values):
            raise ValueError("lambda_values must be positive with --rho.")
        if any(value < 0.0 for value in parameter_two_input_values):
            raise ValueError("rho_values must be non-negative.")
    parameter_cells: list[tuple[int, int, float, float, float]] = []
    for lambda_order, lambda_value in enumerate(lambda_values):
        for parameter_two_order, parameter_two_input in enumerate(
            parameter_two_input_values
        ):
            if args.rho_grid:
                rho_value = parameter_two_input
                alpha_value = alpha_from_lambda_rho(
                    lambda_value,
                    rho_value,
                )
            else:
                alpha_value = parameter_two_input
                rho_value = (
                    alpha_value / lambda_value
                    if lambda_value > 0.0
                    else float("nan")
                )
            parameter_cells.append(
                (
                    lambda_order,
                    parameter_two_order,
                    lambda_value,
                    alpha_value,
                    rho_value,
                )
            )
    if len({(cell[2], cell[3]) for cell in parameter_cells}) != len(
        parameter_cells
    ):
        raise ValueError(
            "The requested grid produces duplicate raw lambda/alpha cells."
        )
    alpha_values = list(dict.fromkeys(cell[3] for cell in parameter_cells))
    model_seeds = parse_seed_list(
        str(args.seed)
        if args.seed is not None
        else (args.seed_values or "1,2,3,4,5")
    )
    config = load_model_config(
        tree,
        lambda_values,
        alpha_values,
        model_seeds[0],
        args.max_action_gradient_scale,
        args.parameter_multiplier,
        args.initial_critic_coefficient,
        args.model_epochs,
        args.nonlinear_scale,
        args.reconstruction_hidden_units,
        args.reconstruction_loss_scope,
        args.no_critic_loss,
        args.final_learning_rate,
        args.late_lr_plateau_start_epoch,
        args.late_lr_plateau_end_epoch,
        args.condition,
        rnn_units=args.rnn_units,
    )

    behavior_path = (
        args.behavior_csv.expanduser()
        if args.behavior_csv is not None
        else default_behavior_path(tree)
    )
    require_existing_path(behavior_path, "Behavioral CSV")
    behavior_frame = pd.read_csv(behavior_path, low_memory=False)
    behavior = prepare_behavioral_trials(
        behavior_frame,
        config,
        condition=args.condition,
    )
    all_participant_order = (
        behavior.trials["subject"].drop_duplicates().tolist()
    )
    seed_assignments = assign_nll_model_seeds(
        behavior,
        model_seeds,
        args.seed_selection_seed,
    )
    if args.subject is not None:
        matching_subjects = [
            subject
            for subject in all_participant_order
            if canonical_subject(subject) == canonical_subject(args.subject)
        ]
        if len(matching_subjects) != 1:
            raise ValueError(
                "Expected exactly one behavioral participant matching "
                f"--subject {args.subject!r}; found {len(matching_subjects)}."
            )
        behavior = subset_behavioral_trials(
            behavior,
            np.asarray(matching_subjects),
        )
        seed_assignments = seed_assignments.loc[
            seed_assignments["subject"].isin(matching_subjects)
        ].copy()
    participant_order = (
        behavior.trials["subject"].drop_duplicates().tolist()
    )
    participant_positions = {
        subject: position
        for position, subject in enumerate(participant_order, start=1)
    }
    parameter_combination_count = len(parameter_cells)
    print(
        "Starting participant grid fit: "
        f"{len(participant_order)} participants x "
        f"{parameter_combination_count} parameter combinations.",
        flush=True,
    )
    behavior_by_seed = {
        int(model_seed): subset_behavioral_trials(
            behavior,
            assignment_group["subject"].to_numpy(),
        )
        for model_seed, assignment_group in seed_assignments.groupby(
            "nll_model_seed",
            sort=False,
        )
    }

    grid_results: list[pd.DataFrame] = []
    for (
        lambda_order,
        alpha_order,
        lambda_value,
        alpha_value,
        rho_value,
    ) in parameter_cells:
        parameter_combination_index = alpha_order + (
            lambda_order * len(parameter_two_input_values)
        ) + 1
        for model_seed, seed_behavior in behavior_by_seed.items():
            group_subjects = (
                seed_behavior.trials["subject"]
                .drop_duplicates()
                .tolist()
            )
            for subject in group_subjects:
                print(
                    "Fitting subject "
                    f"{subject} "
                    f"({participant_positions[subject]}/"
                    f"{len(participant_order)}) | "
                    "parameter combination "
                    f"{parameter_combination_index}/"
                    f"{parameter_combination_count} | "
                    f"lambda={lambda_value:g} | "
                    f"alpha={alpha_value:g} | "
                    f"rho={rho_value:g} | "
                    f"model_seed={model_seed}",
                    flush=True,
                )
            tf.keras.backend.clear_session()
            fitted_model, weight_path = build_and_load_model(
                config,
                lambda_value,
                alpha_value,
                model_seed,
            )
            action_policy, reconstruction_policy = estimate_policies(
                fitted_model=fitted_model,
                reward_inputs=seed_behavior.reward_inputs,
                time_steps=config.time_steps,
                num_paths=config.num_paths,
                num_categories=config.num_categories,
                mc_samples=args.mc_samples,
                batch_size=args.batch_size,
                evaluation_seed=args.evaluation_seed,
            )
            model_results = participant_nll(
                seed_behavior,
                action_policy,
                reconstruction_policy,
                lambda_value,
                alpha_value,
                model_seed,
                weight_path,
                args.reward_nll_scale,
                args.condition,
            )
            for result_row in model_results.itertuples(index=False):
                print(
                    "Finished subject "
                    f"{result_row.subject} "
                    f"({participant_positions[result_row.subject]}/"
                    f"{len(participant_order)}) | "
                    "parameter combination "
                    f"{parameter_combination_index}/"
                    f"{parameter_combination_count} | "
                    f"lambda={lambda_value:g} | "
                    f"alpha={alpha_value:g} | "
                    f"rho={rho_value:g} | "
                    f"model_seed={model_seed} | "
                    f"total_nll={result_row.total_nll:.6f}",
                    flush=True,
                )
            model_results["lambda_order"] = lambda_order
            model_results["alpha_order"] = alpha_order
            model_results["rho"] = rho_value
            model_results["parameter_grid_mode"] = parameter_two_name
            grid_results.append(model_results)

            del fitted_model, action_policy, reconstruction_policy
            gc.collect()

    full_grid = pd.concat(grid_results, ignore_index=True)
    full_grid["tree"] = tree.label
    full_grid["fit_condition"] = args.condition
    full_grid["model_condition"] = config.model_condition
    full_grid["action_loss_enabled"] = config.action_loss_enabled
    full_grid["candidate_model_seeds"] = ",".join(
        str(seed) for seed in model_seeds
    )
    full_grid["seed_selection_seed"] = args.seed_selection_seed
    full_grid["model_epochs"] = args.model_epochs
    full_grid["reconstruction_head_type"] = (
        config.reconstruction_head_type
    )
    full_grid["reconstruction_hidden_units"] = (
        config.reconstruction_hidden_units
    )
    full_grid["rnn_units"] = config.rnn_units
    full_grid["reconstruction_loss_scope"] = (
        config.reconstruction_loss_scope
    )
    full_grid["reconstruction_min_reward"] = (
        config.reconstruction_min_reward
    )
    full_grid["reconstruction_max_reward"] = (
        config.reconstruction_max_reward
    )
    full_grid["reconstruction_category_count"] = config.num_categories
    full_grid["reconstruction_architecture_tagged"] = (
        config.reconstruction_architecture_tagged
    )
    full_grid["reward_normalization_constant"] = (
        config.reward_normalization_constant
    )
    full_grid["reward_normalization_scheme"] = (
        config.reward_normalization_scheme
    )
    full_grid["critic_loss_enabled"] = config.critic_loss_enabled
    full_grid["max_action_gradient_scale"] = (
        args.max_action_gradient_scale
    )
    full_grid["parameter_multiplier"] = (
        config.parameter_multiplier
        if config.parameter_multiplier is not None
        else "auto"
    )
    full_grid["initial_critic_coefficient"] = (
        config.initial_critic_coefficient
        if config.initial_critic_coefficient is not None
        else "auto"
    )
    full_grid["final_learning_rate"] = args.final_learning_rate
    full_grid["late_lr_plateau_start_epoch"] = (
        args.late_lr_plateau_start_epoch
    )
    full_grid["late_lr_plateau_end_epoch"] = args.late_lr_plateau_end_epoch
    full_grid["mc_samples"] = args.mc_samples
    full_grid["evaluation_seed"] = args.evaluation_seed
    artifact_suffix = model_configuration_artifact_suffix(
        config,
        tree,
        lambda_values[0],
        alpha_values[0],
        model_seeds[0],
    )
    full_grid["model_configuration_artifact_suffix"] = artifact_suffix
    seed_label = "-".join(str(seed) for seed in model_seeds)
    parameter_grid_suffix = "_rho_grid" if args.rho_grid else ""
    full_fit_run_suffix = (
        f"_seeds_{seed_label}_sel_{args.seed_selection_seed}"
        f"_mc{args.mc_samples}_eval{args.evaluation_seed}"
        f"_reward_nll_scale_{args.reward_nll_scale:g}{artifact_suffix}"
        f"{parameter_grid_suffix}"
    )
    fit_run_suffix, fit_configuration_hash = (
        compact_participant_fit_run_suffix(full_fit_run_suffix)
    )
    full_grid["participant_fit_configuration_hash"] = (
        fit_configuration_hash
    )
    full_grid["participant_fit_artifact_suffix"] = fit_run_suffix
    full_grid = full_grid.sort_values(
        ["subject", "lambda_order", "alpha_order"],
        kind="stable",
    ).reset_index(drop=True)

    best_fits = (
        full_grid.sort_values(
            ["subject", "total_nll", "lambda_order", "alpha_order"],
            kind="stable",
        )
        .groupby("subject", as_index=False, sort=False)
        .first()
        .rename(
            columns={
                "lambda": "best_lambda",
                "alpha": "best_alpha",
                "rho": "best_rho",
            }
        )
    )

    output_directory = ensure_directory(SUMMARY_DIR / "participant_fits")
    safe_tree = "".join(
        character if character.isalnum() or character in "_-" else "_"
        for character in tree.label
    )
    if args.shard_directory is None:
        grid_path = output_directory / (
            f"participant_parameter_grid_{safe_tree}{fit_run_suffix}.csv"
        )
        fit_path = output_directory / (
            f"participant_parameter_fits_{safe_tree}{fit_run_suffix}.csv"
        )
    else:
        if args.subject is None:
            raise ValueError("--shard-directory requires --subject.")
        shard_directory = ensure_directory(args.shard_directory.expanduser())
        subject_token = safe_subject_token(
            best_fits["subject"].drop_duplicates().iloc[0]
        )
        grid_path = shard_directory / f"grid_subject_{subject_token}.csv"
        fit_path = shard_directory / f"fit_subject_{subject_token}.csv"
    write_csv_atomically(
        full_grid.drop(columns=["lambda_order", "alpha_order"]),
        grid_path,
    )
    write_csv_atomically(
        best_fits.drop(columns=["lambda_order", "alpha_order"]),
        fit_path,
    )

    if args.shard_directory is not None:
        print(
            "Completed participant fit shard | "
            f"subject={canonical_subject(best_fits['subject'].iloc[0])} | "
            f"grid={grid_path} | fit={fit_path}",
            flush=True,
        )
        return

    simulation_path = ensure_directory(
        SIMULATION_DIR / "participant_fits"
    ) / (
        f"participant_fitted_simulations_{safe_tree}{fit_run_suffix}"
        f"_n{args.simulation_trials_per_seed}"
        f"_simseed{args.simulation_random_seed}.csv"
    )
    fitted_simulations = simulate_fitted_participants(
        config,
        best_fits,
        model_seeds,
        args.simulation_trials_per_seed,
        args.batch_size,
        args.simulation_random_seed,
        args.reward_nll_scale,
        fit_configuration_hash,
        args.condition,
    )
    write_csv_atomically(fitted_simulations, simulation_path)

    if not args.skip_fitted_plots:
        run_fitted_plots(
            tree,
            grid_path,
            fit_path,
            simulation_path,
            args.condition,
        )

    print(f"Behavioral input: {behavior_path}")
    print(f"Fit condition: {args.condition}")
    print(f"Participants: {full_grid['subject'].nunique()}")
    print(f"Candidate model seeds: {','.join(str(seed) for seed in model_seeds)}")
    print(f"Seed-selection random seed: {args.seed_selection_seed}")
    print(f"Reward-NLL scale: {args.reward_nll_scale:g}")
    print(f"Parameter-grid mode: {parameter_two_name}")
    print(f"RNN units: {args.rnn_units}")
    print(f"Parameter combinations: {len(parameter_cells)}")
    print_local_scp_command(
        (grid_path, fit_path, simulation_path),
        ".local_artifacts/participant_fit_audit",
    )


if __name__ == "__main__":
    main()
