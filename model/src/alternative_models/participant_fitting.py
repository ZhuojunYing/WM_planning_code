"""Participant grid fitting shared by the alternative model families."""

from __future__ import annotations

import argparse
import gc
import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Literal

import numpy as np
import pandas as pd
import tensorflow as tf

from project_paths import (
    SIMULATION_DIR,
    SUMMARY_DIR,
    WEIGHT_DIR,
    ensure_directory,
    print_local_scp_command,
    require_existing_path,
)
from ..main_vae import fit_participant_parameters as shared


ModelFamily = Literal[
    "independent_channel",
    "simultaneous_channel",
    "bays_value",
    "earlier_presentation",
    "later_presentation",
]


@dataclass(frozen=True)
class AlternativeModelSpecification:
    """Names and fitting behavior for one alternative architecture."""

    family: ModelFamily
    artifact_tag: str
    parameter_one_name: str
    parameter_two_name: str
    uses_trained_weights: bool
    stochastic_policy: bool


MODEL_SPECIFICATIONS = {
    "independent_channel": AlternativeModelSpecification(
        family="independent_channel",
        artifact_tag="independent_channels",
        parameter_one_name="lambda",
        parameter_two_name="alpha",
        uses_trained_weights=True,
        stochastic_policy=True,
    ),
    "simultaneous_channel": AlternativeModelSpecification(
        family="simultaneous_channel",
        artifact_tag="simultaneous_channel",
        parameter_one_name="lambda",
        parameter_two_name="alpha",
        uses_trained_weights=True,
        stochastic_policy=True,
    ),
    "bays_value": AlternativeModelSpecification(
        family="bays_value",
        artifact_tag="bays_reward_fixed_resource",
        parameter_one_name="resource_budget",
        parameter_two_name="precision_exponent",
        uses_trained_weights=False,
        stochastic_policy=False,
    ),
    "earlier_presentation": AlternativeModelSpecification(
        family="earlier_presentation",
        artifact_tag="earlier_presentation_resource",
        parameter_one_name="priority_one_resource",
        parameter_two_name="precision_exponent",
        uses_trained_weights=False,
        stochastic_policy=False,
    ),
    "later_presentation": AlternativeModelSpecification(
        family="later_presentation",
        artifact_tag="later_presentation_resource",
        parameter_one_name="priority_one_resource",
        parameter_two_name="precision_exponent",
        uses_trained_weights=False,
        stochastic_policy=False,
    ),
}


def parse_args(
    specification: AlternativeModelSpecification,
) -> argparse.Namespace:
    """Parse a model-family-specific participant fitting request."""
    parser = argparse.ArgumentParser(
        description=(
            f"Fit {specification.parameter_one_name} and "
            f"{specification.parameter_two_name} for the "
            f"{specification.family} model."
        )
    )
    parser.add_argument("tree")
    parser.add_argument(
        "parameter_one_values",
        help=f"Comma-separated {specification.parameter_one_name} values.",
    )
    parser.add_argument(
        "parameter_two_values",
        help=(
            f"Comma-separated {specification.parameter_two_name} values. "
            "For independent/simultaneous channels, --rho instead interprets "
            "these as rho = alpha / lambda."
        ),
    )
    parser.add_argument(
        "--rho",
        dest="rho_grid",
        action="store_true",
        help=(
            "For independent/simultaneous-channel fits, interpret the second parameter "
            "list as rho = alpha / lambda and fit alpha = lambda * rho."
        ),
    )
    parser.add_argument("--seed-values", default="1,2,3,4,5")
    parser.add_argument("--seed-selection-seed", type=int, default=2026)
    parser.add_argument("--mc-samples", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--evaluation-seed", type=int, default=2026)
    parser.add_argument("--reward-nll-scale", type=float, default=0.5)
    parser.add_argument(
        "--simulation-trials-per-seed",
        type=int,
        default=100,
    )
    parser.add_argument("--simulation-random-seed", type=int, default=2027)
    parser.add_argument("--behavior-csv", type=Path, default=None)
    parser.add_argument("--subject", default=None,
                        help="Fit one subject, retaining full-cohort seed assignment.")
    parser.add_argument("--skip-fitted-plots", action="store_true")
    parser.add_argument("--participant-index", type=int, default=None)
    parser.add_argument("--shard-directory", type=Path, default=None)
    parser.add_argument("--aggregate-shards", action="store_true")

    parser.add_argument("--model-epochs", type=int, default=320)
    parser.add_argument("--max-action-gradient-scale", type=float, default=1.0)
    parser.add_argument("--parameter-multiplier", type=float, default=None)
    critic_group = parser.add_mutually_exclusive_group()
    critic_group.add_argument(
        "--critic-loss",
        dest="critic_loss_enabled",
        action="store_true",
    )
    critic_group.add_argument(
        "--no-critic-loss",
        dest="critic_loss_enabled",
        action="store_false",
    )
    parser.set_defaults(
        critic_loss_enabled=(
            specification.family == "simultaneous_channel"
        )
    )
    parser.add_argument(
        "--initial-critic-coefficient",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--channel-kl-aggregation",
        choices=("sum", "mean"),
        default="sum",
    )
    parser.add_argument(
        "--channel-kl-time-weighting",
        choices=("remaining_steps", "uniform"),
        default="remaining_steps",
    )
    head_group = parser.add_mutually_exclusive_group()
    head_group.add_argument(
        "--nonlinear-scale",
        dest="reconstruction_head_type",
        action="store_const",
        const="nonlinear_location_scale",
    )
    head_group.add_argument(
        "--nonlinear-categorical",
        dest="reconstruction_head_type",
        action="store_const",
        const="nonlinear_categorical",
    )
    parser.set_defaults(reconstruction_head_type="nonlinear_location_scale")
    parser.add_argument("--reconstruction-hidden-units", type=int, default=128)
    parser.add_argument("--rnn-units", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--final-learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--late-lr-plateau-start-epoch",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--late-lr-plateau-end-epoch",
        type=int,
        default=300,
    )
    parser.add_argument("--no-late-lr-plateau", action="store_true")
    args = parser.parse_args()

    for name in (
        "seed_selection_seed",
        "evaluation_seed",
        "simulation_random_seed",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative.")
    for name in (
        "mc_samples",
        "batch_size",
        "simulation_trials_per_seed",
        "model_epochs",
        "reconstruction_hidden_units",
        "rnn_units",
        "latent_dim",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if not math.isfinite(args.reward_nll_scale) or args.reward_nll_scale <= 0:
        parser.error("--reward-nll-scale must be finite and positive.")
    if (
        not math.isfinite(args.max_action_gradient_scale)
        or args.max_action_gradient_scale < 1
    ):
        parser.error(
            "--max-action-gradient-scale must be finite and at least 1."
        )
    if args.parameter_multiplier is not None and (
        not math.isfinite(args.parameter_multiplier)
        or args.parameter_multiplier <= 0
    ):
        parser.error("--parameter-multiplier must be finite and positive.")
    if args.initial_critic_coefficient is not None and (
        not math.isfinite(args.initial_critic_coefficient)
        or args.initial_critic_coefficient < 0
    ):
        parser.error(
            "--initial-critic-coefficient must be finite and non-negative."
        )
    if (
        specification.family != "simultaneous_channel"
        and args.critic_loss_enabled
    ):
        parser.error(
            "Only the simultaneous-channel alternative supports critic loss."
        )
    if (
        not args.critic_loss_enabled
        and args.initial_critic_coefficient is not None
    ):
        parser.error(
            "--initial-critic-coefficient requires critic loss to be enabled."
        )
    if args.rho_grid and specification.family not in {"simultaneous_channel", "independent_channel"}:
        parser.error("--rho is supported only for independent/simultaneous-channel fits.")
    if (
        specification.family == "simultaneous_channel"
        and args.latent_dim != 16
    ):
        parser.error(
            "The simultaneous-channel architecture fixes --latent-dim at 16."
        )
    if (
        not math.isfinite(args.final_learning_rate)
        or args.final_learning_rate <= 0
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
    if args.participant_index is not None or args.aggregate_shards or args.shard_directory is not None:
        if specification.family not in {"simultaneous_channel", "independent_channel"} or args.shard_directory is None:
            parser.error("Parallel shards require independent/simultaneous channels and --shard-directory.")
        if args.aggregate_shards == (args.participant_index is not None):
            parser.error("Select exactly one of --participant-index or --aggregate-shards.")
        if args.participant_index is not None and args.participant_index < 0:
            parser.error("--participant-index must be non-negative.")
    return args


def load_config(
    tree: shared.TreeSpecification,
    parameter_one_values: list[float],
    parameter_two_values: list[float],
    model_seeds: list[int],
    args: argparse.Namespace,
) -> ModuleType:
    """Load the centralized tree and artifact configuration once."""
    original_arguments = sys.argv[:]
    sys.argv = [
        "alternative_participant_fit",
        ",".join(str(value) for value in parameter_one_values),
        ",".join(str(value) for value in parameter_two_values),
        "default",
        "default",
        str(args.model_epochs),
        str(model_seeds[0]),
        str(tree.size),
        "simulate",
        tree.tree_type,
        "1",
        "--max-action-gradient-scale",
        str(args.max_action_gradient_scale),
        "--reconstruction-hidden-units",
        str(args.reconstruction_hidden_units),
        "--final-learning-rate",
        str(args.final_learning_rate),
        "--channel-kl-aggregation",
        args.channel_kl_aggregation,
        "--channel-kl-time-weighting",
        args.channel_kl_time_weighting,
        "--rnn-units",
        str(args.rnn_units),
        "--latent-dim",
        str(args.latent_dim),
        (
            "--nonlinear-scale"
            if args.reconstruction_head_type == "nonlinear_location_scale"
            else "--nonlinear-categorical"
        ),
    ]
    if not args.critic_loss_enabled:
        sys.argv.append("--no-critic-loss")
    if args.initial_critic_coefficient is not None:
        sys.argv.extend(
            [
                "--initial-critic-coefficient",
                str(args.initial_critic_coefficient),
            ]
        )
    if args.late_lr_plateau_start_epoch is not None:
        sys.argv.extend(
            [
                "--late-lr-plateau-start-epoch",
                str(args.late_lr_plateau_start_epoch),
                "--late-lr-plateau-end-epoch",
                str(args.late_lr_plateau_end_epoch),
            ]
        )
    if args.parameter_multiplier is not None:
        sys.argv.extend(
            ["--parameter-multiplier", str(args.parameter_multiplier)]
        )
    try:
        return importlib.import_module("model.src.main_vae.config")
    finally:
        sys.argv = original_arguments


def validate_parameter_grid(
    specification: AlternativeModelSpecification,
    parameter_one_values: list[float],
    parameter_two_values: list[float],
) -> None:
    """Apply model-family parameter-domain constraints."""
    if specification.uses_trained_weights:
        if any(value < 0 for value in parameter_one_values):
            raise ValueError(
                f"{specification.family} lambda values must be non-negative."
            )
        if any(value < 0 for value in parameter_two_values):
            raise ValueError(
                f"{specification.family} alpha values must be non-negative."
            )
        return
    if any(value <= 0 for value in parameter_one_values):
        raise ValueError(
            f"{specification.parameter_one_name} values must be positive."
        )
    if any(value <= 0 for value in parameter_two_values):
        raise ValueError("precision_exponent values must be positive.")


def alpha_from_lambda_rho(lambda_value: float, rho_value: float) -> float:
    """Return a stable raw-alpha value for one rho-parameterized cell."""
    alpha_value = lambda_value * rho_value
    if not math.isfinite(alpha_value):
        raise ValueError(
            "Derived alpha is not finite for "
            f"lambda={lambda_value:g}, rho={rho_value:g}."
        )
    return float(f"{alpha_value:.12g}")


def build_model(
    specification: AlternativeModelSpecification,
    config: ModuleType,
    parameter_one: float,
    parameter_two: float,
    model_seed: int,
) -> tuple[tf.keras.Model, str]:
    """Construct one alternative model and load weights when applicable."""
    tf.random.set_seed(model_seed)
    if specification.family == "independent_channel":
        model_main = importlib.import_module(
            "model.src.alternative_models.independent_channel.main"
        )
        fitted_model = model_main.build_independent_channel_model(
            parameter_one,
            parameter_two,
        )
        artifact_name = model_main.independent_channel_artifact_name(
            parameter_one,
            parameter_two,
            model_seed,
        )
        weight_path = WEIGHT_DIR / f"{artifact_name}.weights.h5"
        require_existing_path(weight_path, "Independent-channel weight file")
        fitted_model(
            tf.zeros((1, config.time_steps, 1), dtype=tf.float32),
            training=False,
        )
        fitted_model.load_weights(str(weight_path))
        return fitted_model, str(weight_path)

    if specification.family == "simultaneous_channel":
        model_main = importlib.import_module(
            "model.src.alternative_models.simultaneous_channel.main"
        )
        fitted_model = model_main.build_simultaneous_channel_model(
            parameter_one,
            parameter_two,
        )
        artifact_name = model_main.simultaneous_channel_artifact_name(
            parameter_one,
            parameter_two,
            model_seed,
        )
        weight_path = WEIGHT_DIR / f"{artifact_name}.weights.h5"
        require_existing_path(weight_path, "Simultaneous-channel weight file")
        fitted_model(
            tf.zeros((1, config.time_steps, 1), dtype=tf.float32),
            training=False,
        )
        fitted_model.load_weights(str(weight_path))
        return fitted_model, str(weight_path)

    if specification.family == "bays_value":
        model_main = importlib.import_module(
            "model.src.alternative_models.bays_value.main"
        )
        fitted_model = model_main.build_bays_reward_resource_model(
            parameter_one,
            parameter_two,
        )
        artifact_name = model_main.bays_reward_resource_artifact_name(
            parameter_one,
            parameter_two,
            model_seed,
        )
    elif specification.family in {
        "earlier_presentation",
        "later_presentation",
    }:
        order_main = importlib.import_module(
            "model.src.alternative_models.presentation_order.main"
        )
        presentation_direction = (
            "earlier"
            if specification.family == "earlier_presentation"
            else "later"
        )
        order_model = importlib.import_module(
            "model.src.alternative_models.presentation_order."
            f"{presentation_direction}_model"
        )
        model_class = getattr(
            order_model,
            (
                "EarlierPresentationResourceModel"
                if presentation_direction == "earlier"
                else "LaterPresentationResourceModel"
            ),
        )
        fitted_model = model_class(
            priority_one_resource=parameter_one,
            precision_exponent=parameter_two,
            time_steps=config.time_steps,
            num_paths=config.num_paths,
            index_path_map=config.index_path_map,
        )
        artifact_name = order_main._artifact_name(
            presentation_direction,
            parameter_one,
            parameter_two,
            model_seed,
        )
    else:
        raise ValueError(
            f"Unsupported alternative model family: {specification.family}"
        )

    fitted_model(
        tf.zeros((1, config.time_steps, 1), dtype=tf.float32),
        training=False,
    )
    return fitted_model, f"analytic:{artifact_name}"


def participant_nll(
    behavior: shared.BehavioralTrials,
    action_probabilities: np.ndarray,
    reconstruction_probabilities: np.ndarray,
    specification: AlternativeModelSpecification,
    parameter_one: float,
    parameter_two: float,
    model_seed: int,
    model_artifact: str,
    reward_nll_scale: float,
) -> pd.DataFrame:
    """Calculate participant path and recalled-reward NLL components."""
    epsilon = np.finfo(np.float64).tiny
    trials = behavior.trials.copy()
    trial_indices = trials["trial_index"].to_numpy(dtype=int)
    path_indices = trials["path_index"].to_numpy(dtype=int)
    selected_path_probabilities = action_probabilities[
        trial_indices,
        path_indices,
    ]
    trials["path_nll"] = -np.log(
        np.clip(selected_path_probabilities, epsilon, 1.0)
    )
    path_summary = trials.groupby("subject", as_index=False).agg(
        path_nll=("path_nll", "sum"),
        n_path_choices=("path_nll", "size"),
    )

    recalls = behavior.recalls.copy()
    recall_probabilities = reconstruction_probabilities[
        recalls["trial_index"].to_numpy(dtype=int),
        recalls["node"].to_numpy(dtype=int) - 1,
        recalls["category_index"].to_numpy(dtype=int),
    ]
    recalls["reconstruction_nll"] = -np.log(
        np.clip(recall_probabilities, epsilon, 1.0)
    )
    reward_summary = recalls.groupby("subject", as_index=False).agg(
        reconstruction_nll=("reconstruction_nll", "sum"),
        n_reward_choices=("reconstruction_nll", "size"),
    )
    summary = path_summary.merge(
        reward_summary,
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
    # Lambda/alpha aliases keep the established fitted-plot interface usable.
    summary["lambda"] = parameter_one
    summary["alpha"] = parameter_two
    summary[specification.parameter_one_name] = parameter_one
    summary[specification.parameter_two_name] = parameter_two
    summary["parameter_one_name"] = specification.parameter_one_name
    summary["parameter_two_name"] = specification.parameter_two_name
    summary["nll_model_seed"] = model_seed
    summary["model_artifact"] = model_artifact
    summary["model_family"] = specification.family
    return summary


def _path_nll_tail_share(
    path_nll: np.ndarray,
    fraction: float,
) -> float:
    """Return the fraction of NLL contributed by the largest losses."""
    if path_nll.size == 0:
        return float("nan")
    total = float(np.sum(path_nll))
    if total <= 0.0:
        return 0.0
    tail_count = max(1, int(math.ceil(path_nll.size * fraction)))
    largest_losses = np.partition(
        path_nll,
        path_nll.size - tail_count,
    )[-tail_count:]
    return float(np.sum(largest_losses) / total)


def _summarize_path_diagnostic_rows(
    diagnostics: pd.DataFrame,
) -> dict[str, float | int | bool]:
    """Summarize exact-choice probability and reward diagnostics."""
    selected_probabilities = diagnostics[
        "chosen_path_probability"
    ].to_numpy(dtype=np.float64)
    path_nll = diagnostics["path_nll"].to_numpy(dtype=np.float64)
    choice_count = int(len(diagnostics))
    result: dict[str, float | int | bool] = {
        "path_diagnostic_choice_count": choice_count,
        "path_diagnostic_path_nll_total": float(np.sum(path_nll)),
        "path_diagnostic_chosen_probability_sum": float(
            np.sum(selected_probabilities)
        ),
        "path_diagnostic_chosen_probability_mean": float(
            np.mean(selected_probabilities)
        ),
        "path_diagnostic_chosen_probability_geometric_mean": float(
            np.exp(-np.mean(path_nll))
        ),
        "path_diagnostic_chosen_probability_min": float(
            np.min(selected_probabilities)
        ),
        "path_diagnostic_chosen_probability_p01": float(
            np.quantile(selected_probabilities, 0.01)
        ),
        "path_diagnostic_chosen_probability_p05": float(
            np.quantile(selected_probabilities, 0.05)
        ),
        "path_diagnostic_chosen_probability_median": float(
            np.median(selected_probabilities)
        ),
        "path_diagnostic_path_nll_mean_per_choice": float(
            np.mean(path_nll)
        ),
        "path_diagnostic_path_nll_top_1pct_share": (
            _path_nll_tail_share(path_nll, 0.01)
        ),
        "path_diagnostic_path_nll_top_5pct_share": (
            _path_nll_tail_share(path_nll, 0.05)
        ),
        "path_diagnostic_policy_confidence_sum": float(
            diagnostics["action_policy_confidence"].sum()
        ),
        "path_diagnostic_policy_confidence_mean": float(
            diagnostics["action_policy_confidence"].mean()
        ),
        "path_diagnostic_policy_normalized_entropy_sum": float(
            diagnostics["action_policy_normalized_entropy"].sum()
        ),
        "path_diagnostic_policy_normalized_entropy_mean": float(
            diagnostics["action_policy_normalized_entropy"].mean()
        ),
        "path_diagnostic_expected_normalized_reward_sum": float(
            diagnostics["action_policy_expected_normalized_reward"].sum()
        ),
        "path_diagnostic_expected_normalized_reward_mean": float(
            diagnostics["action_policy_expected_normalized_reward"].mean()
        ),
        "path_diagnostic_optimal_path_probability_sum": float(
            diagnostics["action_policy_optimal_path_probability"].sum()
        ),
        "path_diagnostic_optimal_path_probability_mean": float(
            diagnostics["action_policy_optimal_path_probability"].mean()
        ),
        "path_diagnostic_participant_normalized_reward_sum": float(
            diagnostics["participant_chosen_path_normalized_reward"].sum()
        ),
        "path_diagnostic_participant_normalized_reward_mean": float(
            diagnostics["participant_chosen_path_normalized_reward"].mean()
        ),
        "path_diagnostic_modal_choice_match_count": int(
            diagnostics["model_modal_path_matches_participant"].sum()
        ),
        "path_diagnostic_modal_choice_match_rate": float(
            diagnostics["model_modal_path_matches_participant"].mean()
        ),
        "path_diagnostic_participant_optimal_choice_count": int(
            diagnostics["participant_chosen_path_is_optimal"].sum()
        ),
        "path_diagnostic_participant_optimal_choice_rate": float(
            diagnostics["participant_chosen_path_is_optimal"].mean()
        ),
        "path_diagnostic_chosen_probability_rank_sum": float(
            diagnostics["chosen_path_probability_rank"].sum()
        ),
        "path_diagnostic_chosen_probability_rank_mean": float(
            diagnostics["chosen_path_probability_rank"].mean()
        ),
        "path_diagnostic_probability_values_test_passed": True,
        "path_diagnostic_probability_row_sum_test_passed": True,
        "path_diagnostic_path_nll_identity_test_passed": True,
        "path_diagnostic_normalized_reward_bounds_test_passed": True,
    }
    for threshold_label, threshold in (
        ("1e_12", 1e-12),
        ("1e_09", 1e-9),
        ("1e_06", 1e-6),
        ("1e_04", 1e-4),
        ("1e_03", 1e-3),
    ):
        below_threshold = selected_probabilities < threshold
        result[
            f"path_diagnostic_chosen_probability_below_{threshold_label}_count"
        ] = int(np.sum(below_threshold))
        result[
            f"path_diagnostic_chosen_probability_below_{threshold_label}_fraction"
        ] = float(np.mean(below_threshold))
        result[
            f"path_diagnostic_path_nll_from_below_{threshold_label}"
        ] = float(np.sum(path_nll[below_threshold]))
    return result


def simultaneous_path_policy_diagnostics(
    behavior: shared.BehavioralTrials,
    action_probabilities: np.ndarray,
    config: ModuleType,
    trial_records: list[pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int | bool]]:
    """Test and summarize why similar rewards can have different path NLL."""
    probabilities = np.asarray(action_probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError(
            "Path diagnostic expected a two-dimensional action policy."
        )
    if probabilities.shape != (
        behavior.reward_inputs.shape[0],
        config.num_paths,
    ):
        raise ValueError(
            "Path diagnostic action-policy dimensions do not match trials."
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Path diagnostic found non-finite probabilities.")
    if np.any(probabilities < -1e-8) or np.any(probabilities > 1.0 + 1e-8):
        raise ValueError("Path diagnostic found probabilities outside [0, 1].")
    probability_row_sums = probabilities.sum(axis=1)
    if not np.allclose(probability_row_sums, 1.0, rtol=0.0, atol=1e-5):
        maximum_error = float(np.max(np.abs(probability_row_sums - 1.0)))
        raise ValueError(
            "Path diagnostic action probabilities do not sum to one; "
            f"maximum error={maximum_error:.8g}."
        )

    trials = behavior.trials.copy()
    if trials["trial_index"].duplicated().any():
        raise ValueError(
            "Path diagnostic requires exactly one path choice per trial."
        )
    trial_indices = trials["trial_index"].to_numpy(dtype=int)
    path_indices = trials["path_index"].to_numpy(dtype=int)
    if (
        np.any(trial_indices < 0)
        or np.any(trial_indices >= len(probabilities))
        or np.any(path_indices < 0)
        or np.any(path_indices >= config.num_paths)
    ):
        raise ValueError("Path diagnostic found an invalid trial or path index.")

    trial_probabilities = probabilities[trial_indices]
    # Preserve the raw selected probabilities for an exact identity with the
    # fitter's path NLL.  Normalize a separate copy for bounded summaries:
    # averaging float32 softmax policies can leave row sums a few ULPs from 1,
    # which is harmless for fitting but can make tied-optimum mass 1.00000x.
    diagnostic_probabilities = trial_probabilities / np.sum(
        trial_probabilities,
        axis=1,
        keepdims=True,
    )
    row_indices = np.arange(len(trials))
    chosen_probabilities = trial_probabilities[row_indices, path_indices]
    diagnostic_chosen_probabilities = diagnostic_probabilities[
        row_indices,
        path_indices,
    ]
    epsilon = np.finfo(np.float64).tiny
    path_nll = -np.log(np.clip(chosen_probabilities, epsilon, 1.0))

    reward_inputs = np.asarray(
        behavior.reward_inputs[trial_indices],
        dtype=np.float64,
    )
    if reward_inputs.ndim != 3 or reward_inputs.shape[-1] != 1:
        raise ValueError(
            "Path diagnostic reward inputs must have shape [trial, node, 1]."
        )
    path_rewards = shared.calculate_path_reward_matrix(
        config,
        reward_inputs[..., 0],
    )
    minimum_rewards = np.min(path_rewards, axis=1, keepdims=True)
    maximum_rewards = np.max(path_rewards, axis=1, keepdims=True)
    normalized_path_rewards = (
        (path_rewards - minimum_rewards)
        / (maximum_rewards - minimum_rewards + 1e-6)
    )
    expected_normalized_reward = np.sum(
        diagnostic_probabilities * normalized_path_rewards,
        axis=1,
    )
    chosen_normalized_reward = normalized_path_rewards[
        row_indices,
        path_indices,
    ]
    optimal_paths = np.isclose(
        path_rewards,
        maximum_rewards,
        rtol=0.0,
        atol=1e-9,
    )
    optimal_path_probability = np.sum(
        diagnostic_probabilities * optimal_paths,
        axis=1,
    )
    participant_choice_is_optimal = optimal_paths[
        row_indices,
        path_indices,
    ]
    policy_confidence = np.max(diagnostic_probabilities, axis=1)
    modal_choice_matches = np.isclose(
        diagnostic_chosen_probabilities,
        policy_confidence,
        rtol=0.0,
        atol=1e-12,
    )
    chosen_probability_rank = 1 + np.sum(
        diagnostic_probabilities
        > diagnostic_chosen_probabilities[:, np.newaxis] + 1e-12,
        axis=1,
    )
    safe_probabilities = np.clip(diagnostic_probabilities, epsilon, 1.0)
    policy_entropy = -np.sum(
        diagnostic_probabilities * np.log(safe_probabilities),
        axis=1,
    )
    normalized_entropy = (
        policy_entropy / math.log(config.num_paths)
        if config.num_paths > 1
        else np.zeros_like(policy_entropy)
    )

    for diagnostic_name, values in (
        ("expected normalized reward", expected_normalized_reward),
        ("chosen normalized reward", chosen_normalized_reward),
        ("optimal-path probability", optimal_path_probability),
        ("normalized policy entropy", normalized_entropy),
    ):
        if np.any(values < -1e-7) or np.any(values > 1.0 + 1e-7):
            minimum_value = float(np.min(values))
            maximum_value = float(np.max(values))
            raise ValueError(
                f"Path diagnostic {diagnostic_name} is outside [0, 1]: "
                f"minimum={minimum_value:.9g}, maximum={maximum_value:.9g}."
            )

    trial_diagnostics = pd.DataFrame(
        {
            "subject": trials["subject"].to_numpy(),
            "chosen_path_probability": chosen_probabilities,
            "path_nll": path_nll,
            "action_policy_confidence": policy_confidence,
            "action_policy_normalized_entropy": normalized_entropy,
            "action_policy_expected_normalized_reward": (
                expected_normalized_reward
            ),
            "action_policy_optimal_path_probability": (
                optimal_path_probability
            ),
            "participant_chosen_path_normalized_reward": (
                chosen_normalized_reward
            ),
            "participant_chosen_path_is_optimal": (
                participant_choice_is_optimal
            ),
            "model_modal_path_matches_participant": modal_choice_matches,
            "chosen_path_probability_rank": chosen_probability_rank,
        }
    )
    participant_rows = []
    for subject, subject_rows in trial_diagnostics.groupby(
        "subject",
        sort=False,
    ):
        participant_rows.append(
            {
                "subject": subject,
                **_summarize_path_diagnostic_rows(subject_rows),
            }
        )
    participant_summary = pd.DataFrame.from_records(participant_rows)
    cell_summary = _summarize_path_diagnostic_rows(trial_diagnostics)

    participant_path_nll = (
        trial_diagnostics.groupby("subject", sort=False)["path_nll"]
        .sum()
        .rename("diagnostic_path_nll")
        .reset_index()
    )
    identity_check = trials[["subject"]].copy()
    identity_check["path_nll"] = path_nll
    identity_check = (
        identity_check.groupby("subject", sort=False)["path_nll"]
        .sum()
        .reset_index()
        .merge(
            participant_path_nll,
            on="subject",
            how="inner",
            validate="one_to_one",
        )
    )
    if not np.allclose(
        identity_check["path_nll"],
        identity_check["diagnostic_path_nll"],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise AssertionError(
            "Path diagnostic failed its internal NLL identity test."
        )
    if trial_records is not None:
        trial_records.append(trial_diagnostics.copy())
    return participant_summary, cell_summary


def aggregate_path_policy_diagnostics(
    by_seed: pd.DataFrame,
    specification: AlternativeModelSpecification,
) -> pd.DataFrame:
    """Aggregate exact simultaneous-policy diagnostics across assigned seeds."""
    parameter_columns = [
        specification.parameter_one_name,
        specification.parameter_two_name,
    ]
    sum_columns = [
        "path_diagnostic_choice_count",
        "path_diagnostic_path_nll_total",
        "path_diagnostic_chosen_probability_sum",
        "path_diagnostic_policy_confidence_sum",
        "path_diagnostic_policy_normalized_entropy_sum",
        "path_diagnostic_expected_normalized_reward_sum",
        "path_diagnostic_optimal_path_probability_sum",
        "path_diagnostic_participant_normalized_reward_sum",
        "path_diagnostic_modal_choice_match_count",
        "path_diagnostic_participant_optimal_choice_count",
        "path_diagnostic_chosen_probability_rank_sum",
    ]
    threshold_labels = ("1e_12", "1e_09", "1e_06", "1e_04", "1e_03")
    for threshold_label in threshold_labels:
        sum_columns.extend(
            [
                (
                    "path_diagnostic_chosen_probability_below_"
                    f"{threshold_label}_count"
                ),
                f"path_diagnostic_path_nll_from_below_{threshold_label}",
            ]
        )

    grouped = by_seed.groupby(parameter_columns, sort=False, dropna=False)
    aggregated = grouped[sum_columns].sum().reset_index()
    seed_counts = (
        grouped.size().rename("path_diagnostic_seed_cell_count").reset_index()
    )
    aggregated = aggregated.merge(
        seed_counts,
        on=parameter_columns,
        how="left",
        validate="one_to_one",
    )
    aggregated["lambda"] = aggregated[specification.parameter_one_name]
    aggregated["alpha"] = aggregated[specification.parameter_two_name]
    aggregated["rho"] = np.divide(
        aggregated["alpha"].to_numpy(dtype=np.float64),
        aggregated["lambda"].to_numpy(dtype=np.float64),
        out=np.full(len(aggregated), np.nan, dtype=np.float64),
        where=aggregated["lambda"].to_numpy(dtype=np.float64) > 0.0,
    )
    choice_count = aggregated["path_diagnostic_choice_count"].to_numpy(
        dtype=np.float64
    )
    path_nll_total = aggregated[
        "path_diagnostic_path_nll_total"
    ].to_numpy(dtype=np.float64)
    if np.any(choice_count <= 0):
        raise AssertionError(
            "Path diagnostic aggregation found a parameter cell with no choices."
        )

    mean_mappings = {
        "path_diagnostic_path_nll_mean_per_choice": (
            "path_diagnostic_path_nll_total"
        ),
        "path_diagnostic_chosen_probability_mean": (
            "path_diagnostic_chosen_probability_sum"
        ),
        "path_diagnostic_policy_confidence_mean": (
            "path_diagnostic_policy_confidence_sum"
        ),
        "path_diagnostic_policy_normalized_entropy_mean": (
            "path_diagnostic_policy_normalized_entropy_sum"
        ),
        "path_diagnostic_expected_normalized_reward_mean": (
            "path_diagnostic_expected_normalized_reward_sum"
        ),
        "path_diagnostic_optimal_path_probability_mean": (
            "path_diagnostic_optimal_path_probability_sum"
        ),
        "path_diagnostic_participant_normalized_reward_mean": (
            "path_diagnostic_participant_normalized_reward_sum"
        ),
        "path_diagnostic_modal_choice_match_rate": (
            "path_diagnostic_modal_choice_match_count"
        ),
        "path_diagnostic_participant_optimal_choice_rate": (
            "path_diagnostic_participant_optimal_choice_count"
        ),
        "path_diagnostic_chosen_probability_rank_mean": (
            "path_diagnostic_chosen_probability_rank_sum"
        ),
    }
    for output_column, sum_column in mean_mappings.items():
        aggregated[output_column] = (
            aggregated[sum_column].to_numpy(dtype=np.float64) / choice_count
        )
    aggregated["path_diagnostic_chosen_probability_geometric_mean"] = np.exp(
        -path_nll_total / choice_count
    )
    for threshold_label in threshold_labels:
        count_column = (
            "path_diagnostic_chosen_probability_below_"
            f"{threshold_label}_count"
        )
        nll_column = f"path_diagnostic_path_nll_from_below_{threshold_label}"
        aggregated[
            "path_diagnostic_chosen_probability_below_"
            f"{threshold_label}_fraction"
        ] = aggregated[count_column].to_numpy(dtype=np.float64) / choice_count
        aggregated[
            f"path_diagnostic_path_nll_below_{threshold_label}_share"
        ] = np.divide(
            aggregated[nll_column].to_numpy(dtype=np.float64),
            path_nll_total,
            out=np.zeros_like(path_nll_total),
            where=path_nll_total > 0.0,
        )

    test_columns = [
        "path_diagnostic_probability_values_test_passed",
        "path_diagnostic_probability_row_sum_test_passed",
        "path_diagnostic_path_nll_identity_test_passed",
        "path_diagnostic_normalized_reward_bounds_test_passed",
    ]
    test_results = grouped[test_columns].all().reset_index()
    aggregated = aggregated.merge(
        test_results,
        on=parameter_columns,
        how="left",
        validate="one_to_one",
    )
    return aggregated


def compare_path_policy_parameter_one_cells(
    aggregated: pd.DataFrame,
    specification: AlternativeModelSpecification,
    reward_similarity_tolerance: float = 0.005,
    fixed_parameter_name: str | None = None,
) -> pd.DataFrame:
    """Compare every lambda pair at a fixed second grid coordinate."""
    one_name = specification.parameter_one_name
    two_name = fixed_parameter_name or specification.parameter_two_name
    comparisons: list[dict[str, float | bool]] = []
    for parameter_two, group in aggregated.groupby(
        two_name,
        sort=False,
        dropna=False,
    ):
        ordered = group.sort_values(one_name, kind="stable").reset_index(
            drop=True
        )
        for left_index in range(len(ordered)):
            for right_index in range(left_index + 1, len(ordered)):
                left = ordered.iloc[left_index]
                right = ordered.iloc[right_index]
                reward_difference = float(
                    right["path_diagnostic_expected_normalized_reward_mean"]
                    - left["path_diagnostic_expected_normalized_reward_mean"]
                )
                nll_difference = float(
                    right["path_diagnostic_path_nll_mean_per_choice"]
                    - left["path_diagnostic_path_nll_mean_per_choice"]
                )
                comparisons.append(
                    {
                        two_name: float(parameter_two),
                        "parameter_one_left": float(left[one_name]),
                        "parameter_one_right": float(right[one_name]),
                        "expected_normalized_reward_left": float(
                            left[
                                "path_diagnostic_expected_normalized_reward_mean"
                            ]
                        ),
                        "expected_normalized_reward_right": float(
                            right[
                                "path_diagnostic_expected_normalized_reward_mean"
                            ]
                        ),
                        "expected_normalized_reward_difference_right_minus_left": (
                            reward_difference
                        ),
                        "path_nll_per_choice_left": float(
                            left[
                                "path_diagnostic_path_nll_mean_per_choice"
                            ]
                        ),
                        "path_nll_per_choice_right": float(
                            right[
                                "path_diagnostic_path_nll_mean_per_choice"
                            ]
                        ),
                        "path_nll_per_choice_difference_right_minus_left": (
                            nll_difference
                        ),
                        "chosen_probability_geometric_mean_left": float(
                            left[
                                "path_diagnostic_chosen_probability_geometric_mean"
                            ]
                        ),
                        "chosen_probability_geometric_mean_right": float(
                            right[
                                "path_diagnostic_chosen_probability_geometric_mean"
                            ]
                        ),
                        "expected_rewards_similar_within_0.005": bool(
                            abs(reward_difference)
                            <= reward_similarity_tolerance
                        ),
                        "lower_path_nll_parameter_one": float(
                            left[one_name]
                            if nll_difference >= 0.0
                            else right[one_name]
                        ),
                    }
                )
    return pd.DataFrame.from_records(comparisons)


def simulate_fitted_participants(
    specification: AlternativeModelSpecification,
    config: ModuleType,
    best_fits: pd.DataFrame,
    model_seeds: list[int],
    trials_per_seed: int,
    batch_size: int,
    simulation_random_seed: int,
    reward_nll_scale: float,
    fit_configuration_hash: str,
) -> pd.DataFrame:
    """Simulate every fitted participant under every requested model seed."""
    category_values = np.asarray(config.reconstruction_values, dtype=np.float64)
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
    participant_order = best_fits["subject"].drop_duplicates().tolist()
    participant_positions = {
        subject: position
        for position, subject in enumerate(participant_order, start=1)
    }
    print(
        "Starting fitted simulations: "
        f"{len(participant_order)} participants x {len(model_seeds)} seeds x "
        f"{trials_per_seed} trials.",
        flush=True,
    )

    frames: list[pd.DataFrame] = []
    next_graph_index = 0
    fit_groups = best_fits.groupby(
        ["best_lambda", "best_alpha"],
        sort=False,
        dropna=False,
    )
    for parameter_index, ((parameter_one, parameter_two), group) in enumerate(
        fit_groups
    ):
        participants = group["subject"].to_numpy()
        nll_seeds = group["nll_model_seed"].to_numpy(dtype=int)
        participant_by_trial = np.repeat(participants, trials_per_seed)
        nll_seed_by_trial = np.repeat(nll_seeds, trials_per_seed)
        within_participant_trial = np.tile(
            np.arange(trials_per_seed, dtype=int),
            len(participants),
        )
        trial_count = len(participant_by_trial)
        reward_generator = np.random.default_rng(
            np.random.SeedSequence(
                [simulation_random_seed, parameter_index, 0]
            )
        )
        trial_rewards = reward_generator.choice(
            reward_values,
            size=(trial_count, config.time_steps),
            replace=True,
        )
        model_inputs = trial_rewards.astype(np.float32)[..., np.newaxis]
        path_rewards = shared.calculate_path_reward_matrix(config, trial_rewards)
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

        for seed_position, model_seed in enumerate(model_seeds, start=1):
            tf.keras.backend.clear_session()
            fitted_model, model_artifact = build_model(
                specification,
                config,
                float(parameter_one),
                float(parameter_two),
                model_seed,
            )
            tensorflow_seed = int(
                np.random.SeedSequence(
                    [simulation_random_seed, parameter_index, model_seed, 1]
                ).generate_state(1, dtype=np.uint32)[0]
                % np.iinfo(np.int32).max
            )
            tf.random.set_seed(tensorflow_seed)
            (
                reconstruction_probabilities,
                action_probabilities,
                information_costs,
            ) = shared.predict_probabilities_once(
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
            sampled_indices = shared.sample_probability_rows(
                reconstruction_probabilities.reshape(
                    -1,
                    config.num_categories,
                ),
                response_generator,
            ).reshape(trial_count, config.time_steps)
            sampled_rewards = category_values[sampled_indices]
            modal_rewards = category_values[
                np.argmax(reconstruction_probabilities, axis=-1)
            ]
            expected_rewards = np.sum(
                reconstruction_probabilities
                * category_values[None, None, :],
                axis=-1,
            )
            true_indices = np.clip(
                np.rint(
                    config.reconstruction_max_reward - trial_rewards
                ).astype(int),
                0,
                config.num_categories - 1,
            )
            true_probabilities = np.take_along_axis(
                reconstruction_probabilities,
                true_indices[..., np.newaxis],
                axis=-1,
            )[..., 0]
            selected_paths = shared.sample_probability_rows(
                action_probabilities,
                response_generator,
            )
            chosen_path_rewards = path_rewards[
                np.arange(trial_count),
                selected_paths,
            ]
            graph_indices = np.arange(
                next_graph_index,
                next_graph_index + trial_count,
                dtype=int,
            )
            next_graph_index += trial_count
            nodes = np.arange(1, config.time_steps + 1, dtype=int)
            frame = pd.DataFrame(
                {
                    "subject": np.repeat(
                        participant_by_trial,
                        config.time_steps,
                    ),
                    "best_lambda": float(parameter_one),
                    "best_alpha": float(parameter_two),
                    f"best_{specification.parameter_one_name}": float(
                        parameter_one
                    ),
                    f"best_{specification.parameter_two_name}": float(
                        parameter_two
                    ),
                    "parameter_one_name": specification.parameter_one_name,
                    "parameter_two_name": specification.parameter_two_name,
                    "model_family": specification.family,
                    "nll_model_seed": np.repeat(
                        nll_seed_by_trial,
                        config.time_steps,
                    ),
                    "simulation_model_seed": model_seed,
                    "simulation_random_seed": simulation_random_seed,
                    "simulation_trials_per_seed": trials_per_seed,
                    "reward_nll_scale": reward_nll_scale,
                    "participant_fit_configuration_hash": (
                        fit_configuration_hash
                    ),
                    "simulation_trial": np.repeat(
                        within_participant_trial,
                        config.time_steps,
                    ),
                    "graph": np.repeat(graph_indices, config.time_steps),
                    "node": np.tile(nodes, trial_count),
                    "actual_reward": trial_rewards.reshape(-1),
                    "estimated_reward": sampled_rewards.reshape(-1),
                    "modal_estimated_reward": modal_rewards.reshape(-1),
                    "expected_estimated_reward": expected_rewards.reshape(-1),
                    "true_reward_probability": true_probabilities.reshape(-1),
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
                    "MI_cost": np.repeat(information_costs, config.time_steps),
                    "is_leaf": np.tile(leaf_flags, trial_count),
                    "model_weight_file": model_artifact,
                    "model_artifact": model_artifact,
                    "reconstruction_head_type": (
                        fitted_model.reconstruction_head_type
                    ),
                    "reconstruction_hidden_units": int(
                        fitted_model.reconstruction_hidden_units
                    ),
                    "reconstruction_loss_scope": (
                        fitted_model.reconstruction_loss_scope
                    ),
                    "reconstruction_min_reward": int(
                        fitted_model.reconstruction_min_reward
                    ),
                    "reconstruction_max_reward": int(
                        fitted_model.reconstruction_max_reward
                    ),
                    "reconstruction_category_count": int(
                        fitted_model.num_categories
                    ),
                    "reward_normalization_constant": float(
                        getattr(
                            fitted_model,
                            "reward_normalization_constant",
                            np.nan,
                        )
                    ),
                    "reward_normalization_scheme": getattr(
                        fitted_model,
                        "reward_normalization_scheme",
                        "not_applicable",
                    ),
                    "critic_loss_enabled": bool(
                        getattr(
                            fitted_model,
                            "critic_loss_enabled",
                            False,
                        )
                    ),
                }
            )
            if specification.family == "independent_channel":
                frame["best_rho"] = float(parameter_two / parameter_one) if parameter_one > 0 else np.nan
                frame["parameter_grid_mode"] = (
                    str(group["parameter_grid_mode"].iloc[0])
                    if "parameter_grid_mode" in group.columns else "alpha"
                )
                frame["channel_kl_aggregation"] = fitted_model.kl_aggregation
                frame["channel_kl_time_weighting"] = (
                    fitted_model.kl_time_weighting
                )
                frame["channel_kl_time_weights"] = ",".join(
                    str(weight)
                    for weight in fitted_model.channel_kl_time_weights
                )
                frame["independent_channel_rnn_units"] = (
                    fitted_model.rnn_units
                )
                frame["independent_channel_latent_dim"] = (
                    fitted_model.latent_dim
                )
            elif specification.family == "simultaneous_channel":
                frame["best_rho"] = (
                    float(parameter_two / parameter_one)
                    if parameter_one > 0
                    else np.nan
                )
                frame["parameter_grid_mode"] = (
                    str(group["parameter_grid_mode"].iloc[0])
                    if "parameter_grid_mode" in group.columns
                    else "alpha"
                )
                frame["observation_schedule"] = (
                    "all_rewards_simultaneous"
                )
                frame["channel_count"] = 1
                frame["kl_terms_per_trial"] = 1
                frame["single_channel_rnn_units"] = fitted_model.rnn_units
                frame["single_channel_latent_dim"] = (
                    fitted_model.latent_dim
                )
                frame["initial_critic_coefficient"] = (
                    config.initial_critic_coefficient
                    if (
                        config.critic_loss_enabled
                        and config.initial_critic_coefficient is not None
                    )
                    else np.nan
                )
            elif specification.family in {
                "earlier_presentation",
                "later_presentation",
            }:
                frame["presentation_order_direction"] = (
                    "earlier"
                    if specification.family == "earlier_presentation"
                    else "later"
                )
            frames.append(frame)
            for subject in participants:
                print(
                    f"Finished simulation for subject {subject} "
                    f"({participant_positions[subject]}/"
                    f"{len(participant_order)}) | seed "
                    f"{seed_position}/{len(model_seeds)} "
                    f"(seed={model_seed}) | trials={trials_per_seed}",
                    flush=True,
                )
            del fitted_model
            gc.collect()

    if not frames:
        raise RuntimeError("No fitted participant simulations were generated.")
    return pd.concat(frames, ignore_index=True)


def configuration_suffix(
    specification: AlternativeModelSpecification,
    args: argparse.Namespace,
    model_seeds: list[int],
    config: ModuleType,
) -> tuple[str, str]:
    """Return a compact, family-specific participant-fit suffix."""
    seed_label = "-".join(str(seed) for seed in model_seeds)
    suffix = (
        f"_seeds_{seed_label}"
        f"_sel_{args.seed_selection_seed}_mc{args.mc_samples}"
        f"_eval{args.evaluation_seed}"
        f"_reward_nll_scale_{args.reward_nll_scale:g}"
    )
    if specification.family == "independent_channel":
        suffix += (
            f"_head_{args.reconstruction_head_type}"
            "_recon_range_minus5_plus5"
            f"_recon_units_{args.reconstruction_hidden_units}"
            "_recon_scope_all_channels"
            f"_kl_{args.channel_kl_aggregation}"
            f"_time_{args.channel_kl_time_weighting}"
            f"_epochs_{args.model_epochs}"
            "_critic_off"
            f"_gradient_scale_{args.max_action_gradient_scale:g}"
            f"_final_lr_{args.final_learning_rate:g}"
        )
        if args.rnn_units != 64 or args.latent_dim != 16:
            suffix += (
                f"_rnn_units_{args.rnn_units}"
                f"_latent_dim_{args.latent_dim}"
            )
        if args.late_lr_plateau_start_epoch is not None:
            suffix += (
                f"_lr_plateau_{args.late_lr_plateau_start_epoch}"
                f"_{args.late_lr_plateau_end_epoch}"
            )
        if args.parameter_multiplier is not None:
            suffix += f"_pm_{args.parameter_multiplier:g}"
        if args.rho_grid:
            suffix += "_rho_grid"
    elif specification.family == "simultaneous_channel":
        suffix += (
            f"_head_{args.reconstruction_head_type}"
            "_recon_range_minus5_plus5"
            f"_recon_units_{args.reconstruction_hidden_units}"
            "_recon_scope_all_nodes"
            f"_rnn_units_{args.rnn_units}_latent_dim_16"
            "_kl_once"
            f"_epochs_{args.model_epochs}"
            f"{config.critic_artifact_suffix()}"
            f"_gradient_scale_{args.max_action_gradient_scale:g}"
            f"_final_lr_{args.final_learning_rate:g}"
        )
        if args.late_lr_plateau_start_epoch is not None:
            suffix += (
                f"_lr_plateau_{args.late_lr_plateau_start_epoch}"
                f"_{args.late_lr_plateau_end_epoch}"
            )
        if args.parameter_multiplier is not None:
            suffix += f"_pm_{args.parameter_multiplier:g}"
        if config.critic_loss_enabled:
            suffix += config.initial_critic_coefficient_artifact_suffix()
        if args.rho_grid:
            suffix += "_rho_grid"
    elif specification.family == "bays_value":
        suffix += "_fixed_reward_allocation_precision_power_law"
    elif specification.family in {
        "earlier_presentation",
        "later_presentation",
    }:
        presentation_direction = (
            "earlier"
            if specification.family == "earlier_presentation"
            else "later"
        )
        suffix += (
            f"_{presentation_direction}_order_inverse_rank_"
            "precision_power_law"
        )
    else:
        raise ValueError(
            f"Unsupported alternative model family: {specification.family}"
        )
    return shared.compact_participant_fit_run_suffix(suffix)


def run(specification: AlternativeModelSpecification) -> None:
    """Fit one alternative model family and write centralized artifacts."""
    args = parse_args(specification)
    tree = shared.resolve_tree(args.tree)
    parameter_one_values = shared.parse_number_list(
        args.parameter_one_values,
        specification.parameter_one_name,
    )
    parameter_two_input_name = (
        "rho"
        if args.rho_grid
        else specification.parameter_two_name
    )
    parameter_two_input_values = shared.parse_number_list(
        args.parameter_two_values,
        parameter_two_input_name,
    )
    if args.rho_grid:
        if any(value <= 0 for value in parameter_one_values):
            raise ValueError(
                "Lambda values must be strictly "
                "positive with --rho."
            )
        if any(value < 0 for value in parameter_two_input_values):
            raise ValueError(
                "Rho values must be non-negative."
            )
    parameter_cells: list[tuple[int, int, float, float, float]] = []
    for one_order, parameter_one in enumerate(parameter_one_values):
        for two_order, parameter_two_input in enumerate(
            parameter_two_input_values
        ):
            if args.rho_grid:
                rho_value = parameter_two_input
                parameter_two = alpha_from_lambda_rho(
                    parameter_one,
                    rho_value,
                )
            else:
                parameter_two = parameter_two_input
                rho_value = (
                    parameter_two / parameter_one
                    if parameter_one > 0
                    else float("nan")
                )
            parameter_cells.append(
                (
                    one_order,
                    two_order,
                    parameter_one,
                    parameter_two,
                    rho_value,
                )
            )
    parameter_two_values = list(
        dict.fromkeys(cell[3] for cell in parameter_cells)
    )
    validate_parameter_grid(
        specification,
        parameter_one_values,
        parameter_two_values,
    )
    model_seeds = shared.parse_seed_list(args.seed_values)
    config = load_config(
        tree,
        parameter_one_values,
        parameter_two_values,
        model_seeds,
        args,
    )
    behavior_path = (
        args.behavior_csv.expanduser()
        if args.behavior_csv is not None
        else shared.default_behavior_path(tree)
    )
    require_existing_path(behavior_path, "Behavioral CSV")
    behavior = shared.prepare_behavioral_trials(
        pd.read_csv(behavior_path, low_memory=False),
        config,
    )
    participants = behavior.trials["subject"].drop_duplicates().tolist()
    participant_positions = {
        subject: position
        for position, subject in enumerate(participants, start=1)
    }
    seed_assignments = shared.assign_nll_model_seeds(
        behavior,
        model_seeds,
        args.seed_selection_seed,
    )
    if args.subject is not None:
        if args.shard_directory is not None:
            raise ValueError("--subject cannot be combined with shard modes.")
        from ..main_vae.participant_fit_shards import canonical_subject
        participants = [s for s in participants
                        if canonical_subject(s) == canonical_subject(args.subject)]
        if len(participants) != 1:
            raise ValueError("--subject must match exactly one behavioral participant.")
        participant_positions = {participants[0]: 1}
        seed_assignments = seed_assignments.loc[
            seed_assignments["subject"].isin(participants)
        ].copy()
    shard_identity = None
    if args.shard_directory is not None:
        if specification.family == "independent_channel":
            from . import independent_fit_shards as shards
        else:
            from . import simultaneous_fit_shards as shards
        from ..main_vae.participant_fit_shards import canonical_subject, read_manifest
        if read_manifest(args.shard_directory / "participants.txt") != [
            canonical_subject(subject) for subject in participants
        ]:
            raise ValueError("Prepared behavioral cohort differs from the submission manifest.")
        shard_identity = shards.run_identity(args, behavior_path)
        if args.participant_index is not None:
            if args.participant_index >= len(participants):
                raise ValueError("Participant index is outside the cohort.")
            participants = [participants[args.participant_index]]
            participant_positions = {participants[0]: 1}
            # Preserve full-cohort seed assignment, then select this participant.
            seed_assignments = seed_assignments.loc[
                seed_assignments["subject"].isin(participants)
            ].copy()
            print(f"Participant index={args.participant_index}, subject={participants[0]}", flush=True)
    behavior_by_seed = {
        int(model_seed): shared.subset_behavioral_trials(
            behavior,
            group["subject"].to_numpy(),
        )
        for model_seed, group in seed_assignments.groupby(
            "nll_model_seed",
            sort=False,
        )
    }
    combination_count = len(parameter_cells)
    print(
        f"Starting {specification.family} participant fit: "
        f"{len(participants)} participants x {combination_count} cells "
        f"({parameter_two_input_name} grid).",
        flush=True,
    )

    grid_frames: list[pd.DataFrame] = []
    path_diagnostic_records: list[dict[str, object]] = []
    trial_diagnostic_records: list[pd.DataFrame] = []
    for (
        one_order,
        two_order,
        parameter_one,
        parameter_two,
        rho_value,
    ) in ([] if args.aggregate_shards else parameter_cells):
        combination_index = (
            one_order * len(parameter_two_input_values) + two_order + 1
        )
        for model_seed, seed_behavior in behavior_by_seed.items():
            rho_log = (
                f" | rho={rho_value:g}"
                if specification.family in {"simultaneous_channel", "independent_channel"}
                else ""
            )
            for subject in seed_behavior.trials[
                "subject"
            ].drop_duplicates():
                print(
                    f"Fitting subject {subject} "
                    f"({participant_positions[subject]}/"
                    f"{len(participants)}) | cell {combination_index}/"
                    f"{combination_count} | "
                    f"{specification.parameter_one_name}="
                    f"{parameter_one:g} | "
                    f"{specification.parameter_two_name}="
                    f"{parameter_two:g}{rho_log} | "
                    f"seed={model_seed}",
                    flush=True,
                )
            tf.keras.backend.clear_session()
            fitted_model, model_artifact = build_model(
                specification,
                config,
                parameter_one,
                parameter_two,
                model_seed,
            )
            action_policy, reconstruction_policy = shared.estimate_policies(
                fitted_model,
                seed_behavior.reward_inputs,
                config.time_steps,
                config.num_paths,
                config.num_categories,
                args.mc_samples,
                args.batch_size,
                args.evaluation_seed,
            )
            results = participant_nll(
                seed_behavior,
                action_policy,
                reconstruction_policy,
                specification,
                parameter_one,
                parameter_two,
                model_seed,
                model_artifact,
                args.reward_nll_scale,
            )
            if specification.family in {"simultaneous_channel", "independent_channel"}:
                results["rho"] = rho_value
                results["parameter_grid_mode"] = parameter_two_input_name
            if specification.family == "simultaneous_channel":
                (
                    participant_diagnostics,
                    cell_diagnostics,
                ) = simultaneous_path_policy_diagnostics(
                    seed_behavior,
                    action_policy,
                    config,
                    trial_records=(trial_diagnostic_records if args.participant_index is not None else None),
                )
                if args.participant_index is not None:
                    raw_diagnostics = trial_diagnostic_records[-1]
                    for column, value in {
                        "lambda": parameter_one, "alpha": parameter_two, "rho": rho_value,
                        "parameter_one_order": one_order, "parameter_two_order": two_order,
                        "nll_model_seed": model_seed, "model_artifact": model_artifact,
                    }.items():
                        raw_diagnostics[column] = value
                results = results.merge(
                    participant_diagnostics,
                    on="subject",
                    how="left",
                    validate="one_to_one",
                )
                if results[
                    "path_diagnostic_path_nll_total"
                ].isna().any():
                    raise AssertionError(
                        "Path diagnostic did not cover every fitted subject."
                    )
                if not np.allclose(
                    results["path_nll"].to_numpy(dtype=np.float64),
                    results[
                        "path_diagnostic_path_nll_total"
                    ].to_numpy(dtype=np.float64),
                    rtol=1e-12,
                    atol=1e-12,
                ):
                    raise AssertionError(
                        "Path diagnostic NLL does not reproduce the fitter's "
                        "participant path NLL."
                    )
                cell_diagnostics.update(
                    {
                        "lambda": parameter_one,
                        "alpha": parameter_two,
                        "rho": rho_value,
                        "parameter_grid_mode": parameter_two_input_name,
                        specification.parameter_one_name: parameter_one,
                        specification.parameter_two_name: parameter_two,
                        "parameter_one_order": one_order,
                        "parameter_two_order": two_order,
                        "nll_model_seed": model_seed,
                        "participant_count": int(
                            results["subject"].nunique()
                        ),
                        "model_artifact": model_artifact,
                    }
                )
                path_diagnostic_records.append(cell_diagnostics)
                chosen_probability_geometric_mean = cell_diagnostics[
                    "path_diagnostic_chosen_probability_geometric_mean"
                ]
                below_1e_06_fraction = cell_diagnostics[
                    "path_diagnostic_chosen_probability_below_1e_06_fraction"
                ]
                expected_normalized_reward_mean = cell_diagnostics[
                    "path_diagnostic_expected_normalized_reward_mean"
                ]
                optimal_path_probability_mean = cell_diagnostics[
                    "path_diagnostic_optimal_path_probability_mean"
                ]
                normalized_entropy_mean = cell_diagnostics[
                    "path_diagnostic_policy_normalized_entropy_mean"
                ]
                print(
                    "Path-policy diagnostic | "
                    f"lambda={parameter_one:g} | alpha={parameter_two:g} | "
                    f"rho={rho_value:g} | "
                    f"seed={model_seed} | "
                    f"choices={cell_diagnostics['path_diagnostic_choice_count']} | "
                    "nll/choice="
                    f"{cell_diagnostics['path_diagnostic_path_nll_mean_per_choice']:.6f} | "
                    "chosen_p(mean/geo/p05/min)="
                    f"{cell_diagnostics['path_diagnostic_chosen_probability_mean']:.6g}/"
                    f"{chosen_probability_geometric_mean:.6g}/"
                    f"{cell_diagnostics['path_diagnostic_chosen_probability_p05']:.6g}/"
                    f"{cell_diagnostics['path_diagnostic_chosen_probability_min']:.6g} | "
                    "p<1e-6="
                    f"{below_1e_06_fraction:.3%} | "
                    "expected_norm_reward="
                    f"{expected_normalized_reward_mean:.6f} | "
                    "optimal_mass="
                    f"{optimal_path_probability_mean:.6f} | "
                    "modal_match="
                    f"{cell_diagnostics['path_diagnostic_modal_choice_match_rate']:.3%} | "
                    "entropy="
                    f"{normalized_entropy_mean:.6f} | "
                    "top5pct_nll_share="
                    f"{cell_diagnostics['path_diagnostic_path_nll_top_5pct_share']:.3%}",
                    flush=True,
                )
            results["parameter_one_order"] = one_order
            results["parameter_two_order"] = two_order
            for row in results.itertuples(index=False):
                print(
                    f"Finished subject {row.subject} "
                    f"({participant_positions[row.subject]}/"
                    f"{len(participants)}) | cell {combination_index}/"
                    f"{combination_count} | total_nll={row.total_nll:.6f}",
                    flush=True,
                )
            grid_frames.append(results)
            del fitted_model, action_policy, reconstruction_policy
            gc.collect()

    if args.aggregate_shards:
        full_grid, raw_diagnostics = shards.read_shards(
            args.shard_directory, participants, shard_identity, parameter_cells, seed_assignments,
        )
        diagnostic_keys = ["parameter_one_order", "parameter_two_order", "nll_model_seed"]
        for _, rows in (raw_diagnostics.groupby(diagnostic_keys, sort=True)
                        if not raw_diagnostics.empty else []):
            record = _summarize_path_diagnostic_rows(rows)
            record.update(rows.iloc[0][diagnostic_keys + ["lambda", "alpha", "rho", "model_artifact"]].to_dict())
            record["participant_count"] = rows["subject"].nunique()
            record["parameter_grid_mode"] = parameter_two_input_name
            path_diagnostic_records.append(record)
        del raw_diagnostics
    else:
        full_grid = pd.concat(grid_frames, ignore_index=True)
    if args.participant_index is not None:
        shards.write_shard(
            args.shard_directory, args.participant_index, participants[0], shard_identity,
            full_grid, (pd.concat(trial_diagnostic_records, ignore_index=True)
                        if trial_diagnostic_records else None),
        )
        print("Participant fitting shard completed. Aggregation will generate fitted simulations and plots.", flush=True)
        return
    full_grid["tree"] = tree.label
    full_grid["candidate_model_seeds"] = ",".join(map(str, model_seeds))
    full_grid["seed_selection_seed"] = args.seed_selection_seed
    full_grid["mc_samples"] = args.mc_samples
    full_grid["evaluation_seed"] = args.evaluation_seed
    full_grid["uses_trained_weights"] = specification.uses_trained_weights
    full_grid["critic_loss_enabled"] = (
        config.critic_loss_enabled
        if specification.uses_trained_weights
        else False
    )
    full_grid["initial_critic_coefficient"] = (
        config.initial_critic_coefficient
        if (
            specification.uses_trained_weights
            and config.critic_loss_enabled
            and config.initial_critic_coefficient is not None
        )
        else np.nan
    )
    full_grid["reward_normalization_constant"] = (
        config.reward_normalization_constant
        if specification.uses_trained_weights
        else np.nan
    )
    full_grid["reward_normalization_scheme"] = (
        config.reward_normalization_scheme
        if specification.uses_trained_weights
        else "not_applicable"
    )
    full_grid["model_epochs"] = (
        args.model_epochs if specification.uses_trained_weights else 0
    )
    full_grid["reconstruction_head_type_requested"] = (
        args.reconstruction_head_type
        if specification.uses_trained_weights
        else "analytic_discretized_gaussian"
    )
    full_grid["reconstruction_hidden_units_requested"] = (
        args.reconstruction_hidden_units
        if specification.uses_trained_weights
        else 0
    )
    full_grid["max_action_gradient_scale"] = (
        args.max_action_gradient_scale
        if specification.uses_trained_weights
        else 1.0
    )
    full_grid["parameter_multiplier_requested"] = (
        args.parameter_multiplier
        if args.parameter_multiplier is not None
        else np.nan
    )
    full_grid["final_learning_rate"] = (
        args.final_learning_rate
        if specification.uses_trained_weights
        else np.nan
    )
    full_grid["late_lr_plateau_start_epoch"] = (
        args.late_lr_plateau_start_epoch
        if specification.uses_trained_weights
        else np.nan
    )
    full_grid["late_lr_plateau_end_epoch"] = (
        args.late_lr_plateau_end_epoch
        if specification.uses_trained_weights
        else np.nan
    )
    full_grid["channel_kl_aggregation"] = (
        args.channel_kl_aggregation
        if specification.family == "independent_channel"
        else "not_applicable"
    )
    full_grid["channel_kl_time_weighting"] = (
        args.channel_kl_time_weighting
        if specification.family == "independent_channel"
        else "not_applicable"
    )
    full_grid["independent_channel_rnn_units"] = (
        args.rnn_units
        if specification.family == "independent_channel"
        else 0
    )
    full_grid["independent_channel_latent_dim"] = (
        args.latent_dim
        if specification.family == "independent_channel"
        else 0
    )
    full_grid["observation_schedule"] = (
        "all_rewards_simultaneous"
        if specification.family == "simultaneous_channel"
        else "not_applicable"
    )
    full_grid["channel_count"] = (
        1 if specification.family == "simultaneous_channel" else 0
    )
    full_grid["kl_terms_per_trial"] = (
        1 if specification.family == "simultaneous_channel" else 0
    )
    full_grid["single_channel_rnn_units"] = (
        args.rnn_units
        if specification.family == "simultaneous_channel"
        else 0
    )
    full_grid["single_channel_latent_dim"] = (
        16 if specification.family == "simultaneous_channel" else 0
    )
    full_grid = full_grid.sort_values(
        ["subject", "parameter_one_order", "parameter_two_order"],
        kind="stable",
    ).reset_index(drop=True)
    best_fits = (
        full_grid.sort_values(
            [
                "subject",
                "total_nll",
                "parameter_one_order",
                "parameter_two_order",
            ],
            kind="stable",
        )
        .groupby("subject", as_index=False, sort=False)
        .first()
        .rename(
            columns={
                "lambda": "best_lambda",
                "alpha": "best_alpha",
                specification.parameter_one_name: (
                    f"best_{specification.parameter_one_name}"
                ),
                specification.parameter_two_name: (
                    f"best_{specification.parameter_two_name}"
                ),
            }
        )
    )
    if specification.family in {"simultaneous_channel", "independent_channel"}:
        best_fits["best_rho"] = np.divide(
            best_fits["best_alpha"].to_numpy(dtype=np.float64),
            best_fits["best_lambda"].to_numpy(dtype=np.float64),
            out=np.full(len(best_fits), np.nan, dtype=np.float64),
            where=(
                best_fits["best_lambda"].to_numpy(dtype=np.float64) > 0.0
            ),
        )
    fit_suffix, fit_hash = configuration_suffix(
        specification,
        args,
        model_seeds,
        config,
    )
    full_grid["participant_fit_configuration_hash"] = fit_hash
    best_fits["participant_fit_configuration_hash"] = fit_hash
    full_grid["participant_fit_artifact_suffix"] = fit_suffix
    best_fits["participant_fit_artifact_suffix"] = fit_suffix
    safe_tree = "".join(
        character if character.isalnum() or character in "_-" else "_"
        for character in tree.label
    )
    summary_directory = ensure_directory(SUMMARY_DIR / "participant_fits")
    simulation_directory = ensure_directory(
        SIMULATION_DIR / "participant_fits"
    )
    stem = f"{specification.artifact_tag}_{safe_tree}{fit_suffix}"
    grid_path = summary_directory / f"participant_parameter_grid_{stem}.csv"
    fit_path = summary_directory / f"participant_parameter_fits_{stem}.csv"
    simulation_path = simulation_directory / (
        f"participant_fitted_simulations_{stem}"
        f"_n{args.simulation_trials_per_seed}"
        f"_simseed{args.simulation_random_seed}.csv"
    )
    diagnostic_paths: tuple[Path, ...] = ()
    if specification.family == "simultaneous_channel":
        if not path_diagnostic_records:
            raise AssertionError(
                "Simultaneous fitting completed without path-policy diagnostics."
            )
        diagnostic_by_seed = pd.DataFrame.from_records(
            path_diagnostic_records
        ).sort_values(
            ["parameter_one_order", "parameter_two_order", "nll_model_seed"],
            kind="stable",
        )
        diagnostic_by_seed["tree"] = tree.label
        diagnostic_by_seed["participant_fit_configuration_hash"] = fit_hash
        diagnostic_by_seed["participant_fit_artifact_suffix"] = fit_suffix
        diagnostic_aggregate = aggregate_path_policy_diagnostics(
            diagnostic_by_seed,
            specification,
        ).sort_values(
            [specification.parameter_one_name, specification.parameter_two_name],
            kind="stable",
        )
        diagnostic_aggregate["tree"] = tree.label
        diagnostic_aggregate["parameter_grid_mode"] = parameter_two_input_name
        diagnostic_aggregate[
            "participant_fit_configuration_hash"
        ] = fit_hash
        diagnostic_aggregate["participant_fit_artifact_suffix"] = fit_suffix
        diagnostic_comparisons = compare_path_policy_parameter_one_cells(
            diagnostic_aggregate,
            specification,
            fixed_parameter_name=("rho" if args.rho_grid else None),
        )
        if not diagnostic_comparisons.empty:
            diagnostic_comparisons["tree"] = tree.label
            diagnostic_comparisons[
                "participant_fit_configuration_hash"
            ] = fit_hash
            diagnostic_comparisons[
                "participant_fit_artifact_suffix"
            ] = fit_suffix

        diagnostic_seed_path = summary_directory / (
            f"path_policy_diagnostics_seed_{stem}.csv"
        )
        diagnostic_cell_path = summary_directory / (
            f"path_policy_diagnostics_cell_{stem}.csv"
        )
        diagnostic_comparison_path = summary_directory / (
            f"path_policy_lambda_comparison_{stem}.csv"
        )
        shared.write_csv_atomically(
            diagnostic_by_seed.drop(
                columns=["parameter_one_order", "parameter_two_order"]
            ),
            diagnostic_seed_path,
        )
        shared.write_csv_atomically(
            diagnostic_aggregate,
            diagnostic_cell_path,
        )
        shared.write_csv_atomically(
            diagnostic_comparisons,
            diagnostic_comparison_path,
        )
        diagnostic_paths = (
            diagnostic_seed_path,
            diagnostic_cell_path,
            diagnostic_comparison_path,
        )
        for _, diagnostic_row in diagnostic_aggregate.iterrows():
            below_1e_06_fraction = diagnostic_row[
                "path_diagnostic_chosen_probability_below_1e_06_fraction"
            ]
            print(
                "Aggregate path-policy diagnostic | "
                f"lambda={diagnostic_row[specification.parameter_one_name]:g} | "
                f"alpha={diagnostic_row[specification.parameter_two_name]:g} | "
                f"rho={diagnostic_row['rho']:g} | "
                "nll/choice="
                f"{diagnostic_row['path_diagnostic_path_nll_mean_per_choice']:.6f} | "
                "chosen_p(mean/geo)="
                f"{diagnostic_row['path_diagnostic_chosen_probability_mean']:.6g}/"
                f"{diagnostic_row['path_diagnostic_chosen_probability_geometric_mean']:.6g} | "
                "p<1e-6="
                f"{below_1e_06_fraction:.3%} | "
                "nll_from_p<1e-6="
                f"{diagnostic_row['path_diagnostic_path_nll_below_1e_06_share']:.3%} | "
                "expected_norm_reward="
                f"{diagnostic_row['path_diagnostic_expected_normalized_reward_mean']:.6f} | "
                "optimal_mass="
                f"{diagnostic_row['path_diagnostic_optimal_path_probability_mean']:.6f} | "
                "modal_match="
                f"{diagnostic_row['path_diagnostic_modal_choice_match_rate']:.3%} | "
                "entropy="
                f"{diagnostic_row['path_diagnostic_policy_normalized_entropy_mean']:.6f}",
                flush=True,
            )
        if not diagnostic_comparisons.empty:
            similar_comparisons = diagnostic_comparisons.loc[
                diagnostic_comparisons[
                    "expected_rewards_similar_within_0.005"
                ]
            ]
            for _, comparison in similar_comparisons.iterrows():
                print(
                    "Similar-reward lambda comparison | "
                    f"{parameter_two_input_name}="
                    f"{comparison[parameter_two_input_name]:g} | "
                    f"lambda={comparison['parameter_one_left']:g} vs "
                    f"{comparison['parameter_one_right']:g} | "
                    "expected_norm_reward="
                    f"{comparison['expected_normalized_reward_left']:.6f} vs "
                    f"{comparison['expected_normalized_reward_right']:.6f} | "
                    "path_nll/choice="
                    f"{comparison['path_nll_per_choice_left']:.6f} vs "
                    f"{comparison['path_nll_per_choice_right']:.6f} | "
                    "chosen_p_geo="
                    f"{comparison['chosen_probability_geometric_mean_left']:.6g} vs "
                    f"{comparison['chosen_probability_geometric_mean_right']:.6g} | "
                    "lower_nll_lambda="
                    f"{comparison['lower_path_nll_parameter_one']:g}",
                    flush=True,
                )
    shared.write_csv_atomically(
        full_grid.drop(
            columns=["parameter_one_order", "parameter_two_order"]
        ),
        grid_path,
    )
    shared.write_csv_atomically(
        best_fits.drop(
            columns=["parameter_one_order", "parameter_two_order"]
        ),
        fit_path,
    )
    simulations = simulate_fitted_participants(
        specification,
        config,
        best_fits,
        model_seeds,
        args.simulation_trials_per_seed,
        args.batch_size,
        args.simulation_random_seed,
        args.reward_nll_scale,
        fit_hash,
    )
    simulations["participant_fit_artifact_suffix"] = fit_suffix
    shared.write_csv_atomically(simulations, simulation_path)
    if not args.skip_fitted_plots:
        shared.run_fitted_plots(
            tree,
            grid_path,
            fit_path,
            simulation_path,
        )

    print(f"Behavioral input: {behavior_path}")
    print(f"Participants: {len(participants)}")
    print(f"Model family: {specification.family}")
    print(
        f"Parameters: {specification.parameter_one_name}, "
        f"{parameter_two_input_name}"
    )
    print_local_scp_command(
        (grid_path, fit_path, simulation_path, *diagnostic_paths),
        ".local_artifacts/participant_fit_audit",
    )


def run_family(family: ModelFamily) -> None:
    """Public entry point used by each family-specific thin module."""
    run(MODEL_SPECIFICATIONS[family])
