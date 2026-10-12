"""Recover known Main-VAE lambda and alpha values from synthetic behavior.

The analysis preserves the empirical trial designs from the selected human
condition, but replaces path choices and recalled rewards with samples from a
known generating model. Synthetic participants are fitted in matched-seed mode:

* ``matched_seed`` evaluates every candidate parameter pair using the trained
  seed that generated that participant.
No seed-robust fits are evaluated. All generated artifacts use the
repository-wide centralized paths.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_paths import (
    behavior_path as behavior_input_path,
    REPO_ROOT,  # noqa: E402
    DATA_DIR,
    PLOT_DIR,
    RESULT_DIR,
    SUMMARY_DIR,
    ensure_directory,
    job_temporary_path,
    print_local_scp_command,
    require_existing_path,
)

from .fit_participant_parameters import (  # noqa: E402
    alpha_from_lambda_rho,
    BehavioralTrials,
    TreeSpecification,
    build_and_load_model,
    default_behavior_path,
    estimate_policies,
    load_model_config,
    model_configuration_artifact_suffix,
    parse_number_list,
    parse_seed_list,
    participant_nll,
    prepare_behavioral_trials,
    resolve_tree,
    sample_probability_rows,
    subset_behavioral_trials,
    write_csv_atomically,
)


RECOVERY_MODES = ("matched_seed",)


@dataclass
class RecoveryDesign:
    """Synthetic participant design plus its known parameter assignments."""

    behavior: BehavioralTrials
    assignments: pd.DataFrame


def log_runtime_stage(message: str) -> None:
    """Print a timestamped recovery-stage message without output buffering."""
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse parameter-recovery and model-artifact arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic participants from known Main-VAE lambda and "
            "alpha values, then evaluate matched-seed recovery only."
        )
    )
    parser.add_argument("tree", choices=("7", "13", "31"), help="Human task design; 31 mixes all four topologies.")
    parser.add_argument("lambda_values", help="Comma-separated lambda grid.")
    parser.add_argument("alpha_values", help="Comma-separated alpha grid.")
    parser.add_argument("--rho", action="store_true", help="Use rho coordinates, with alpha=lambda*rho.")
    parser.add_argument("--rnn-units", type=int, default=128)
    parser.add_argument("--participant-index", type=int, default=None,
                        help="Zero-based synthetic participant index for a recovery shard.")
    parser.add_argument("--shard-directory", type=Path, default=None)
    parser.add_argument("--aggregate-directory", type=Path, default=None)
    parser.add_argument(
        "--seed-values",
        default="1,2,3,4,5",
        help="Comma-separated trained-model seeds (default: 1,2,3,4,5).",
    )
    parser.add_argument(
        "--replicates-per-cell",
        type=int,
        default=10,
        help=(
            "Synthetic participants generated per true lambda/alpha cell "
            "(default: 10)."
        ),
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=1,
        help="Latent draws averaged during fitting (default: 1, matching MC1 fits).",
    )
    parser.add_argument(
        "--generation-mc-samples",
        type=int,
        default=None,
        help=(
            "Latent draws averaged before sampling synthetic responses; "
            "defaults to --mc-samples."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Model-evaluation batch size (default: 128).",
    )
    parser.add_argument(
        "--condition",
        choices=("planning", "exposure"),
        default="planning",
        help=(
            "Human design and likelihood to recover. Planning fits path plus "
            "recall NLL; exposure fits recall NLL only (default: planning)."
        ),
    )
    parser.add_argument(
        "--reward-nll-scale",
        type=float,
        default=None,
        help="Recall-NLL weight; defaults to 0.5 planning and 1 exposure.",
    )
    parser.add_argument(
        "--behavior-csv",
        type=Path,
        default=None,
        help="Optional processed behavioral CSV supplying trial templates.",
    )
    parser.add_argument(
        "--recovery-seed",
        type=int,
        default=2028,
        help="Synthetic design and response RNG seed.",
    )
    parser.add_argument(
        "--evaluation-seed",
        type=int,
        default=2026,
        help="Common Monte Carlo seed used for all candidate fits.",
    )
    parser.add_argument(
        "--model-epochs",
        type=int,
        default=320,
        help="Training epoch tag identifying model artifacts (default: 320).",
    )
    parser.add_argument(
        "--max-action-gradient-scale",
        type=float,
        default=1.0,
        help="Action-gradient artifact tag (default: 1).",
    )
    parser.add_argument(
        "--parameter-multiplier",
        type=float,
        default=None,
        help="Optional common loss multiplier artifact tag.",
    )
    parser.add_argument(
        "--initial-critic-coefficient",
        type=float,
        default=None,
        help="Optional initial critic coefficient artifact tag.",
    )
    parser.add_argument(
        "--action-entropy-bonus",
        type=float,
        default=0.0,
        help="Action-entropy bonus artifact tag (default: 0).",
    )
    parser.add_argument(
        "--no-critic-loss",
        action="store_true",
        help="Use artifacts trained without critic loss.",
    )
    head_group = parser.add_mutually_exclusive_group()
    head_group.add_argument(
        "--nonlinear-scale",
        dest="nonlinear_scale",
        action="store_true",
        help="Use the nonlinear location-scale reconstruction head.",
    )
    head_group.add_argument(
        "--nonlinear-categorical",
        dest="nonlinear_scale",
        action="store_false",
        help="Use the nonlinear categorical reconstruction head.",
    )
    parser.set_defaults(nonlinear_scale=True)
    parser.add_argument(
        "--reconstruction-hidden-units",
        type=int,
        default=128,
        help="Reconstruction hidden width artifact tag (default: 128).",
    )
    parser.add_argument(
        "--reconstruction-loss-scope",
        choices=("all_timesteps", "final_timestep"),
        default="final_timestep",
        help="Reconstruction supervision artifact tag.",
    )
    parser.add_argument(
        "--final-learning-rate",
        type=float,
        default=3e-4,
        help="Final learning-rate artifact tag (default: 3e-4).",
    )
    parser.add_argument(
        "--late-lr-plateau-start-epoch",
        type=int,
        default=200,
        help="Late plateau start artifact tag (default: 200).",
    )
    parser.add_argument(
        "--late-lr-plateau-end-epoch",
        type=int,
        default=300,
        help="Late plateau end artifact tag (default: 300).",
    )
    parser.add_argument(
        "--no-late-lr-plateau",
        action="store_true",
        help="Use artifacts without late learning-rate plateau tags.",
    )
    args = parser.parse_args()
    if args.rnn_units < 1:
        parser.error("--rnn-units must be positive.")
    if args.participant_index is not None and (
        args.participant_index < 0 or args.shard_directory is None
    ):
        parser.error("--participant-index requires a nonnegative index and --shard-directory.")
    if args.tree == "31" and args.condition != "planning":
        parser.error("The mixed 31-node design supports planning only.")

    if args.replicates_per_cell < 1:
        parser.error("--replicates-per-cell must be positive.")
    if args.mc_samples < 1:
        parser.error("--mc-samples must be positive.")
    if args.generation_mc_samples is None:
        args.generation_mc_samples = args.mc_samples
    if args.generation_mc_samples < 1:
        parser.error("--generation-mc-samples must be positive.")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    if args.recovery_seed < 0 or args.evaluation_seed < 0:
        parser.error("Recovery and evaluation seeds must be non-negative.")
    if args.model_epochs < 1:
        parser.error("--model-epochs must be positive.")
    if (
        not np.isfinite(args.max_action_gradient_scale)
        or args.max_action_gradient_scale < 1
    ):
        parser.error("--max-action-gradient-scale must be at least 1.")
    if args.parameter_multiplier is not None and (
        not np.isfinite(args.parameter_multiplier)
        or args.parameter_multiplier <= 0
    ):
        parser.error("--parameter-multiplier must be positive.")
    if args.initial_critic_coefficient is not None and (
        not np.isfinite(args.initial_critic_coefficient)
        or args.initial_critic_coefficient < 0
    ):
        parser.error("--initial-critic-coefficient must be non-negative.")
    if (
        not np.isfinite(args.action_entropy_bonus)
        or args.action_entropy_bonus < 0
    ):
        parser.error("--action-entropy-bonus must be non-negative.")
    if args.condition == "exposure":
        if args.action_entropy_bonus != 0:
            parser.error("Exposure recovery requires zero action entropy bonus.")
        if args.initial_critic_coefficient is not None:
            parser.error(
                "--initial-critic-coefficient is not applicable to exposure."
            )
        args.no_critic_loss = True
    if args.reconstruction_hidden_units < 1:
        parser.error("--reconstruction-hidden-units must be positive.")
    if (
        not np.isfinite(args.final_learning_rate)
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
        parser.error("Plateau epochs must satisfy 0 < start < end < epochs.")
    if args.reward_nll_scale is None:
        args.reward_nll_scale = 0.5 if args.condition == "planning" else 1.0
    if (
        not np.isfinite(args.reward_nll_scale)
        or args.reward_nll_scale <= 0
    ):
        parser.error("--reward-nll-scale must be positive.")
    return args


def build_recovery_design(
    template_behavior: BehavioralTrials,
    lambda_values: list[float],
    alpha_values: list[float],
    model_seeds: list[int],
    replicates_per_cell: int,
    recovery_seed: int,
    condition: str,
    rho_grid: bool = False,
    participant_index_only: int | None = None,
) -> RecoveryDesign:
    """Replicate empirical trial structures across every generating cell."""
    template_subjects = (
        template_behavior.trials["subject"].drop_duplicates().tolist()
    )
    if not template_subjects:
        raise ValueError("No participant trial templates were available.")

    rng = np.random.default_rng(recovery_seed)
    template_order = rng.permutation(np.asarray(template_subjects, dtype=object))
    trial_frames: list[pd.DataFrame] = []
    recall_frames: list[pd.DataFrame] = []
    reward_chunks: list[np.ndarray] = []
    assignment_rows: list[dict[str, object]] = []
    next_trial_index = 0
    participant_index = 0

    for lambda_order, true_lambda in enumerate(lambda_values):
        for alpha_order, parameter_two in enumerate(alpha_values):
            true_alpha = alpha_from_lambda_rho(true_lambda, parameter_two) if rho_grid else parameter_two
            cell_index = lambda_order * len(alpha_values) + alpha_order
            for replicate in range(1, replicates_per_cell + 1):
                template_subject = template_order[
                    participant_index % len(template_order)
                ]
                synthetic_subject = (
                    f"recovery_l{lambda_order + 1}_a{alpha_order + 1}_"
                    f"r{replicate}"
                )
                generating_seed = model_seeds[
                    (cell_index * replicates_per_cell + replicate - 1)
                    % len(model_seeds)
                ]
                # Preserve legacy RNG draws and assignment metadata so that
                # removing robust fits does not change the synthetic design.
                # This seed is recorded for compatibility but never evaluated.
                robust_seed = int(rng.choice(np.asarray(model_seeds, dtype=int)))
                if participant_index_only is not None and participant_index != participant_index_only:
                    participant_index += 1
                    continue

                template_trials = template_behavior.trials.loc[
                    template_behavior.trials["subject"].eq(template_subject)
                ].copy()
                original_trial_indices = template_trials[
                    "trial_index"
                ].to_numpy(dtype=int)
                new_trial_indices = np.arange(
                    next_trial_index,
                    next_trial_index + len(template_trials),
                    dtype=int,
                )
                trial_index_map = dict(
                    zip(original_trial_indices, new_trial_indices, strict=True)
                )
                template_trials["trial_index"] = new_trial_indices
                template_trials["subject"] = synthetic_subject
                if condition == "planning":
                    template_trials["path_index"] = -1
                    template_trials["chosen_path"] = ""
                trial_frames.append(template_trials)

                template_recalls = template_behavior.recalls.loc[
                    template_behavior.recalls["subject"].eq(template_subject)
                ].copy()
                template_recalls["trial_index"] = (
                    template_recalls["trial_index"].map(trial_index_map)
                )
                if template_recalls["trial_index"].isna().any():
                    raise RuntimeError("Failed to remap a template recall trial.")
                template_recalls["trial_index"] = template_recalls[
                    "trial_index"
                ].astype(int)
                template_recalls["subject"] = synthetic_subject
                template_recalls["selected_reward"] = np.nan
                template_recalls["category_index"] = -1
                recall_frames.append(template_recalls)
                reward_chunks.append(
                    template_behavior.reward_inputs[original_trial_indices]
                )

                assignment_rows.append(
                    {
                        "subject": synthetic_subject,
                        "true_lambda": float(true_lambda),
                        "true_alpha": float(true_alpha),
                        "true_rho": float(parameter_two if rho_grid else true_alpha / true_lambda),
                        "lambda_order": int(lambda_order),
                        "alpha_order": int(alpha_order),
                        "replicate": int(replicate),
                        "template_subject": str(template_subject),
                        "generating_seed": int(generating_seed),
                        "matched_fit_seed": int(generating_seed),
                        "robust_fit_seed": int(robust_seed),
                    }
                )
                next_trial_index += len(template_trials)
                participant_index += 1

    if not trial_frames:
        raise ValueError("Synthetic participant index is outside the recovery design.")
    trials = pd.concat(trial_frames, ignore_index=True)
    recalls = pd.concat(recall_frames, ignore_index=True)
    reward_inputs = np.concatenate(reward_chunks, axis=0).astype(np.float32)
    if not np.array_equal(
        trials["trial_index"].to_numpy(dtype=int),
        np.arange(len(trials), dtype=int),
    ):
        raise RuntimeError("Synthetic trial indices are not contiguous.")
    if len(reward_inputs) != len(trials):
        raise RuntimeError("Synthetic rewards and trial rows are misaligned.")
    return RecoveryDesign(
        behavior=BehavioralTrials(trials, recalls, reward_inputs),
        assignments=pd.DataFrame.from_records(assignment_rows),
    )


def deterministic_seed(*parts: int) -> int:
    """Return a positive TensorFlow-compatible seed from integer parts."""
    state = np.random.SeedSequence(parts).generate_state(1, dtype=np.uint32)[0]
    return int(state % np.iinfo(np.int32).max)


def simulate_recovery_responses(
    config: ModuleType,
    design: RecoveryDesign,
    generation_mc_samples: int,
    batch_size: int,
    recovery_seed: int,
    condition: str,
) -> None:
    """Sample choices and recalls from each known generating model."""
    behavior = design.behavior
    assignments = design.assignments
    category_values = np.asarray(config.reconstruction_values, dtype=float)
    index_to_path = {
        int(path_index): path_name
        for path_name, path_index in config.est_best_path_map.items()
    }
    groups = assignments.groupby(
        ["true_lambda", "true_alpha", "generating_seed"],
        sort=False,
    )
    group_count = groups.ngroups

    for group_index, (
        (true_lambda, true_alpha, generating_seed),
        group,
    ) in enumerate(groups, start=1):
        subjects = group["subject"].tolist()
        selected_trial_rows = behavior.trials.loc[
            behavior.trials["subject"].isin(subjects)
        ].sort_values("trial_index")
        global_trial_indices = selected_trial_rows[
            "trial_index"
        ].to_numpy(dtype=int)
        log_runtime_stage(
            "Generating recovery responses | "
            f"group={group_index}/{group_count} | "
            f"lambda={true_lambda:g} | alpha={true_alpha:g} | "
            f"seed={int(generating_seed)} | "
            f"participants={len(subjects)} | trials={len(global_trial_indices)}"
        )
        group_started = time.monotonic()
        log_runtime_stage("Clearing the prior TensorFlow session.")
        tf.keras.backend.clear_session()
        log_runtime_stage("Constructing the generating model and loading weights.")
        generating_model, generating_weight_path = build_and_load_model(
            config,
            float(true_lambda),
            float(true_alpha),
            int(generating_seed),
        )
        log_runtime_stage(
            "Generating weights loaded | "
            f"elapsed={time.monotonic() - group_started:.1f}s | "
            f"path={generating_weight_path}"
        )
        generation_seed = deterministic_seed(
            recovery_seed,
            group_index,
            int(generating_seed),
            1,
        )
        evaluation_started = time.monotonic()
        log_runtime_stage(
            "Starting generating-policy Monte Carlo evaluation | "
            f"draws={generation_mc_samples} | batch_size={batch_size}"
        )
        action_policy, reconstruction_policy = estimate_policies(
            generating_model,
            behavior.reward_inputs[global_trial_indices],
            config.time_steps,
            config.num_paths,
            config.num_categories,
            generation_mc_samples,
            batch_size,
            generation_seed,
        )
        log_runtime_stage(
            "Generating-policy evaluation finished | "
            f"elapsed={time.monotonic() - evaluation_started:.1f}s"
        )
        response_rng = np.random.default_rng(
            np.random.SeedSequence(
                [recovery_seed, group_index, int(generating_seed), 2]
            )
        )

        if condition == "planning":
            sampled_path_indices = sample_probability_rows(
                action_policy,
                response_rng,
            ).astype(int)
            behavior.trials.loc[
                selected_trial_rows.index,
                "path_index",
            ] = sampled_path_indices
            behavior.trials.loc[
                selected_trial_rows.index,
                "chosen_path",
            ] = [index_to_path[index] for index in sampled_path_indices]

        selected_recalls = behavior.recalls.loc[
            behavior.recalls["subject"].isin(subjects)
        ].copy()
        global_to_local = {
            int(global_index): local_index
            for local_index, global_index in enumerate(global_trial_indices)
        }
        local_trial_indices = selected_recalls["trial_index"].map(
            global_to_local
        )
        if local_trial_indices.isna().any():
            raise RuntimeError("Failed to map a generated recall trial.")
        local_trial_indices = local_trial_indices.to_numpy(dtype=int)
        node_indices = selected_recalls["node"].to_numpy(dtype=int) - 1
        recall_probabilities = reconstruction_policy[
            local_trial_indices,
            node_indices,
            :,
        ]
        sampled_categories = sample_probability_rows(
            recall_probabilities,
            response_rng,
        ).astype(int)
        behavior.recalls.loc[
            selected_recalls.index,
            "category_index",
        ] = sampled_categories
        behavior.recalls.loc[
            selected_recalls.index,
            "selected_reward",
        ] = category_values[sampled_categories]

        del generating_model, action_policy, reconstruction_policy
        gc.collect()
        log_runtime_stage(
            "Generating group finished | "
            f"elapsed={time.monotonic() - group_started:.1f}s"
        )

    if condition == "planning" and (behavior.trials["path_index"] < 0).any():
        raise RuntimeError("At least one synthetic path response was not generated.")
    if (behavior.recalls["category_index"] < 0).any():
        raise RuntimeError("At least one synthetic recall response was not generated.")


def fit_recovery_grid(
    config: ModuleType,
    design: RecoveryDesign,
    lambda_values: list[float],
    alpha_values: list[float],
    model_seeds: list[int],
    mc_samples: int,
    batch_size: int,
    evaluation_seed: int,
    reward_nll_scale: float,
    condition: str,
    rho_grid: bool = False,
) -> pd.DataFrame:
    """Evaluate matched-seed recovery over the complete candidate grid."""
    behavior = design.behavior
    assignments = design.assignments
    result_frames: list[pd.DataFrame] = []
    cell_count = len(lambda_values) * len(alpha_values)

    for lambda_order, candidate_lambda in enumerate(lambda_values):
        for alpha_order, parameter_two in enumerate(alpha_values):
            candidate_alpha = alpha_from_lambda_rho(candidate_lambda, parameter_two) if rho_grid else parameter_two
            cell_index = lambda_order * len(alpha_values) + alpha_order + 1
            for model_seed in model_seeds:
                matched_subjects = assignments.loc[
                    assignments["matched_fit_seed"].eq(model_seed),
                    "subject",
                ].tolist()
                union_subjects = sorted(set(matched_subjects))
                if not union_subjects:
                    continue
                log_runtime_stage(
                    "Fitting recovery grid | "
                    f"cell={cell_index}/{cell_count} | "
                    f"lambda={candidate_lambda:g} | alpha={candidate_alpha:g} | "
                    f"seed={model_seed} | participants={len(union_subjects)}"
                )
                fit_started = time.monotonic()
                seed_behavior = subset_behavioral_trials(
                    behavior,
                    np.asarray(union_subjects, dtype=object),
                )
                log_runtime_stage("Clearing the prior TensorFlow session.")
                tf.keras.backend.clear_session()
                log_runtime_stage("Constructing the candidate model and loading weights.")
                candidate_model, weight_path = build_and_load_model(
                    config,
                    float(candidate_lambda),
                    float(candidate_alpha),
                    int(model_seed),
                )
                log_runtime_stage(
                    "Candidate weights loaded | "
                    f"elapsed={time.monotonic() - fit_started:.1f}s | "
                    f"path={weight_path}"
                )
                evaluation_started = time.monotonic()
                log_runtime_stage(
                    "Starting candidate-policy Monte Carlo evaluation | "
                    f"draws={mc_samples} | batch_size={batch_size}"
                )
                action_policy, reconstruction_policy = estimate_policies(
                    candidate_model,
                    seed_behavior.reward_inputs,
                    config.time_steps,
                    config.num_paths,
                    config.num_categories,
                    mc_samples,
                    batch_size,
                    evaluation_seed,
                )
                log_runtime_stage(
                    "Candidate-policy evaluation finished | "
                    f"elapsed={time.monotonic() - evaluation_started:.1f}s"
                )
                model_nll = participant_nll(
                    seed_behavior,
                    action_policy,
                    reconstruction_policy,
                    float(candidate_lambda),
                    float(candidate_alpha),
                    int(model_seed),
                    weight_path,
                    reward_nll_scale,
                    condition,
                ).merge(
                    assignments,
                    on="subject",
                    how="left",
                    validate="one_to_one",
                )
                for mode, subjects in (
                    ("matched_seed", matched_subjects),
                ):
                    mode_results = model_nll.loc[
                        model_nll["subject"].isin(subjects)
                    ].copy()
                    mode_results["recovery_mode"] = mode
                    mode_results["candidate_lambda_order"] = lambda_order
                    mode_results["candidate_alpha_order"] = alpha_order
                    mode_results["rho"] = parameter_two if rho_grid else candidate_alpha / candidate_lambda
                    mode_results["fitting_seed"] = int(model_seed)
                    mode_results["fitting_seed_matches_generating_seed"] = (
                        mode_results["fitting_seed"]
                        == mode_results["generating_seed"]
                    )
                    result_frames.append(mode_results)

                del candidate_model, action_policy, reconstruction_policy
                gc.collect()
                log_runtime_stage(
                    "Recovery-grid model finished | "
                    f"elapsed={time.monotonic() - fit_started:.1f}s"
                )

    if not result_frames:
        raise RuntimeError("No parameter-recovery likelihoods were calculated.")
    full_grid = pd.concat(result_frames, ignore_index=True)
    expected_rows = (
        len(assignments)
        * len(lambda_values)
        * len(alpha_values)
        * len(RECOVERY_MODES)
    )
    if len(full_grid) != expected_rows:
        raise RuntimeError(
            "Parameter-recovery grid is incomplete: "
            f"expected {expected_rows} rows, found {len(full_grid)}."
        )
    return full_grid


def pseudo_log(values: pd.Series | np.ndarray, sigma: float) -> np.ndarray:
    """Match a base-10 pseudo-log transform while retaining zero."""
    return np.arcsinh(np.asarray(values, dtype=float) / (2 * sigma)) / np.log(10)


def select_recovered_parameters(
    full_grid: pd.DataFrame,
    lambda_values: list[float],
    alpha_values: list[float],
) -> pd.DataFrame:
    """Select minimum-NLL cells and calculate recovery diagnostics."""
    selected = (
        full_grid.sort_values(
            [
                "recovery_mode",
                "subject",
                "total_nll",
                "candidate_lambda_order",
                "candidate_alpha_order",
            ],
            kind="stable",
        )
        .groupby(["recovery_mode", "subject"], as_index=False, sort=False)
        .first()
        .rename(
            columns={
                "lambda": "recovered_lambda",
                "alpha": "recovered_alpha",
            }
        )
    )
    selected["lambda_recovered_exactly"] = np.isclose(
        selected["recovered_lambda"],
        selected["true_lambda"],
        rtol=1e-10, atol=0,
    )
    selected["alpha_recovered_exactly"] = np.isclose(
        selected["recovered_alpha"],
        selected["true_alpha"],
        rtol=1e-10, atol=0,
    )
    selected["jointly_recovered_exactly"] = (
        selected["lambda_recovered_exactly"]
        & selected["alpha_recovered_exactly"]
    )
    selected["lambda_grid_error"] = (
        selected["candidate_lambda_order"] - selected["lambda_order"]
    )
    selected["alpha_grid_error"] = (
        selected["candidate_alpha_order"] - selected["alpha_order"]
    )
    selected["lambda_within_one_grid_step"] = (
        selected["lambda_grid_error"].abs() <= 1
    )
    selected["alpha_within_one_grid_step"] = (
        selected["alpha_grid_error"].abs() <= 1
    )
    alpha_positive = [value for value in alpha_values if value > 0]
    alpha_sigma = min(alpha_positive) / 2 if alpha_positive else 1.0
    selected["lambda_log10_error"] = np.log10(
        selected["recovered_lambda"]
    ) - np.log10(selected["true_lambda"])
    selected["alpha_pseudo_log_error"] = pseudo_log(
        selected["recovered_alpha"],
        alpha_sigma,
    ) - pseudo_log(selected["true_alpha"], alpha_sigma)
    return selected


def recovery_summaries(
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return overall and true-cell recovery summaries."""

    def summarize(group: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "participant_count": len(group),
                "lambda_exact_rate": group[
                    "lambda_recovered_exactly"
                ].mean(),
                "alpha_exact_rate": group[
                    "alpha_recovered_exactly"
                ].mean(),
                "joint_exact_rate": group[
                    "jointly_recovered_exactly"
                ].mean(),
                "lambda_within_one_step_rate": group[
                    "lambda_within_one_grid_step"
                ].mean(),
                "alpha_within_one_step_rate": group[
                    "alpha_within_one_grid_step"
                ].mean(),
                "lambda_log10_bias": group["lambda_log10_error"].mean(),
                "lambda_log10_rmse": np.sqrt(
                    np.mean(np.square(group["lambda_log10_error"]))
                ),
                "alpha_pseudo_log_bias": group[
                    "alpha_pseudo_log_error"
                ].mean(),
                "alpha_pseudo_log_rmse": np.sqrt(
                    np.mean(np.square(group["alpha_pseudo_log_error"]))
                ),
                "fitting_seed_match_rate": group[
                    "fitting_seed_matches_generating_seed"
                ].mean(),
            }
        )

    overall = (
        selected.groupby("recovery_mode", sort=False)
        .apply(summarize)
        .reset_index()
    )
    by_cell = (
        selected.groupby(
            ["recovery_mode", "true_lambda", "true_alpha"],
            sort=False,
        )
        .apply(summarize)
        .reset_index()
    )
    return overall, by_cell


def format_parameter(value: float) -> str:
    """Format parameter values compactly for plot axes."""
    if value == 0:
        return "0"
    if abs(value) < 0.001 or abs(value) >= 1000:
        return f"{value:.1e}"
    return f"{value:g}"


def save_figure_atomically(figure: plt.Figure, output_path: Path) -> None:
    """Publish a PNG only after matplotlib finishes writing it."""
    ensure_directory(output_path.parent)
    temporary_path = job_temporary_path(
        output_path,
        label="parameter_recovery_plot",
        suffix=".png.in_progress",
    )
    temporary_path.unlink(missing_ok=True)
    figure.savefig(
        temporary_path,
        format="png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    temporary_path.replace(output_path)


def plot_confusion_recovery(
    selected: pd.DataFrame,
    mode: str,
    lambda_values: list[float],
    alpha_values: list[float],
    output_path: Path,
    second_label: str = "Alpha",
) -> None:
    """Save row-normalized true-by-recovered confusion matrices."""
    mode_data = selected.loc[selected["recovery_mode"].eq(mode)]
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.7), constrained_layout=True)
    specifications = (
        ("true_lambda", "recovered_lambda", lambda_values, r"$\lambda$"),
        ("true_alpha", "recovered_alpha", alpha_values, second_label),
    )
    image = None
    for axis, (true_column, recovered_column, values, label) in zip(
        axes,
        specifications,
        strict=True,
    ):
        confusion = pd.crosstab(
            mode_data[true_column],
            mode_data[recovered_column],
            normalize="index",
        ).reindex(index=values, columns=values, fill_value=0.0)
        image = axis.imshow(
            confusion.to_numpy(),
            vmin=0,
            vmax=1,
            cmap="Blues",
            aspect="auto",
        )
        axis.set_xticks(range(len(values)), [format_parameter(v) for v in values])
        axis.set_yticks(range(len(values)), [format_parameter(v) for v in values])
        axis.tick_params(axis="x", labelrotation=45, labelsize=7)
        axis.tick_params(axis="y", labelsize=7)
        axis.set_xlabel(f"Recovered {label.lower()}", fontsize=8)
        axis.set_ylabel(f"True {label.lower()}", fontsize=8)
        axis.set_title(f"{label} recovery", fontsize=9)
        for row_index in range(len(values)):
            for column_index in range(len(values)):
                value = confusion.iloc[row_index, column_index]
                if value >= 0.01:
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white" if value > 0.55 else "black",
                    )
    if image is not None:
        figure.colorbar(image, ax=axes, label="Recovery proportion", shrink=0.85)
    figure.suptitle(mode.replace("_", " ").title(), fontsize=10)
    save_figure_atomically(figure, output_path)
    plt.close(figure)


def plot_cross_parameter_recovery(
    selected: pd.DataFrame,
    mode: str,
    lambda_values: list[float],
    alpha_values: list[float],
    output_path: Path,
    second_label: str = "Alpha",
) -> None:
    """Plot whether either true parameter systematically shifts the other."""
    mode_data = selected.loc[selected["recovery_mode"].eq(mode)].copy()
    grouped = (
        mode_data.groupby(["true_lambda", "true_alpha"], as_index=False)
        .agg(
            mean_recovered_lambda=("recovered_lambda", "mean"),
            se_recovered_lambda=(
                "recovered_lambda",
                lambda x: x.std(ddof=1) / np.sqrt(len(x)),
            ),
            mean_recovered_alpha=("recovered_alpha", "mean"),
            se_recovered_alpha=(
                "recovered_alpha",
                lambda x: x.std(ddof=1) / np.sqrt(len(x)),
            ),
        )
        .fillna(0.0)
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), constrained_layout=True)
    lambda_colors = plt.cm.viridis(np.linspace(0.12, 0.9, len(lambda_values)))
    alpha_colors = plt.cm.plasma(np.linspace(0.12, 0.9, len(alpha_values)))
    minimum_lambda = min(lambda_values)

    for color, true_lambda in zip(lambda_colors, lambda_values, strict=True):
        line_data = grouped.loc[
            np.isclose(grouped["true_lambda"], true_lambda)
        ].sort_values("true_alpha")
        lower_error = np.minimum(
            line_data["se_recovered_lambda"].to_numpy(),
            np.maximum(
                line_data["mean_recovered_lambda"].to_numpy()
                - minimum_lambda * 0.5,
                0,
            ),
        )
        axes[0].errorbar(
            line_data["true_alpha"],
            line_data["mean_recovered_lambda"],
            yerr=np.vstack(
                [lower_error, line_data["se_recovered_lambda"].to_numpy()]
            ),
            marker="o",
            markersize=3,
            linewidth=1,
            capsize=2,
            color=color,
            label=rf"True $\lambda$={format_parameter(true_lambda)}",
        )
    positive_alpha = [value for value in alpha_values if value > 0]
    alpha_linthresh = min(positive_alpha) / 2 if positive_alpha else 1.0
    axes[0].set_xscale("symlog", linthresh=alpha_linthresh, base=10)
    axes[0].set_yscale("log", base=10)
    axes[0].set_xticks(alpha_values, [format_parameter(v) for v in alpha_values])
    axes[0].set_yticks(lambda_values, [format_parameter(v) for v in lambda_values])
    axes[0].set_xlabel(f"True {second_label.lower()}")
    axes[0].set_ylabel(r"Mean recovered $\lambda$")
    axes[0].set_title(rf"$\lambda$ cross-talk from {second_label.lower()}")
    axes[0].legend(fontsize=6, frameon=False, loc="best")

    for color, true_alpha in zip(alpha_colors, alpha_values, strict=True):
        line_data = grouped.loc[
            np.isclose(grouped["true_alpha"], true_alpha)
        ].sort_values("true_lambda")
        lower_error = np.minimum(
            line_data["se_recovered_alpha"].to_numpy(),
            line_data["mean_recovered_alpha"].to_numpy(),
        )
        axes[1].errorbar(
            line_data["true_lambda"],
            line_data["mean_recovered_alpha"],
            yerr=np.vstack(
                [lower_error, line_data["se_recovered_alpha"].to_numpy()]
            ),
            marker="o",
            markersize=3,
            linewidth=1,
            capsize=2,
            color=color,
            label=f"True {second_label.lower()}={format_parameter(true_alpha)}",
        )
    axes[1].set_xscale("log", base=10)
    axes[1].set_yscale("symlog", linthresh=alpha_linthresh, base=10)
    axes[1].set_xticks(lambda_values, [format_parameter(v) for v in lambda_values])
    axes[1].set_yticks(alpha_values, [format_parameter(v) for v in alpha_values])
    axes[1].set_xlabel(r"True $\lambda$")
    axes[1].set_ylabel(f"Mean recovered {second_label.lower()}")
    axes[1].set_title(rf"{second_label} cross-talk from $\lambda$")
    axes[1].legend(fontsize=6, frameon=False, loc="best")
    for axis in axes:
        axis.tick_params(labelsize=7)
        axis.grid(alpha=0.15, linewidth=0.5)
    figure.suptitle(mode.replace("_", " ").title(), fontsize=10)
    save_figure_atomically(figure, output_path)
    plt.close(figure)


def recovery_artifact_tag(
    tree: TreeSpecification,
    args: argparse.Namespace,
    lambda_values: list[float],
    alpha_values: list[float],
    model_seeds: list[int],
    model_suffix: str,
) -> tuple[str, str, dict[str, object]]:
    """Return a compact tag, configuration hash, and metadata dictionary."""
    metadata: dict[str, object] = {
        "tree": tree.label,
        "lambda_values": lambda_values,
        "rho_values" if args.rho else "alpha_values": alpha_values,
        "parameter_grid_mode": "rho" if args.rho else "alpha",
        "rnn_units": args.rnn_units,
        "recovery_workflow_version": 3,
        "recovery_modes": list(RECOVERY_MODES),
        "execution_layout": "sharded" if args.participant_index is not None or args.aggregate_directory is not None else "batched",
        "behavior_csv": str(args.behavior_csv) if args.behavior_csv else None,
        "model_seeds": model_seeds,
        "replicates_per_cell": args.replicates_per_cell,
        "mc_samples": args.mc_samples,
        "generation_mc_samples": args.generation_mc_samples,
        "batch_size": args.batch_size,
        "condition": args.condition,
        "reward_nll_scale": args.reward_nll_scale,
        "recovery_seed": args.recovery_seed,
        "evaluation_seed": args.evaluation_seed,
        "model_epochs": args.model_epochs,
        "max_action_gradient_scale": args.max_action_gradient_scale,
        "parameter_multiplier": args.parameter_multiplier,
        "initial_critic_coefficient": args.initial_critic_coefficient,
        "action_entropy_bonus": args.action_entropy_bonus,
        "no_critic_loss": args.no_critic_loss,
        "nonlinear_scale": args.nonlinear_scale,
        "reconstruction_hidden_units": args.reconstruction_hidden_units,
        "reconstruction_loss_scope": args.reconstruction_loss_scope,
        "final_learning_rate": args.final_learning_rate,
        "late_lr_plateau_start_epoch": args.late_lr_plateau_start_epoch,
        "late_lr_plateau_end_epoch": args.late_lr_plateau_end_epoch,
        "model_configuration_artifact_suffix": model_suffix,
    }
    serialized = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    configuration_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
    safe_tree = "".join(
        character if character.isalnum() or character in "_-" else "_"
        for character in tree.label
    )
    tag = (
        f"tree_{safe_tree}_cfg_{configuration_hash}_"
        f"rep{args.replicates_per_cell}_mc{args.mc_samples}"
    )
    return tag, configuration_hash, metadata


def synthetic_response_tables(
    design: RecoveryDesign,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return auditable trial and recall response tables with assignments."""
    assignment_columns = [
        "subject",
        "true_lambda",
        "true_alpha",
        "true_rho",
        "replicate",
        "template_subject",
        "generating_seed",
        "matched_fit_seed",
        "robust_fit_seed",
    ]
    assignments = design.assignments[assignment_columns]
    trials = design.behavior.trials.merge(
        assignments,
        on="subject",
        how="left",
        suffixes=("", "_assignment"),
        validate="many_to_one",
    )
    recalls = design.behavior.recalls.merge(
        assignments,
        on="subject",
        how="left",
        validate="many_to_one",
    )
    return trials, recalls


def concatenate_behaviors(behaviors: list[BehavioralTrials]) -> BehavioralTrials:
    """Join topology-specific designs without colliding trial indices."""
    trials, recalls, rewards = [], [], []
    offset = 0
    for behavior in behaviors:
        trial = behavior.trials.copy()
        recall = behavior.recalls.copy()
        mapping = dict(zip(trial["trial_index"], range(offset, offset + len(trial)), strict=True))
        trial["trial_index"] = trial["trial_index"].map(mapping)
        recall["trial_index"] = recall["trial_index"].map(mapping)
        trials.append(trial)
        recalls.append(recall)
        rewards.append(behavior.reward_inputs)
        offset += len(trial)
    return BehavioralTrials(pd.concat(trials, ignore_index=True),
                            pd.concat(recalls, ignore_index=True),
                            np.concatenate(rewards))


def topology_design(design: RecoveryDesign, tree_type: str) -> RecoveryDesign:
    """Extract one topology while retaining the same synthetic participants."""
    trials = design.behavior.trials.loc[design.behavior.trials["tree_type"].eq(tree_type)].copy()
    indices = trials["trial_index"].to_numpy(dtype=int)
    recalls = design.behavior.recalls.loc[design.behavior.recalls["trial_index"].isin(indices)].copy()
    mapping = dict(zip(indices, range(len(indices)), strict=True))
    trials["trial_index"] = trials["trial_index"].map(mapping)
    recalls["trial_index"] = recalls["trial_index"].map(mapping)
    assignments = design.assignments.loc[design.assignments["subject"].isin(trials["subject"])].copy()
    return RecoveryDesign(BehavioralTrials(trials.reset_index(drop=True), recalls.reset_index(drop=True),
                                           design.behavior.reward_inputs[indices]), assignments)


def sum_topology_grids(grids: list[pd.DataFrame]) -> pd.DataFrame:
    """Sum all topology likelihoods before selecting a participant's pair."""
    detail = pd.concat(grids, ignore_index=True)
    keys = ["recovery_mode", "subject", "candidate_lambda_order", "candidate_alpha_order"]
    sums = ["path_nll", "reconstruction_nll", "scaled_reconstruction_nll",
            "total_nll", "n_path_choices", "n_reward_choices"]
    aggregations = {column: "first" for column in detail.columns if column not in keys}
    aggregations.update({column: "sum" for column in sums})
    aggregations["weight_file"] = lambda values: "|".join(map(str, values))
    aggregations["tree_type"] = lambda values: ",".join(values)
    return detail.groupby(keys, sort=False, as_index=False).agg(aggregations)


def recovery_coordinate_grid(full_grid: pd.DataFrame, rho_grid: bool) -> pd.DataFrame:
    """Adapt existing two-coordinate diagnostics; raw checkpoint alpha is preserved."""
    result = full_grid.copy()
    if rho_grid:
        result["raw_alpha"] = result["alpha"]
        result["raw_true_alpha"] = result["true_alpha"]
        result["alpha"] = result["rho"]
        result["true_alpha"] = result["true_rho"]
    return result


def exported_recovery_statistics(frame: pd.DataFrame, rho_grid: bool) -> pd.DataFrame:
    if not rho_grid:
        return frame
    # Existing rho columns duplicate the diagnostic coordinates below.
    frame = frame.drop(columns=[c for c in ("rho", "true_rho") if c in frame]).copy()
    names = {column: column.replace("alpha", "rho") for column in frame.columns}
    names.update(raw_alpha="recovered_alpha", raw_true_alpha="true_alpha")
    return frame.rename(columns=names)


def main() -> None:
    """Run matched-seed-only parameter recovery."""
    args = parse_args()
    tree = TreeSpecification(31, "mixed", "31") if args.tree == "31" else resolve_tree(args.tree)
    lambda_values = parse_number_list(args.lambda_values, "lambda_values")
    alpha_values = parse_number_list(args.alpha_values, "alpha_values")
    if any(value <= 0 for value in lambda_values):
        raise ValueError("All recovery lambda values must be positive.")
    if any(value < 0 for value in alpha_values):
        raise ValueError("All recovery alpha values must be non-negative.")
    model_seeds = parse_seed_list(args.seed_values)

    raw_alpha_values = list(dict.fromkeys(
        alpha_from_lambda_rho(lam, value) if args.rho else value
        for lam in lambda_values for value in alpha_values
    ))

    if tree.size == 13:
        if args.parameter_multiplier is None:
            args.parameter_multiplier = 1.0
        if args.condition == "planning" and args.initial_critic_coefficient is None:
            args.initial_critic_coefficient = 1.0

    if args.aggregate_directory is not None:
        aggregate_recovery(args, tree, lambda_values, alpha_values, model_seeds)
        return

    if tree.size == 31:
        from .fit_participant_parameters_31 import TREE_TYPES, load_isolated_config
        configs = {kind: load_isolated_config(kind, lambda_values, raw_alpha_values, model_seeds[0], args)
                   for kind in TREE_TYPES}
        config = configs[TREE_TYPES[0]]
    else:
        config = load_model_config(
        tree,
        lambda_values,
        raw_alpha_values,
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
        args.action_entropy_bonus,
        rnn_units=args.rnn_units,
        )
        configs = {"standard": config}
    behavior_path = (
        args.behavior_csv.expanduser()
        if args.behavior_csv is not None
        else (default_behavior_path(tree) if tree.size != 31 else
              behavior_input_path("exp3"))
    )
    require_existing_path(behavior_path, "Behavioral template CSV")
    template_frame = pd.read_csv(behavior_path, low_memory=False)
    templates = []
    for kind, topology_config in configs.items():
        rows = template_frame if kind == "standard" else template_frame.loc[
            template_frame["tree_type"].astype(str).str.strip().str.lower().eq(kind)
        ]
        behavior = prepare_behavioral_trials(rows, topology_config, condition=args.condition)
        behavior.trials["tree_type"] = kind
        templates.append(behavior)
    template_behavior = concatenate_behaviors(templates)
    print(
        "Starting parameter recovery | "
        f"tree={tree.label} | condition={args.condition} | "
        f"cells={len(lambda_values) * len(alpha_values)} | "
        f"replicates_per_cell={args.replicates_per_cell} | "
        f"candidate_seeds={model_seeds} | fit_mc={args.mc_samples} | "
        f"generation_mc={args.generation_mc_samples}",
        flush=True,
    )
    design = build_recovery_design(
        template_behavior,
        lambda_values,
        alpha_values,
        model_seeds,
        args.replicates_per_cell,
        args.recovery_seed,
        args.condition,
        rho_grid=args.rho,
        participant_index_only=args.participant_index,
    )
    fitted_grids, generated_behaviors = [], []
    for topology_index, (kind, topology_config) in enumerate(configs.items()):
        topology = topology_design(design, kind)
        if topology.assignments.empty:
            continue
        log_runtime_stage(f"Recovery topology: {kind}")
        simulate_recovery_responses(topology_config, topology, args.generation_mc_samples,
                                    args.batch_size, deterministic_seed(args.recovery_seed, topology_index), args.condition)
        fitted_grid = fit_recovery_grid(topology_config, topology, lambda_values, alpha_values,
                                        model_seeds, args.mc_samples, args.batch_size, args.evaluation_seed,
                                        args.reward_nll_scale, args.condition, rho_grid=args.rho)
        fitted_grid["tree_type"] = kind
        fitted_grids.append(fitted_grid)
        generated_behaviors.append(topology.behavior)
    full_grid = sum_topology_grids(fitted_grids)
    design.behavior = concatenate_behaviors(generated_behaviors)
    model_suffix = model_configuration_artifact_suffix(
        config,
        TreeSpecification(31, "deep_breadth", "deep_breadth") if tree.size == 31 else tree,
        lambda_values[0],
        raw_alpha_values[0],
        model_seeds[0],
    )
    synthetic_trials, synthetic_recalls = synthetic_response_tables(design)
    if args.participant_index is not None:
        shard = ensure_directory(args.shard_directory / f"participant_{args.participant_index}")
        tag, _, metadata = recovery_artifact_tag(tree, args, lambda_values, alpha_values, model_seeds, model_suffix)
        write_csv_atomically(full_grid, shard / "grid.csv")
        write_csv_atomically(synthetic_trials, shard / "trials.csv")
        write_csv_atomically(synthetic_recalls, shard / "recalls.csv")
        (shard / "config.json").write_text(json.dumps(metadata, sort_keys=True))
        print(f"Completed recovery participant {args.participant_index}: {shard}", flush=True)
        return
    publish_recovery(args, tree, lambda_values, alpha_values, model_seeds, model_suffix,
                     full_grid, synthetic_trials, synthetic_recalls)


def aggregate_recovery(args, tree, lambda_values, alpha_values, model_seeds):
    """Publish only a complete, configuration-consistent recovery cohort."""
    cells = len(lambda_values) * len(alpha_values)
    count = cells * args.replicates_per_cell
    grids, trials, recalls = [], [], []
    model_suffix = None
    subjects = set()
    for index in range(count):
        shard = args.aggregate_directory / f"participant_{index}"
        require_existing_path(shard / "config.json", "Completed recovery shard")
        metadata = json.loads((shard / "config.json").read_text())
        suffix = metadata["model_configuration_artifact_suffix"]
        expected = recovery_artifact_tag(tree, args, lambda_values, alpha_values, model_seeds, suffix)[2]
        if metadata != expected or (model_suffix is not None and suffix != model_suffix):
            raise ValueError(f"Recovery shard configuration mismatch: {shard}")
        model_suffix = suffix
        grid = pd.read_csv(shard / "grid.csv")
        shard_subjects = set(grid["subject"])
        keys = ["subject", "recovery_mode", "candidate_lambda_order", "candidate_alpha_order"]
        if (len(grid) != cells * len(RECOVERY_MODES) or
                set(grid["recovery_mode"]) != set(RECOVERY_MODES) or len(shard_subjects) != 1 or
                subjects.intersection(shard_subjects) or grid.duplicated(keys).any()):
            raise ValueError(f"Incomplete or duplicate recovery grid: {shard}")
        subjects.update(shard_subjects)
        grids.append(grid)
        trials.append(pd.read_csv(shard / "trials.csv"))
        recalls.append(pd.read_csv(shard / "recalls.csv"))
    publish_recovery(args, tree, lambda_values, alpha_values, model_seeds, model_suffix,
                     pd.concat(grids, ignore_index=True), pd.concat(trials, ignore_index=True),
                     pd.concat(recalls, ignore_index=True))


def publish_recovery(args, tree, lambda_values, alpha_values, model_seeds, model_suffix,
                     full_grid, synthetic_trials, synthetic_recalls):
    selected = select_recovered_parameters(recovery_coordinate_grid(full_grid, args.rho), lambda_values, alpha_values)
    overall_summary, cell_summary = recovery_summaries(selected)
    tag, configuration_hash, metadata = recovery_artifact_tag(
        tree,
        args,
        lambda_values,
        alpha_values,
        model_seeds,
        model_suffix,
    )
    for frame in (full_grid, selected, overall_summary, cell_summary):
        frame["parameter_recovery_configuration_hash"] = configuration_hash
        frame["parameter_recovery_artifact_tag"] = tag

    result_directory = ensure_directory(
        RESULT_DIR / "analysis" / "parameter_recovery"
    )
    summary_directory = ensure_directory(
        SUMMARY_DIR / "analysis" / "parameter_recovery"
    )
    plot_directory = ensure_directory(PLOT_DIR / "analysis" / "tests")
    grid_path = result_directory / f"parameter_recovery_grid_{tag}.csv"
    trial_path = result_directory / f"parameter_recovery_trials_{tag}.csv"
    recall_path = result_directory / f"parameter_recovery_recalls_{tag}.csv"
    fits_path = summary_directory / f"parameter_recovery_fits_{tag}.csv"
    overall_path = summary_directory / f"parameter_recovery_summary_{tag}.csv"
    cell_path = summary_directory / f"parameter_recovery_by_cell_{tag}.csv"
    metadata_path = summary_directory / f"parameter_recovery_config_{tag}.json"

    write_csv_atomically(full_grid, grid_path)
    write_csv_atomically(synthetic_trials, trial_path)
    write_csv_atomically(synthetic_recalls, recall_path)
    write_csv_atomically(exported_recovery_statistics(selected, args.rho), fits_path)
    write_csv_atomically(exported_recovery_statistics(overall_summary, args.rho), overall_path)
    write_csv_atomically(exported_recovery_statistics(cell_summary, args.rho), cell_path)
    metadata_temporary_path = job_temporary_path(
        metadata_path,
        label="parameter_recovery_config",
        suffix=".json.in_progress",
    )
    metadata_temporary_path.unlink(missing_ok=True)
    metadata_temporary_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata_temporary_path.replace(metadata_path)

    plot_paths: list[Path] = []
    for mode in RECOVERY_MODES:
        confusion_path = plot_directory / f"parameter_recovery_{mode}_{tag}.png"
        cross_path = plot_directory / (
            f"parameter_recovery_cross_parameters_{mode}_{tag}.png"
        )
        plot_confusion_recovery(
            selected,
            mode,
            lambda_values,
            alpha_values,
            confusion_path,
            second_label=r"$\alpha/\lambda$" if args.rho else "Alpha",
        )
        plot_cross_parameter_recovery(
            selected,
            mode,
            lambda_values,
            alpha_values,
            cross_path,
            second_label=r"$\alpha/\lambda$" if args.rho else "Alpha",
        )
        plot_paths.extend((confusion_path, cross_path))

    print("\nParameter recovery completed.", flush=True)
    print(exported_recovery_statistics(overall_summary, args.rho).to_string(index=False), flush=True)
    print_local_scp_command(
        [grid_path, trial_path, recall_path],
        ".local_artifacts/parameter_recovery",
    )
    print_local_scp_command(
        [fits_path, overall_path, cell_path, metadata_path],
        "analysis/summaries/tests",
    )
    print_local_scp_command(plot_paths, "analysis/figures/tests")


if __name__ == "__main__":
    main()
