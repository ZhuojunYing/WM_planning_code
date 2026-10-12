"""Fit one lambda/alpha (or lambda/rho) pair per participant in the mixed 31-node task.

Experiment 3 randomly assigns trials to four tree-by-presentation-order
topologies.  A participant-grid score must therefore route every trial to the
matching ``deep_breadth``, ``deep_depth``, ``wide_breadth``, or ``wide_depth``
model before summing likelihoods across the participant.  Planning fits use

    chosen-path NLL + 0.5 * recalled-reward NLL

by default.  One trained-model seed is sampled per participant and held fixed
over every topology and lambda/alpha grid cell, so seed is not a fitted
parameter.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import sys
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
    SUMMARY_DIR,
    ensure_directory,
    print_local_scp_command,
    require_existing_path,
)

from .fit_participant_parameters import (  # noqa: E402
    TreeSpecification,
    alpha_from_lambda_rho,
    build_and_load_model,
    compact_participant_fit_run_suffix,
    estimate_policies,
    model_configuration_artifact_suffix,
    parse_number_list,
    parse_seed_list,
    participant_nll,
    prepare_behavioral_trials,
    subset_behavioral_trials,
    write_csv_atomically,
)


TREE_TYPES = (
    "deep_breadth",
    "deep_depth",
    "wide_breadth",
    "wide_depth",
)


def parse_args() -> argparse.Namespace:
    """Parse and validate the mixed-tree fitting configuration."""
    parser = argparse.ArgumentParser(
        description=(
            "Fit lambda and alpha to 31-node planning behavior while routing "
            "each trial through its matching tree topology."
        )
    )
    parser.add_argument("tree", choices=("31",), help="Must be 31.")
    parser.add_argument("lambda_values", help="Comma-separated lambda values.")
    parser.add_argument("alpha_values", help="Comma-separated alpha values, or rho values with --rho.")
    parser.add_argument("--rho", dest="rho_grid", action="store_true",
                        help="Interpret the second grid as rho = alpha / lambda; evaluate alpha = lambda * rho.")
    parser.add_argument(
        "--seed-values",
        default="1,2,3,4,5",
        help="Comma-separated trained-model seeds (default: 1,2,3,4,5).",
    )
    parser.add_argument(
        "--seed-selection-seed",
        type=int,
        default=2026,
        help="Participant-to-model-seed assignment seed (default: 2026).",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=32,
        help="Stochastic forward passes averaged per likelihood (default: 32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Behavioral trials evaluated per model call (default: 128).",
    )
    parser.add_argument(
        "--evaluation-seed",
        type=int,
        default=2026,
        help="TensorFlow seed for likelihood evaluation (default: 2026).",
    )
    parser.add_argument(
        "--reward-nll-scale",
        type=float,
        default=0.5,
        help="Recall-NLL multiplier in the fit objective (default: 0.5).",
    )
    parser.add_argument(
        "--behavior-csv",
        type=Path,
        default=None,
        help=(
            "Processed Experiment 3 CSV override. Defaults to the centralized "
            "data/analysis/processed_data/exp3_beh.csv."
        ),
    )
    parser.add_argument(
        "--model-epochs",
        type=int,
        default=320,
        help="Training duration identifying the weights (default: 320).",
    )
    parser.add_argument(
        "--max-action-gradient-scale",
        type=float,
        default=1.0,
        help="Action-gradient scale identifying the weights (default: 1).",
    )
    parser.add_argument(
        "--action-entropy-bonus",
        type=float,
        default=0.0,
        help="Actor entropy bonus identifying the weights (default: 0).",
    )
    parser.add_argument(
        "--rnn-units",
        type=int,
        default=64,
        help="LSTM hidden-state width identifying the weights (default: 64).",
    )
    reconstruction_group = parser.add_mutually_exclusive_group()
    reconstruction_group.add_argument(
        "--nonlinear-scale",
        dest="nonlinear_scale",
        action="store_true",
        help="Use nonlinear location-scale reconstruction weights (default).",
    )
    reconstruction_group.add_argument(
        "--nonlinear-categorical",
        dest="nonlinear_scale",
        action="store_false",
        help="Use nonlinear categorical reconstruction weights.",
    )
    parser.set_defaults(nonlinear_scale=True)
    parser.add_argument(
        "--reconstruction-hidden-units",
        type=int,
        default=128,
        help="Reconstruction-head hidden width (default: 128).",
    )
    parser.add_argument(
        "--reconstruction-loss-scope",
        choices=("all_timesteps", "final_timestep"),
        default="final_timestep",
        help="Reconstruction supervision scope (default: final_timestep).",
    )
    parser.add_argument(
        "--final-learning-rate",
        type=float,
        default=3e-4,
        help="Final learning rate identifying the weights (default: 3e-4).",
    )
    parser.add_argument(
        "--late-lr-plateau-start-epoch",
        type=int,
        default=200,
        help="Late learning-rate plateau start (default: 200).",
    )
    parser.add_argument(
        "--late-lr-plateau-end-epoch",
        type=int,
        default=300,
        help="Late learning-rate plateau end (default: 300).",
    )
    parser.add_argument(
        "--no-late-lr-plateau",
        action="store_true",
        help="Load weights without late-learning-rate plateau tags.",
    )
    parser.add_argument(
        "--parameter-multiplier",
        type=float,
        default=None,
        help="Explicit common loss multiplier identifying the weights.",
    )
    parser.add_argument(
        "--initial-critic-coefficient",
        type=float,
        default=None,
        help="Explicit initial critic coefficient identifying the weights.",
    )
    parser.add_argument(
        "--no-critic-loss",
        action="store_true",
        help="Load weights trained without critic loss.",
    )
    parser.add_argument("--participant-index", type=int, default=None,
                        help="Fit only this zero-based planning-participant index.")
    parser.add_argument("--participant-manifest", type=Path, default=None)
    parser.add_argument("--output-directory", type=Path, default=None,
                        help="Isolated destination for participant shards.")
    args = parser.parse_args()
    if args.participant_index is not None:
        if args.participant_index < 0 or args.output_directory is None:
            parser.error("--participant-index requires a nonnegative index and --output-directory.")

    if args.seed_selection_seed < 0:
        parser.error("--seed-selection-seed must be non-negative.")
    if args.mc_samples < 1:
        parser.error("--mc-samples must be positive.")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    if args.evaluation_seed < 0:
        parser.error("--evaluation-seed must be non-negative.")
    if not np.isfinite(args.reward_nll_scale) or args.reward_nll_scale <= 0:
        parser.error("--reward-nll-scale must be finite and positive.")
    if args.model_epochs < 1:
        parser.error("--model-epochs must be positive.")
    if (
        not np.isfinite(args.max_action_gradient_scale)
        or args.max_action_gradient_scale < 1
    ):
        parser.error("--max-action-gradient-scale must be at least 1.")
    if (
        not np.isfinite(args.action_entropy_bonus)
        or args.action_entropy_bonus < 0
    ):
        parser.error("--action-entropy-bonus must be non-negative.")
    if args.rnn_units < 1:
        parser.error("--rnn-units must be positive.")
    if args.reconstruction_hidden_units < 1:
        parser.error("--reconstruction-hidden-units must be positive.")
    if (
        not np.isfinite(args.final_learning_rate)
        or args.final_learning_rate <= 0
        or args.final_learning_rate > 3e-4
    ):
        parser.error("--final-learning-rate must be in (0, 3e-4].")
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


def build_parameter_cells(lambda_values, parameter_values, rho_grid=False):
    """Build only requested cells, using the training launcher's alpha rounding."""
    if rho_grid and any(value <= 0 for value in lambda_values):
        raise ValueError("lambda_values must be positive with --rho.")
    if rho_grid and any(value < 0 for value in parameter_values):
        raise ValueError("rho_values must be non-negative.")
    cells = []
    for lambda_order, lambda_value in enumerate(lambda_values):
        for alpha_order, value in enumerate(parameter_values):
            alpha = alpha_from_lambda_rho(lambda_value, value) if rho_grid else value
            rho = value if rho_grid else (alpha / lambda_value if lambda_value > 0 else float("nan"))
            cells.append((lambda_order, alpha_order, lambda_value, alpha, rho))
    if len({(cell[2], cell[3]) for cell in cells}) != len(cells):
        raise ValueError("The requested grid produces duplicate raw lambda/alpha cells.")
    return cells


def load_isolated_config(
    tree_type: str,
    lambda_values: list[float],
    alpha_values: list[float],
    seed: int,
    args: argparse.Namespace,
) -> ModuleType:
    """Load one tree configuration without overwriting another topology."""
    original_argv = sys.argv[:]
    sys.argv = [
        "fit_participant_parameters_31.py",
        ",".join(str(value) for value in lambda_values),
        ",".join(str(value) for value in alpha_values),
        "default",
        "default",
        str(args.model_epochs),
        str(seed),
        "31",
        "simulate",
        tree_type,
        "1",
        "--max-action-gradient-scale",
        str(args.max_action_gradient_scale),
        "--action-entropy-bonus",
        str(args.action_entropy_bonus),
        "--rnn-units",
        str(args.rnn_units),
        "--reconstruction-hidden-units",
        str(args.reconstruction_hidden_units),
        "--reconstruction-loss-scope",
        args.reconstruction_loss_scope,
        "--final-learning-rate",
        str(args.final_learning_rate),
        "--condition",
        "planning",
    ]
    sys.argv.append(
        "--nonlinear-scale"
        if args.nonlinear_scale
        else "--nonlinear-categorical"
    )
    if args.no_critic_loss:
        sys.argv.append("--no-critic-loss")
    if args.parameter_multiplier is not None:
        sys.argv.extend(
            ["--parameter-multiplier", str(args.parameter_multiplier)]
        )
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

    module_name = (
        "model.src.main_vae._participant_fit_config_31_"
        + tree_type
    )
    config_path = Path(__file__).with_name("config.py")
    specification = importlib.util.spec_from_file_location(
        module_name,
        config_path,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load model configuration: {config_path}")
    config = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = config
    try:
        specification.loader.exec_module(config)
    finally:
        sys.argv = original_argv
        sys.modules.pop(module_name, None)
    return config


def assign_participant_seeds(
    subjects: list[object],
    model_seeds: list[int],
    selection_seed: int,
) -> pd.DataFrame:
    """Assign one model seed per participant for every topology and cell."""
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


def aggregate_tree_likelihoods(detail: pd.DataFrame) -> pd.DataFrame:
    """Sum the four topology-specific NLL contributions by participant."""
    detail = detail.copy()
    detail["tree_weight_file"] = (
        detail["tree_type"].astype(str)
        + "="
        + detail["weight_file"].astype(str)
    )
    group_columns = [
        "subject",
        "lambda",
        "alpha",
        "nll_model_seed",
        "reward_nll_scale",
        "fit_condition",
        "fit_likelihood",
        "lambda_order",
        "alpha_order",
    ]
    if "rho" in detail:
        group_columns.append("rho")
    summed = detail.groupby(
        group_columns,
        as_index=False,
        sort=False,
        dropna=False,
    ).agg(
        path_nll=("path_nll", "sum"),
        reconstruction_nll=("reconstruction_nll", "sum"),
        scaled_reconstruction_nll=("scaled_reconstruction_nll", "sum"),
        total_nll=("total_nll", "sum"),
        n_path_choices=("n_path_choices", "sum"),
        n_reward_choices=("n_reward_choices", "sum"),
        n_tree_types_evaluated=("tree_type", "nunique"),
    )
    audit = detail.groupby(
        group_columns,
        as_index=False,
        sort=False,
        dropna=False,
    ).agg(
        tree_types_evaluated=(
            "tree_type",
            lambda values: ",".join(sorted(set(values))),
        ),
        weight_files=(
            "tree_weight_file",
            lambda values: "|".join(sorted(set(values))),
        ),
    )
    combined = summed.merge(
        audit,
        on=group_columns,
        how="left",
        validate="one_to_one",
    )
    if not np.allclose(
        combined["total_nll"],
        combined["path_nll"] + combined["scaled_reconstruction_nll"],
        rtol=1e-10,
        atol=1e-10,
    ):
        raise RuntimeError("Aggregated NLL components do not sum correctly.")
    return combined


def main() -> None:
    """Evaluate the mixed-tree likelihood grid and select participant fits."""
    args = parse_args()
    lambda_values = parse_number_list(args.lambda_values, "lambda_values")
    parameter_values = parse_number_list(args.alpha_values, "rho_values" if args.rho_grid else "alpha_values")
    parameter_cells = build_parameter_cells(lambda_values, parameter_values, args.rho_grid)
    # Config/checkpoint loaders always receive actual alpha, never rho.
    alpha_values = list(dict.fromkeys(cell[3] for cell in parameter_cells))
    model_seeds = parse_seed_list(args.seed_values)

    behavior_path = (
        args.behavior_csv.expanduser()
        if args.behavior_csv is not None
        else behavior_input_path("exp3")
    )
    require_existing_path(behavior_path, "Experiment 3 behavioral CSV")
    behavior_frame = pd.read_csv(behavior_path, low_memory=False)
    missing_routing_columns = {
        "condition",
        "tree_type",
    }.difference(behavior_frame.columns)
    if missing_routing_columns:
        raise ValueError(
            "Experiment 3 behavior is missing routing columns: "
            + ", ".join(sorted(missing_routing_columns))
        )
    behavior_frame["tree_type"] = (
        behavior_frame["tree_type"]
        .astype("string")
        .str.strip()
        .str.lower()
    )
    observed_tree_types = set(behavior_frame["tree_type"].dropna().unique())
    unexpected_tree_types = observed_tree_types.difference(TREE_TYPES)
    if unexpected_tree_types:
        raise ValueError(
            "Experiment 3 behavior contains unsupported tree types: "
            + ", ".join(sorted(unexpected_tree_types))
        )

    configs: dict[str, ModuleType] = {}
    behavior_by_tree: dict[str, object] = {}
    for tree_type in TREE_TYPES:
        tree_rows = behavior_frame.loc[
            behavior_frame["tree_type"].eq(tree_type)
        ].copy()
        if tree_rows.empty:
            raise ValueError(
                f"Experiment 3 behavior contains no {tree_type} trials."
            )
        config = load_isolated_config(
            tree_type,
            lambda_values,
            alpha_values,
            model_seeds[0],
            args,
        )
        configs[tree_type] = config
        behavior_by_tree[tree_type] = prepare_behavioral_trials(
            tree_rows,
            config,
            condition="planning",
        )

    subject_order = (
        behavior_frame.loc[
            behavior_frame["condition"]
            .astype(str)
            .str.lower()
            .eq("planning"),
            "subject",
        ]
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    if not subject_order:
        raise ValueError("No planning participants were found in Experiment 3.")
    participant_positions = {
        subject: index
        for index, subject in enumerate(subject_order, start=1)
    }
    seed_assignments = assign_participant_seeds(
        subject_order,
        model_seeds,
        args.seed_selection_seed,
    )
    # Assign seeds on the complete cohort before selecting a shard, so the
    # participant-to-checkpoint mapping agrees with a full-cohort fit.
    if args.participant_manifest is not None:
        from .participant_fit_shards import canonical_subject, read_manifest
        if read_manifest(args.participant_manifest) != [
            canonical_subject(subject) for subject in subject_order
        ]:
            raise ValueError("Behavioral participant order differs from the submission manifest.")
    if args.participant_index is not None:
        if args.participant_index >= len(subject_order):
            raise ValueError("Participant index is outside the planning cohort.")
        selected_subject = subject_order[args.participant_index]
        seed_assignments = seed_assignments.loc[
            seed_assignments["subject"].eq(selected_subject)
        ].copy()
        subject_order = [selected_subject]
        participant_positions = {selected_subject: 1}
        print(f"Fitting participant index={args.participant_index}, subject={selected_subject}", flush=True)

    behavior_by_tree_and_seed: dict[tuple[str, int], object] = {}
    for tree_type, tree_behavior in behavior_by_tree.items():
        available_subjects = set(tree_behavior.trials["subject"].unique())
        for model_seed, assignment_group in seed_assignments.groupby(
            "nll_model_seed",
            sort=False,
        ):
            selected_subjects = np.asarray(
                [
                    subject
                    for subject in assignment_group["subject"].tolist()
                    if subject in available_subjects
                ]
            )
            if selected_subjects.size == 0:
                continue
            behavior_by_tree_and_seed[(tree_type, int(model_seed))] = (
                subset_behavioral_trials(tree_behavior, selected_subjects)
            )

    parameter_count = len(parameter_cells)
    print(
        "Starting mixed 31-node participant fit | "
        f"participants={len(subject_order)} | "
        f"parameter_combinations={parameter_count} | "
        f"grid_mode={'rho' if args.rho_grid else 'alpha'} | "
        f"tree_types={','.join(TREE_TYPES)} | "
        f"candidate_seeds={','.join(str(seed) for seed in model_seeds)} | "
        f"objective=path_nll+{args.reward_nll_scale:g}*recall_nll",
        flush=True,
    )

    detail_results: list[pd.DataFrame] = []
    for lambda_order, lambda_value in enumerate(lambda_values):
        for _, alpha_order, _, alpha_value, rho_value in (
            cell for cell in parameter_cells if cell[0] == lambda_order
        ):
            combination_index = (
                lambda_order * len(parameter_values) + alpha_order + 1
            )
            for tree_type in TREE_TYPES:
                config = configs[tree_type]
                for model_seed in model_seeds:
                    seed_behavior = behavior_by_tree_and_seed.get(
                        (tree_type, model_seed)
                    )
                    if seed_behavior is None:
                        continue
                    print(
                        "Evaluating mixed-tree grid | "
                        f"combination={combination_index}/{parameter_count} | "
                        f"lambda={lambda_value:g} | alpha={alpha_value:g} | "
                        f"rho={rho_value:g} | "
                        f"tree_type={tree_type} | model_seed={model_seed} | "
                        f"trials={len(seed_behavior.trials)}",
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
                        condition="planning",
                    )
                    model_results["tree_type"] = tree_type
                    model_results["tree_size"] = 31
                    model_results["lambda_order"] = lambda_order
                    model_results["alpha_order"] = alpha_order
                    model_results["rho"] = rho_value
                    detail_results.append(model_results)
                    for result in model_results.itertuples(index=False):
                        print(
                            "Finished topology contribution | "
                            f"subject={result.subject} "
                            f"({participant_positions[result.subject]}/"
                            f"{len(subject_order)}) | "
                            f"combination={combination_index}/{parameter_count} | "
                            f"tree_type={tree_type} | "
                            f"total_nll={result.total_nll:.6f}",
                            flush=True,
                        )
                    del fitted_model, action_policy, reconstruction_policy
                    gc.collect()

    if not detail_results:
        raise RuntimeError("No Experiment 3 likelihoods were evaluated.")
    tree_grid = pd.concat(detail_results, ignore_index=True)
    full_grid = aggregate_tree_likelihoods(tree_grid)

    representative_tree = TreeSpecification(31, TREE_TYPES[0], TREE_TYPES[0])
    artifact_suffix = model_configuration_artifact_suffix(
        configs[TREE_TYPES[0]],
        representative_tree,
        lambda_values[0],
        alpha_values[0],
        model_seeds[0],
    )
    for tree_type in TREE_TYPES[1:]:
        candidate_suffix = model_configuration_artifact_suffix(
            configs[tree_type],
            TreeSpecification(31, tree_type, tree_type),
            lambda_values[0],
            alpha_values[0],
            model_seeds[0],
        )
        if candidate_suffix != artifact_suffix:
            raise RuntimeError(
                "The four topology configurations do not share one model "
                "artifact suffix."
            )

    representative_config = configs[TREE_TYPES[0]]
    configuration_columns = {
        "tree": "31",
        "candidate_tree_types": ",".join(TREE_TYPES),
        "candidate_model_seeds": ",".join(
            str(seed) for seed in model_seeds
        ),
        "seed_selection_seed": args.seed_selection_seed,
        "model_epochs": args.model_epochs,
        "mc_samples": args.mc_samples,
        "evaluation_seed": args.evaluation_seed,
        "reconstruction_head_type": (
            representative_config.reconstruction_head_type
        ),
        "reconstruction_hidden_units": (
            representative_config.reconstruction_hidden_units
        ),
        "reconstruction_loss_scope": (
            representative_config.reconstruction_loss_scope
        ),
        "rnn_units": representative_config.rnn_units,
        "latent_dim": representative_config.latent_dim,
        "critic_loss_enabled": representative_config.critic_loss_enabled,
        "max_action_gradient_scale": args.max_action_gradient_scale,
        "action_entropy_bonus": args.action_entropy_bonus,
        "parameter_multiplier": (
            representative_config.parameter_multiplier
            if representative_config.parameter_multiplier is not None
            else "auto"
        ),
        "initial_critic_coefficient": (
            representative_config.initial_critic_coefficient
            if representative_config.initial_critic_coefficient is not None
            else "auto"
        ),
        "final_learning_rate": args.final_learning_rate,
        "late_lr_plateau_start_epoch": args.late_lr_plateau_start_epoch,
        "late_lr_plateau_end_epoch": args.late_lr_plateau_end_epoch,
        "reward_normalization_constant": (
            representative_config.reward_normalization_constant
        ),
        "reward_normalization_scheme": (
            representative_config.reward_normalization_scheme
        ),
        "model_configuration_artifact_suffix": artifact_suffix,
    }
    for column, value in configuration_columns.items():
        full_grid[column] = value
        tree_grid[column] = value

    seed_label = "-".join(str(seed) for seed in model_seeds)
    full_suffix = (
        f"_seeds_{seed_label}_sel_{args.seed_selection_seed}"
        f"_mc{args.mc_samples}_eval{args.evaluation_seed}"
        f"_reward_nll_scale_{args.reward_nll_scale:g}{artifact_suffix}"
        f"{'_rho_grid' if args.rho_grid else ''}"
    )
    fit_suffix, configuration_hash = compact_participant_fit_run_suffix(
        full_suffix
    )
    full_grid["participant_fit_configuration_hash"] = configuration_hash
    tree_grid["participant_fit_configuration_hash"] = configuration_hash
    full_grid["participant_fit_artifact_suffix"] = fit_suffix
    tree_grid["participant_fit_artifact_suffix"] = fit_suffix

    full_grid = full_grid.sort_values(
        ["subject", "lambda_order", "alpha_order"],
        kind="stable",
    ).reset_index(drop=True)
    tree_grid = tree_grid.sort_values(
        ["subject", "lambda_order", "alpha_order", "tree_type"],
        kind="stable",
    ).reset_index(drop=True)
    best_fits = (
        full_grid.sort_values(
            ["subject", "total_nll", "lambda_order", "alpha_order"],
            kind="stable",
        )
        .groupby("subject", sort=False, group_keys=False)
        .head(1)
        .rename(columns={"lambda": "best_lambda", "alpha": "best_alpha", "rho": "best_rho"})
        .reset_index(drop=True)
    )

    output_directory = ensure_directory(
        args.output_directory if args.output_directory is not None
        else SUMMARY_DIR / "participant_fits"
    )
    tree_grid_path = output_directory / (
        f"participant_parameter_grid_by_tree_31{fit_suffix}.csv"
    )
    grid_path = output_directory / (
        f"participant_parameter_grid_31{fit_suffix}.csv"
    )
    fit_path = output_directory / (
        f"participant_parameter_fits_31{fit_suffix}.csv"
    )
    write_csv_atomically(
        tree_grid.drop(columns=["lambda_order", "alpha_order"]),
        tree_grid_path,
    )
    write_csv_atomically(
        full_grid.drop(columns=["lambda_order", "alpha_order"]),
        grid_path,
    )
    write_csv_atomically(
        best_fits.drop(columns=["lambda_order", "alpha_order"]),
        fit_path,
    )

    print(f"Behavioral input: {behavior_path}")
    print("Fit condition: planning")
    print(f"Participants: {full_grid['subject'].nunique()}")
    print(f"Tree types: {','.join(TREE_TYPES)}")
    print(f"Candidate model seeds: {','.join(str(seed) for seed in model_seeds)}")
    print(f"Reward-NLL scale: {args.reward_nll_scale:g}")
    print(f"Per-tree likelihood grid: {tree_grid_path}")
    print(f"Summed participant grid: {grid_path}")
    print(f"Best participant fits: {fit_path}")
    print_local_scp_command(
        (tree_grid_path, grid_path, fit_path),
        ".local_artifacts/participant_fit_audit",
    )


if __name__ == "__main__":
    main()
