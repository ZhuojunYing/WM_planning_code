"""Generate participant-fitted simulations for the mixed 31-node task.

This is intentionally separate from the expensive likelihood-grid fitter.
It reads one completed ``participant_parameter_fits_31_*.csv`` table, loads
the checkpoint selected by each participant's best lambda/alpha pair, and
generates new trials under every requested model seed and all four Experiment
3 tree-by-order topologies.

The default output contains 25 trials for every

    participant x model seed x tree type

combination. Reward trials are paired across model seeds and tree types within
participant so differences are not caused by a different random reward set.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import tensorflow as tf


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_paths import (
    REPO_ROOT,  # noqa: E402
    SIMULATION_DIR,
    ensure_directory,
    print_local_scp_command,
    require_existing_path,
)

from .fit_participant_parameters import (  # noqa: E402
    parse_seed_list,
    simulate_fitted_participants,
    write_csv_atomically,
)
from .fit_participant_parameters_31 import (  # noqa: E402
    TREE_TYPES,
    load_isolated_config,
)


FIT_FILENAME_PREFIX = "participant_parameter_fits_31"
SIMULATION_FILENAME_PREFIX = "participant_fitted_simulations_31"


def parse_args() -> argparse.Namespace:
    """Parse standalone post-fit simulation arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate new fitted simulations from a completed mixed 31-node "
            "participant best-fit CSV."
        )
    )
    parser.add_argument(
        "--fit-csv",
        type=Path,
        required=True,
        help="Completed participant_parameter_fits_31_*.csv table.",
    )
    parser.add_argument(
        "--simulation-seeds",
        default=None,
        help=(
            "Comma-separated trained-model seeds. Defaults to the "
            "candidate_model_seeds recorded in the fit CSV."
        ),
    )
    parser.add_argument(
        "--trials-per-seed-tree",
        type=int,
        default=25,
        help=(
            "New trials generated for every participant/model-seed/tree-type "
            "combination (default: 25; each tree type is a tree/order condition)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Model prediction batch size (default: 128).",
    )
    parser.add_argument(
        "--simulation-random-seed",
        type=int,
        default=2027,
        help="Random seed for rewards and sampled responses (default: 2027).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Optional output path under the centralized simulations root. "
            "By default the filename is derived from the fit CSV."
        ),
    )
    parser.add_argument("--participant-index", type=int, default=None,
                        help="Simulate only this zero-based row of the best-fit CSV; requires --output-csv.")
    args = parser.parse_args()
    if args.participant_index is not None and (args.participant_index < 0 or args.output_csv is None):
        parser.error("--participant-index must be non-negative and requires --output-csv.")
    if args.trials_per_seed_tree < 1:
        parser.error("--trials-per-seed-tree must be positive.")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    if args.simulation_random_seed < 0:
        parser.error("--simulation-random-seed must be non-negative.")
    return args


def one_value(frame: pd.DataFrame, column: str) -> object:
    """Return one non-missing configuration value from a fit table."""
    if column not in frame.columns:
        raise ValueError(
            f"Participant best-fit CSV is missing metadata column: {column}"
        )
    values = frame[column].dropna().drop_duplicates().tolist()
    if len(values) != 1:
        raise ValueError(
            f"Expected one {column} value in the best-fit CSV; "
            f"found {len(values)}."
        )
    return values[0]


def parse_bool(value: object, label: str) -> bool:
    """Parse a CSV Boolean without relying on Python string truthiness."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid Boolean for {label}: {value!r}")


def optional_float(value: object, label: str) -> float | None:
    """Convert numeric metadata while preserving the ``auto`` sentinel."""
    normalized = str(value).strip().lower()
    if normalized in {"", "auto", "none", "nan", "na"}:
        return None
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Non-finite {label}: {value!r}")
    return result


def optional_epoch(frame: pd.DataFrame, column: str) -> int | None:
    """Read one optional epoch value from the fit metadata."""
    if column not in frame.columns:
        raise ValueError(
            f"Participant best-fit CSV is missing metadata column: {column}"
        )
    values = frame[column].dropna().drop_duplicates().tolist()
    if not values:
        return None
    if len(values) != 1:
        raise ValueError(f"Expected one {column}; found {len(values)}.")
    result = int(values[0])
    if result < 1:
        raise ValueError(f"{column} must be positive when present.")
    return result


def configuration_from_fits(best_fits: pd.DataFrame) -> SimpleNamespace:
    """Reconstruct the model-identifying arguments saved by the fitter."""
    head_type = str(
        one_value(best_fits, "reconstruction_head_type")
    ).strip()
    if head_type not in {
        "nonlinear_location_scale",
        "nonlinear_categorical",
    }:
        raise ValueError(
            "Unsupported reconstruction_head_type in participant fits: "
            f"{head_type}"
        )

    plateau_start = optional_epoch(
        best_fits,
        "late_lr_plateau_start_epoch",
    )
    plateau_end = optional_epoch(
        best_fits,
        "late_lr_plateau_end_epoch",
    )
    if (plateau_start is None) != (plateau_end is None):
        raise ValueError(
            "Late learning-rate plateau start and end must both be present "
            "or both be absent."
        )

    return SimpleNamespace(
        model_epochs=int(one_value(best_fits, "model_epochs")),
        max_action_gradient_scale=float(
            one_value(best_fits, "max_action_gradient_scale")
        ),
        action_entropy_bonus=float(
            one_value(best_fits, "action_entropy_bonus")
        ),
        rnn_units=int(one_value(best_fits, "rnn_units")),
        reconstruction_hidden_units=int(
            one_value(best_fits, "reconstruction_hidden_units")
        ),
        reconstruction_loss_scope=str(
            one_value(best_fits, "reconstruction_loss_scope")
        ).strip(),
        nonlinear_scale=head_type == "nonlinear_location_scale",
        final_learning_rate=float(
            one_value(best_fits, "final_learning_rate")
        ),
        late_lr_plateau_start_epoch=plateau_start,
        late_lr_plateau_end_epoch=plateau_end,
        parameter_multiplier=optional_float(
            one_value(best_fits, "parameter_multiplier"),
            "parameter_multiplier",
        ),
        initial_critic_coefficient=optional_float(
            one_value(best_fits, "initial_critic_coefficient"),
            "initial_critic_coefficient",
        ),
        no_critic_loss=not parse_bool(
            one_value(best_fits, "critic_loss_enabled"),
            "critic_loss_enabled",
        ),
    )


def validate_best_fits(best_fits: pd.DataFrame) -> None:
    """Validate participant identities, best parameters, and fit scope."""
    required_columns = {
        "subject",
        "best_lambda",
        "best_alpha",
        "nll_model_seed",
        "fit_condition",
        "participant_fit_configuration_hash",
    }
    missing_columns = sorted(required_columns.difference(best_fits.columns))
    if missing_columns:
        raise ValueError(
            "Participant best-fit CSV is missing required columns: "
            + ", ".join(missing_columns)
        )
    if best_fits.empty:
        raise ValueError("Participant best-fit CSV contains no rows.")
    if best_fits["subject"].isna().any():
        raise ValueError("Participant best-fit CSV contains a missing subject.")
    if best_fits["subject"].duplicated().any():
        raise ValueError(
            "Participant best-fit CSV must contain exactly one row per subject."
        )
    for column in ("best_lambda", "best_alpha", "nll_model_seed"):
        numeric_values = pd.to_numeric(best_fits[column], errors="coerce")
        if numeric_values.isna().any() or not np.all(
            np.isfinite(numeric_values)
        ):
            raise ValueError(f"Participant fits contain invalid {column} values.")
    if (pd.to_numeric(best_fits["nll_model_seed"]) < 0).any():
        raise ValueError("nll_model_seed values must be non-negative.")
    fit_conditions = {
        str(value).strip().lower()
        for value in best_fits["fit_condition"].dropna().unique()
    }
    if fit_conditions != {"planning"}:
        raise ValueError(
            "Mixed 31-node fitted simulations require planning fits; found "
            + ", ".join(sorted(fit_conditions))
        )


def simulation_seeds_from_fits(
    best_fits: pd.DataFrame,
    explicit_seeds: str | None,
) -> list[int]:
    """Use explicit simulation seeds or the fit's candidate seed list."""
    seed_text = (
        explicit_seeds
        if explicit_seeds is not None
        else str(one_value(best_fits, "candidate_model_seeds"))
    )
    return parse_seed_list(seed_text)


def default_output_path(
    fit_path: Path,
    trials_per_seed_tree: int,
    simulation_random_seed: int,
) -> Path:
    """Derive a centralized fitted-simulation path from the fit filename."""
    stem = fit_path.stem
    if not stem.startswith(FIT_FILENAME_PREFIX):
        raise ValueError(
            "The fit CSV filename must start with "
            f"{FIT_FILENAME_PREFIX!r}; received {fit_path.name!r}."
        )
    fit_suffix = stem[len(FIT_FILENAME_PREFIX):]
    filename = (
        f"{SIMULATION_FILENAME_PREFIX}{fit_suffix}"
        f"_n{trials_per_seed_tree}_per_seed_tree"
        f"_simseed{simulation_random_seed}.csv"
    )
    return SIMULATION_DIR / "participant_fits" / filename


def validate_output_path(path: Path) -> Path:
    """Require generated output to remain under centralized simulations."""
    expanded_path = path.expanduser()
    try:
        expanded_path.resolve().relative_to(SIMULATION_DIR.resolve())
    except ValueError as error:
        raise ValueError(
            "--output-csv must be inside the centralized simulations root: "
            f"{SIMULATION_DIR}"
        ) from error
    if expanded_path.suffix.lower() != ".csv":
        raise ValueError("--output-csv must end in .csv.")
    return expanded_path


def main() -> None:
    """Generate and save participant-fitted simulations for all topologies."""
    args = parse_args()
    fit_path = require_existing_path(
        args.fit_csv,
        "Mixed 31-node participant best-fit CSV",
    )
    best_fits = pd.read_csv(fit_path, low_memory=False, dtype={"subject": str})
    validate_best_fits(best_fits)
    model_seeds = simulation_seeds_from_fits(
        best_fits,
        args.simulation_seeds,
    )
    model_configuration = configuration_from_fits(best_fits)
    fit_configuration_hash = str(
        one_value(best_fits, "participant_fit_configuration_hash")
    )
    reward_nll_scale = float(one_value(best_fits, "reward_nll_scale"))
    output_path = validate_output_path(
        args.output_csv
        if args.output_csv is not None
        else default_output_path(
            fit_path,
            args.trials_per_seed_tree,
            args.simulation_random_seed,
        )
    )

    participant_random_seed = args.simulation_random_seed
    if args.participant_index is not None:
        if args.participant_index >= len(best_fits):
            raise ValueError("Participant index is outside the best-fit CSV.")
        best_fits = best_fits.iloc[[args.participant_index]].copy()
        participant_random_seed = int(np.random.SeedSequence(
            [args.simulation_random_seed, args.participant_index, 31]
        ).generate_state(1)[0] % (2**31 - 1))
        print(f"Participant task {args.participant_index}: subject={best_fits.iloc[0]['subject']} | "
              f"sampling_seed={participant_random_seed}", flush=True)

    lambda_values = sorted(
        pd.to_numeric(best_fits["best_lambda"]).astype(float).unique().tolist()
    )
    alpha_values = sorted(
        pd.to_numeric(best_fits["best_alpha"]).astype(float).unique().tolist()
    )
    best_fits = best_fits.copy()
    best_fits["best_lambda"] = pd.to_numeric(
        best_fits["best_lambda"]
    ).astype(float)
    best_fits["best_alpha"] = pd.to_numeric(
        best_fits["best_alpha"]
    ).astype(float)
    best_fits["nll_model_seed"] = pd.to_numeric(
        best_fits["nll_model_seed"]
    ).astype(int)

    participant_count = best_fits["subject"].nunique()
    expected_rows = (
        participant_count
        * len(TREE_TYPES)
        * len(model_seeds)
        * args.trials_per_seed_tree
        * 30
    )
    print(
        "Starting mixed 31-node fitted simulations | "
        f"participants={participant_count} | "
        f"tree_types={len(TREE_TYPES)} | "
        f"model_seeds={model_seeds} | "
        f"trials_per_seed_tree={args.trials_per_seed_tree} | "
        f"expected_node_rows={expected_rows}",
        flush=True,
    )

    topology_frames: list[pd.DataFrame] = []
    for tree_index, tree_type in enumerate(TREE_TYPES, start=1):
        print(
            "Starting fitted topology simulation | "
            f"tree_type={tree_type} | topology={tree_index}/{len(TREE_TYPES)}",
            flush=True,
        )
        config = load_isolated_config(
            tree_type,
            lambda_values,
            alpha_values,
            model_seeds[0],
            model_configuration,
        )
        topology = simulate_fitted_participants(
            config=config,
            best_fits=best_fits,
            model_seeds=model_seeds,
            trials_per_seed=args.trials_per_seed_tree,
            batch_size=args.batch_size,
            simulation_random_seed=participant_random_seed,
            reward_nll_scale=reward_nll_scale,
            fit_configuration_hash=fit_configuration_hash,
            condition="planning",
        )
        topology.insert(0, "tree", 31)
        topology.insert(1, "tree_type", tree_type)
        topology.insert(
            2,
            "tree_shape",
            "deep" if tree_type.startswith("deep_") else "wide",
        )
        topology.insert(
            3,
            "presentation_order",
            "DFS" if tree_type.endswith("_depth") else "BFS",
        )
        topology["simulation_trials_per_seed_tree"] = (
            args.trials_per_seed_tree
        )
        if args.participant_index is not None:
            topology["simulation_random_seed"] = args.simulation_random_seed
            topology["participant_simulation_random_seed"] = participant_random_seed
            topology["simulation_participant_index"] = args.participant_index
        topology["paired_trial_id"] = (
            topology["subject"].astype(str)
            + "|"
            + topology["simulation_trial"].astype(str)
        )
        topology["graph_within_topology"] = topology["graph"].astype(int)
        topology["graph"] = (
            tree_type
            + "|seed="
            + topology["simulation_model_seed"].astype(str)
            + "|graph="
            + topology["graph_within_topology"].astype(str)
        )
        topology_frames.append(topology)
        print(
            "Finished fitted topology simulation | "
            f"tree_type={tree_type} | rows={len(topology)}",
            flush=True,
        )
        tf.keras.backend.clear_session()
        gc.collect()

    simulations = pd.concat(topology_frames, ignore_index=True)
    if len(simulations) != expected_rows:
        raise RuntimeError(
            "Generated fitted-simulation row count does not match the full "
            f"factorial design: observed={len(simulations)}, "
            f"expected={expected_rows}."
        )
    identity_columns = [
        "subject",
        "tree_type",
        "simulation_model_seed",
        "simulation_trial",
        "node",
    ]
    if simulations.duplicated(identity_columns).any():
        raise RuntimeError(
            "Generated fitted simulations contain duplicate participant/"
            "topology/model-seed/trial/node rows."
        )
    observed_tree_types = set(simulations["tree_type"].unique())
    if observed_tree_types != set(TREE_TYPES):
        raise RuntimeError("Generated simulations are missing a tree type.")
    if set(simulations["simulation_model_seed"].unique()) != set(model_seeds):
        raise RuntimeError("Generated simulations are missing a model seed.")

    ensure_directory(output_path.parent)
    write_csv_atomically(simulations, output_path)
    print(f"Participant best-fit input: {fit_path}", flush=True)
    print(f"Participants: {participant_count}", flush=True)
    print(f"Tree types: {','.join(TREE_TYPES)}", flush=True)
    print(
        "Simulation model seeds: "
        + ",".join(str(seed) for seed in model_seeds),
        flush=True,
    )
    print(
        "Trials per participant/model-seed/tree-type: "
        f"{args.trials_per_seed_tree}",
        flush=True,
    )
    print(f"Saved fitted simulations to: {output_path}", flush=True)
    print_local_scp_command(
        output_path,
        ".local_artifacts/participant_fit_audit",
    )


if __name__ == "__main__":
    main()
