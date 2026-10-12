"""Regenerate fitted-participant simulations without repeating model fitting.

This post-fit utility reads a completed 7- or 13-node Main-VAE
``participant_parameter_fits_*.csv`` file, reconstructs the trained-model
configuration stored in that table, and simulates every participant under
every requested model seed.  It does not evaluate the parameter grid or
change any participant's selected lambda, alpha, or assigned NLL seed.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


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
    load_model_config,
    model_configuration_artifact_suffix,
    parse_seed_list,
    resolve_tree,
    simulate_fitted_participants,
    write_csv_atomically,
)


FIT_FILENAME_PREFIX = "participant_parameter_fits_"
SIMULATION_FILENAME_PREFIX = "participant_fitted_simulations_"
MODEL_SEED_TOKEN = re.compile(r"_seed_[^_]+_")


def parse_args() -> argparse.Namespace:
    """Parse simulation-only command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate 7- or 13-node Main-VAE fitted simulations from a "
            "completed participant best-fit CSV without rerunning fitting."
        )
    )
    parser.add_argument("tree", choices=("7", "13"))
    parser.add_argument(
        "--fit-csv",
        type=Path,
        required=True,
        help="Completed participant_parameter_fits_*.csv table.",
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
        "--trials-per-seed",
        type=int,
        default=100,
        help=(
            "Trials generated for every participant/model-seed combination "
            "(default: 100)."
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
        help="Reward and response random seed (default: 2027).",
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
    args = parser.parse_args()
    if args.trials_per_seed < 1:
        parser.error("--trials-per-seed must be positive.")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    if args.simulation_random_seed < 0:
        parser.error("--simulation-random-seed must be non-negative.")
    return args


def one_value(
    frame: pd.DataFrame,
    column: str,
    *,
    default: object | None = None,
    required: bool = True,
) -> object:
    """Return one nonmissing metadata value, with legacy-table fallback."""
    if column not in frame.columns:
        if required:
            raise ValueError(
                f"Participant best-fit CSV is missing metadata column: {column}"
            )
        return default
    values = frame[column].dropna().drop_duplicates().tolist()
    if not values:
        if required:
            raise ValueError(
                f"Participant best-fit CSV has no value for metadata: {column}"
            )
        return default
    if len(values) != 1:
        raise ValueError(
            f"Expected one {column} value in the best-fit CSV; "
            f"found {len(values)}."
        )
    return values[0]


def parse_bool(value: object, label: str) -> bool:
    """Parse Boolean CSV metadata safely."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid Boolean for {label}: {value!r}")


def optional_float(value: object, label: str) -> float | None:
    """Parse optional numeric metadata, preserving auto/none sentinels."""
    normalized = str(value).strip().lower()
    if normalized in {"", "auto", "none", "nan", "na"}:
        return None
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Non-finite {label}: {value!r}")
    return result


def optional_epoch(frame: pd.DataFrame, column: str) -> int | None:
    """Read one optional schedule epoch."""
    value = one_value(frame, column, default=None, required=False)
    if value is None or str(value).strip().lower() in {"", "none", "nan", "na"}:
        return None
    result = int(float(value))
    if result < 1:
        raise ValueError(f"{column} must be positive when present.")
    return result


def infer_condition(best_fits: pd.DataFrame, fit_path: Path) -> str:
    """Read fit condition, supporting planning tables created before tagging."""
    value = one_value(
        best_fits,
        "fit_condition",
        default=None,
        required=False,
    )
    if value is None:
        value = "exposure" if "_exposure_" in fit_path.stem else "planning"
    condition = str(value).strip().lower()
    if condition not in {"planning", "exposure"}:
        raise ValueError(f"Unsupported fit condition: {condition!r}")
    return condition


def validate_best_fits(
    best_fits: pd.DataFrame,
    tree_size: int,
) -> None:
    """Require one valid selected parameter pair per participant."""
    required_columns = {
        "subject",
        "best_lambda",
        "best_alpha",
        "nll_model_seed",
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
        numeric = pd.to_numeric(best_fits[column], errors="coerce")
        if numeric.isna().any() or not np.all(np.isfinite(numeric)):
            raise ValueError(f"Participant fits contain invalid {column} values.")
    if (pd.to_numeric(best_fits["nll_model_seed"]) < 0).any():
        raise ValueError("nll_model_seed values must be non-negative.")
    if "tree" in best_fits.columns:
        tree_values = {
            int(float(value))
            for value in best_fits["tree"].dropna().drop_duplicates()
        }
        if tree_values != {tree_size}:
            raise ValueError(
                f"Fit CSV tree metadata {sorted(tree_values)} does not match "
                f"requested tree {tree_size}."
            )


def simulation_seeds_from_fits(
    best_fits: pd.DataFrame,
    explicit_seeds: str | None,
) -> list[int]:
    """Use explicit seeds or the candidate seed set stored by fitting."""
    seed_text = (
        explicit_seeds
        if explicit_seeds is not None
        else str(
            one_value(
                best_fits,
                "candidate_model_seeds",
                default="1,2,3,4,5",
                required=False,
            )
        )
    )
    return parse_seed_list(seed_text)


def recorded_weight_path_candidates(
    best_fits: pd.DataFrame,
    model_seeds: list[int],
) -> dict[tuple[float, float, int], tuple[Path, ...]]:
    """Recover exact saved and peer-seed weight names from the fit table.

    Main-VAE fit tables store the concrete weight file used for each
    participant's assigned likelihood seed.  Older tables predate the tree
    tag in model filenames, so reconstructing names solely from the current
    config can miss otherwise valid weights.  For every selected parameter
    cell, use all recorded paths as templates and replace only the filename's
    seed token to obtain paths for the remaining requested seeds.
    """
    if "weight_file" not in best_fits.columns:
        print(
            "Fit CSV has no weight_file column; using config-derived model "
            "artifact names only.",
            flush=True,
        )
        return {}

    candidates_by_key: dict[
        tuple[float, float, int], tuple[Path, ...]
    ] = {}
    grouped = best_fits.groupby(
        ["best_lambda", "best_alpha"],
        sort=False,
        dropna=False,
    )
    for (lambda_value, alpha_value), group in grouped:
        recorded_rows = group.loc[
            group["weight_file"].notna(),
            ["nll_model_seed", "weight_file"],
        ].drop_duplicates()
        recorded_paths = [
            (int(row.nll_model_seed), Path(str(row.weight_file)).expanduser())
            for row in recorded_rows.itertuples(index=False)
        ]
        for model_seed in model_seeds:
            candidates: list[Path] = []
            seen: set[Path] = set()

            def add_candidate(path: Path) -> None:
                if path not in seen:
                    candidates.append(path)
                    seen.add(path)

            for recorded_seed, recorded_path in recorded_paths:
                if recorded_seed == model_seed:
                    add_candidate(recorded_path)
            for _, recorded_path in recorded_paths:
                peer_name, replacements = MODEL_SEED_TOKEN.subn(
                    f"_seed_{model_seed}_",
                    recorded_path.name,
                    count=1,
                )
                if replacements == 1:
                    add_candidate(recorded_path.with_name(peer_name))

            if candidates:
                candidates_by_key[
                    (float(lambda_value), float(alpha_value), int(model_seed))
                ] = tuple(candidates)

    print(
        "Recovered saved weight-name candidates for "
        f"{len(candidates_by_key)} parameter/seed combinations from the fit "
        "CSV.",
        flush=True,
    )
    return candidates_by_key


def default_output_path(
    fit_path: Path,
    trials_per_seed: int,
    simulation_random_seed: int,
) -> Path:
    """Derive the established fitted-simulation filename from the fit CSV."""
    if not fit_path.stem.startswith(FIT_FILENAME_PREFIX):
        raise ValueError(
            "The fit CSV filename must start with "
            f"{FIT_FILENAME_PREFIX!r}; received {fit_path.name!r}."
        )
    suffix = fit_path.stem[len(FIT_FILENAME_PREFIX):]
    return SIMULATION_DIR / "participant_fits" / (
        f"{SIMULATION_FILENAME_PREFIX}{suffix}"
        f"_n{trials_per_seed}_simseed{simulation_random_seed}.csv"
    )


def validate_output_path(path: Path) -> Path:
    """Keep generated data within the centralized simulation directory."""
    expanded = path.expanduser()
    try:
        expanded.resolve().relative_to(SIMULATION_DIR.resolve())
    except ValueError as error:
        raise ValueError(
            "--output-csv must be inside the centralized simulations root: "
            f"{SIMULATION_DIR}"
        ) from error
    if expanded.suffix.lower() != ".csv":
        raise ValueError("--output-csv must end in .csv.")
    return expanded


def main() -> None:
    """Load saved fits and generate a new factorial simulation sample."""
    args = parse_args()
    tree = resolve_tree(args.tree)
    fit_path = require_existing_path(
        args.fit_csv,
        "Participant best-fit CSV",
    )
    best_fits = pd.read_csv(fit_path, low_memory=False)
    validate_best_fits(best_fits, tree.size)
    condition = infer_condition(best_fits, fit_path)
    model_seeds = simulation_seeds_from_fits(
        best_fits,
        args.simulation_seeds,
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
    lambda_values = sorted(best_fits["best_lambda"].unique().tolist())
    alpha_values = sorted(best_fits["best_alpha"].unique().tolist())

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
    head_type = str(
        one_value(best_fits, "reconstruction_head_type")
    ).strip()
    if head_type not in {
        "nonlinear_location_scale",
        "nonlinear_categorical",
    }:
        raise ValueError(f"Unsupported reconstruction head: {head_type}")
    critic_enabled = parse_bool(
        one_value(
            best_fits,
            "critic_loss_enabled",
            default=condition == "planning",
            required=False,
        ),
        "critic_loss_enabled",
    )
    parameter_multiplier = optional_float(
        one_value(
            best_fits,
            "parameter_multiplier",
            default="auto",
            required=False,
        ),
        "parameter_multiplier",
    )
    initial_critic_coefficient = optional_float(
        one_value(
            best_fits,
            "initial_critic_coefficient",
            default="auto",
            required=False,
        ),
        "initial_critic_coefficient",
    )
    action_entropy_bonus = float(
        one_value(
            best_fits,
            "action_entropy_bonus",
            default=0.0,
            required=False,
        )
    )
    rnn_units = int(
        float(
            one_value(
                best_fits,
                "rnn_units",
                default=64,
                required=False,
            )
        )
    )
    if rnn_units < 1:
        raise ValueError("Saved fit metadata has invalid rnn_units.")

    config = load_model_config(
        tree=tree,
        lambda_values=lambda_values,
        alpha_values=alpha_values,
        seed=model_seeds[0],
        max_action_gradient_scale=float(
            one_value(best_fits, "max_action_gradient_scale", default=1.0, required=False)
        ),
        parameter_multiplier=parameter_multiplier,
        initial_critic_coefficient=initial_critic_coefficient,
        model_epochs=int(float(one_value(best_fits, "model_epochs"))),
        nonlinear_scale=head_type == "nonlinear_location_scale",
        reconstruction_hidden_units=int(
            float(one_value(best_fits, "reconstruction_hidden_units"))
        ),
        reconstruction_loss_scope=str(
            one_value(best_fits, "reconstruction_loss_scope")
        ).strip(),
        no_critic_loss=not critic_enabled,
        final_learning_rate=float(
            one_value(best_fits, "final_learning_rate")
        ),
        late_lr_plateau_start_epoch=plateau_start,
        late_lr_plateau_end_epoch=plateau_end,
        model_condition=condition,
        action_entropy_bonus=action_entropy_bonus,
        rnn_units=rnn_units,
    )

    stored_artifact_suffix = one_value(
        best_fits,
        "model_configuration_artifact_suffix",
        default=None,
        required=False,
    )
    generated_artifact_suffix = model_configuration_artifact_suffix(
        config,
        tree,
        lambda_values[0],
        alpha_values[0],
        model_seeds[0],
    )
    if (
        stored_artifact_suffix is not None
        and str(stored_artifact_suffix) != generated_artifact_suffix
    ):
        raise ValueError(
            "Saved fit metadata reconstructs a different model artifact. "
            f"Saved suffix={stored_artifact_suffix!r}; "
            f"reconstructed suffix={generated_artifact_suffix!r}."
        )

    fit_configuration_hash = str(
        one_value(
            best_fits,
            "participant_fit_configuration_hash",
            default=hashlib.sha256(fit_path.stem.encode("utf-8")).hexdigest()[:12],
            required=False,
        )
    )
    reward_nll_scale = float(
        one_value(
            best_fits,
            "reward_nll_scale",
            default=0.5 if condition == "planning" else 1.0,
            required=False,
        )
    )
    weight_path_candidates = recorded_weight_path_candidates(
        best_fits,
        model_seeds,
    )
    output_path = validate_output_path(
        args.output_csv
        if args.output_csv is not None
        else default_output_path(
            fit_path,
            args.trials_per_seed,
            args.simulation_random_seed,
        )
    )

    participant_count = best_fits["subject"].nunique()
    expected_rows = (
        participant_count
        * len(model_seeds)
        * args.trials_per_seed
        * config.time_steps
    )
    print(
        "Starting fitted-simulation regeneration | "
        f"tree={tree.size} | condition={condition} | "
        f"participants={participant_count} | model_seeds={model_seeds} | "
        f"trials_per_seed={args.trials_per_seed} | "
        f"expected_node_rows={expected_rows}",
        flush=True,
    )
    simulations = simulate_fitted_participants(
        config=config,
        best_fits=best_fits,
        model_seeds=model_seeds,
        trials_per_seed=args.trials_per_seed,
        batch_size=args.batch_size,
        simulation_random_seed=args.simulation_random_seed,
        reward_nll_scale=reward_nll_scale,
        fit_configuration_hash=fit_configuration_hash,
        condition=condition,
        weight_path_candidates_by_parameter_seed=weight_path_candidates,
    )
    if len(simulations) != expected_rows:
        raise RuntimeError(
            "Generated fitted-simulation row count does not match the full "
            f"design: observed={len(simulations)}, expected={expected_rows}."
        )
    identity_columns = [
        "subject",
        "simulation_model_seed",
        "simulation_trial",
        "node",
    ]
    if simulations.duplicated(identity_columns).any():
        raise RuntimeError(
            "Generated simulations contain duplicate participant/model-seed/"
            "trial/node rows."
        )
    if set(simulations["simulation_model_seed"].unique()) != set(model_seeds):
        raise RuntimeError("Generated simulations are missing a model seed.")
    trial_counts = (
        simulations[[
            "subject",
            "simulation_model_seed",
            "simulation_trial",
        ]]
        .drop_duplicates()
        .groupby(["subject", "simulation_model_seed"])
        .size()
    )
    if not (trial_counts == args.trials_per_seed).all():
        raise RuntimeError(
            "At least one participant/model-seed cell has the wrong trial count."
        )

    ensure_directory(output_path.parent)
    write_csv_atomically(simulations, output_path)
    print(f"Participant best-fit input: {fit_path}", flush=True)
    print(f"Participants: {participant_count}", flush=True)
    print(
        "Simulation model seeds: "
        + ",".join(str(seed) for seed in model_seeds),
        flush=True,
    )
    print(
        "Trials per participant/model seed: "
        f"{args.trials_per_seed}",
        flush=True,
    )
    print(f"Saved fitted simulations to: {output_path}", flush=True)
    print_local_scp_command(
        output_path,
        ".local_artifacts/participant_fit_audit",
    )


if __name__ == "__main__":
    main()
