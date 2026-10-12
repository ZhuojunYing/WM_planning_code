"""Create participant manifests and aggregate parallel Main-VAE fit shards."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from project_paths import behavior_path as behavior_input_path

from project_paths import (
    DATA_DIR,
    SIMULATION_DIR,
    SUMMARY_DIR,
    ensure_directory,
    print_local_scp_command,
    require_existing_path,
)


def canonical_subject(value: object) -> str:
    """Return a stable textual participant identifier."""
    text = str(value).strip()
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer():
        return str(int(numeric))
    return text


def default_behavior_path(tree: str) -> Path:
    """Return the centralized behavioral table for a supported tree."""
    if tree == "7":
        return behavior_input_path("exp1")
    if tree == "13":
        return behavior_input_path("exp2")
    if tree == "31":
        return behavior_input_path("exp3")
    raise ValueError(
        "Participant manifests support tree 7, 13, or 31."
    )


def write_manifest(args: argparse.Namespace) -> None:
    """Write one participant identifier per line in behavioral-table order."""
    behavior_path = (
        args.behavior_csv.expanduser()
        if args.behavior_csv is not None
        else default_behavior_path(args.tree)
    )
    require_existing_path(behavior_path, "Behavioral CSV")
    participants: list[str] = []
    seen: set[str] = set()
    with behavior_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Behavioral CSV has no header.")
        required_columns = {"subject", "condition"}
        missing = required_columns.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                "Behavioral CSV is missing columns: "
                + ", ".join(sorted(missing))
            )
        for row in reader:
            if str(row["condition"]).strip().lower() != args.condition:
                continue
            subject = canonical_subject(row["subject"])
            if subject and subject not in seen:
                participants.append(subject)
                seen.add(subject)
    if not participants:
        raise ValueError(
            f"No {args.condition} participants were found in {behavior_path}."
        )
    ensure_directory(args.output.parent)
    args.output.write_text(
        "".join(f"{subject}\n" for subject in participants),
        encoding="utf-8",
    )
    print(f"Behavioral input: {behavior_path}")
    print(f"Condition: {args.condition}")
    print(f"Participants: {len(participants)}")
    print(f"Manifest: {args.output}")


def read_manifest(path: Path) -> list[str]:
    """Read and validate a participant manifest."""
    require_existing_path(path, "Participant manifest")
    participants = [
        canonical_subject(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not participants:
        raise ValueError("Participant manifest is empty.")
    if len(set(participants)) != len(participants):
        raise ValueError("Participant manifest contains duplicate identifiers.")
    return participants


def read_shards(paths: list[Path], label: str):
    """Read a nonempty collection of CSV shards."""
    import pandas as pd  # pylint: disable=import-outside-toplevel

    if not paths:
        raise FileNotFoundError(f"No {label} shards were found.")
    frames = [pd.read_csv(path, low_memory=False) for path in paths]
    if any(frame.empty for frame in frames):
        raise ValueError(f"At least one {label} shard is empty.")
    return pd.concat(frames, ignore_index=True)


def one_value(frame, column: str) -> object:
    """Return one nonmissing metadata value."""
    if column not in frame.columns:
        raise ValueError(f"Fit shards are missing metadata column {column!r}.")
    values = frame[column].dropna().drop_duplicates().tolist()
    if len(values) != 1:
        raise ValueError(
            f"Expected one {column} value across shards; found {len(values)}."
        )
    return values[0]


def aggregate_shards(args: argparse.Namespace) -> None:
    """Combine completed participant shards and generate fitted simulations."""
    import pandas as pd  # pylint: disable=import-outside-toplevel

    from .fit_participant_parameters import (  # pylint: disable=import-outside-toplevel
        resolve_tree,
        run_fitted_plots,
        write_csv_atomically,
    )
    from .simulate_fitted_participants import (  # pylint: disable=import-outside-toplevel
        default_output_path,
    )

    shard_directory = args.shard_directory.expanduser()
    require_existing_path(shard_directory, "Participant-fit shard directory")
    expected_participants = read_manifest(args.manifest.expanduser())
    grid_paths = sorted(shard_directory.glob("grid_subject_*.csv"))
    fit_paths = sorted(shard_directory.glob("fit_subject_*.csv"))
    if len(grid_paths) != len(expected_participants):
        raise ValueError(
            "Grid shard count does not match the manifest: "
            f"{len(grid_paths)} versus {len(expected_participants)}."
        )
    if len(fit_paths) != len(expected_participants):
        raise ValueError(
            "Best-fit shard count does not match the manifest: "
            f"{len(fit_paths)} versus {len(expected_participants)}."
        )

    full_grid = read_shards(grid_paths, "grid")
    best_fits = read_shards(fit_paths, "best-fit")
    if best_fits["subject"].duplicated().any():
        raise ValueError("Best-fit shards contain duplicate participants.")
    observed_participants = {
        canonical_subject(subject) for subject in best_fits["subject"]
    }
    if observed_participants != set(expected_participants):
        missing = sorted(set(expected_participants) - observed_participants)
        extra = sorted(observed_participants - set(expected_participants))
        raise ValueError(
            f"Shard participants do not match the manifest; missing={missing}, "
            f"extra={extra}."
        )
    grid_participants = {
        canonical_subject(subject) for subject in full_grid["subject"].unique()
    }
    if grid_participants != observed_participants:
        raise ValueError("Grid and best-fit shards contain different participants.")

    participant_order = {
        subject: index for index, subject in enumerate(expected_participants)
    }
    for frame in (full_grid, best_fits):
        frame["_participant_order"] = frame["subject"].map(
            lambda subject: participant_order[canonical_subject(subject)]
        )
        frame.sort_values("_participant_order", kind="stable", inplace=True)
        frame.drop(columns="_participant_order", inplace=True)
        frame.reset_index(drop=True, inplace=True)

    fit_suffix = str(one_value(best_fits, "participant_fit_artifact_suffix"))
    fit_hash = str(one_value(best_fits, "participant_fit_configuration_hash"))
    if fit_hash != str(one_value(full_grid, "participant_fit_configuration_hash")):
        raise ValueError("Grid and best-fit shards have different fit hashes.")
    condition = str(one_value(best_fits, "fit_condition")).strip().lower()
    if condition not in {"planning", "exposure"}:
        raise ValueError(f"Unsupported fit condition {condition!r}.")

    safe_tree = args.tree
    output_directory = ensure_directory(SUMMARY_DIR / "participant_fits")
    grid_path = output_directory / (
        f"participant_parameter_grid_{safe_tree}{fit_suffix}.csv"
    )
    fit_path = output_directory / (
        f"participant_parameter_fits_{safe_tree}{fit_suffix}.csv"
    )
    write_csv_atomically(full_grid, grid_path)
    write_csv_atomically(best_fits, fit_path)

    simulation_path = default_output_path(
        fit_path,
        args.simulation_trials_per_seed,
        args.simulation_random_seed,
    )
    simulation_command = [
        sys.executable,
        "-u",
        "-m",
        "model.src.main_vae.simulate_fitted_participants",
        args.tree,
        "--fit-csv",
        str(fit_path),
        "--trials-per-seed",
        str(args.simulation_trials_per_seed),
        "--batch-size",
        str(args.batch_size),
        "--simulation-random-seed",
        str(args.simulation_random_seed),
        "--output-csv",
        str(simulation_path),
    ]
    print(
        "Participant fit shards combined; generating fitted simulations.",
        flush=True,
    )
    subprocess.run(simulation_command, check=True)
    if not args.skip_fitted_plots:
        tree = resolve_tree(args.tree)
        run_fitted_plots(
            tree,
            grid_path,
            fit_path,
            simulation_path,
            condition,
        )

    print(f"Participants aggregated: {len(expected_participants)}", flush=True)
    print(f"Participant-fit configuration hash: {fit_hash}", flush=True)
    print_local_scp_command(
        (grid_path, fit_path, simulation_path),
        ".local_artifacts/participant_fit_audit",
    )


def parse_args() -> argparse.Namespace:
    """Parse manifest or aggregation arguments."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--tree", choices=("7", "13", "31"), required=True)
    manifest.add_argument(
        "--condition",
        choices=("planning", "exposure"),
        default="planning",
    )
    manifest.add_argument("--behavior-csv", type=Path, default=None)
    manifest.add_argument("--output", type=Path, required=True)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--tree", choices=("7", "13"), required=True)
    aggregate.add_argument("--shard-directory", type=Path, required=True)
    aggregate.add_argument("--manifest", type=Path, required=True)
    aggregate.add_argument("--simulation-trials-per-seed", type=int, default=100)
    aggregate.add_argument("--simulation-random-seed", type=int, default=2027)
    aggregate.add_argument("--batch-size", type=int, default=128)
    aggregate.add_argument("--skip-fitted-plots", action="store_true")
    args = parser.parse_args()
    for name in (
        "simulation_trials_per_seed",
        "batch_size",
    ):
        if hasattr(args, name) and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if (
        hasattr(args, "simulation_random_seed")
        and args.simulation_random_seed < 0
    ):
        parser.error("--simulation-random-seed must be non-negative.")
    return args


def main() -> None:
    """Run the selected shard utility."""
    args = parse_args()
    if args.command == "manifest":
        write_manifest(args)
    else:
        aggregate_shards(args)


if __name__ == "__main__":
    main()
