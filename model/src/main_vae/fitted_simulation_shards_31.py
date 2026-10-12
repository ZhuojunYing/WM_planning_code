"""Snapshot 31-node fits and validate/assemble simulation shards without TensorFlow."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from project_paths import SIMULATION_DIR, print_local_scp_command

TREE_TYPES = ("deep_breadth", "deep_depth", "wide_breadth", "wide_depth")


def read_fits(path):
    fits = pd.read_csv(path, dtype={"subject": str})
    required = {"subject", "best_lambda", "best_alpha", "candidate_model_seeds",
                "participant_fit_configuration_hash", "fit_condition"}
    if not required.issubset(fits.columns) or fits.empty:
        raise ValueError("Invalid or empty 31-node best-fit CSV.")
    if fits.subject.isna().any() or fits.subject.duplicated().any():
        raise ValueError("Missing or duplicate participant identities.")
    if set(fits.fit_condition) != {"planning"}:
        raise ValueError("31-node simulations require planning fits.")
    return fits


def create_run(args):
    source = args.fit_csv.resolve(strict=True)
    if not source.name.startswith("participant_parameter_fits_31_"):
        raise ValueError("Expected participant_parameter_fits_31_*.csv.")
    fits = read_fits(source)
    seed_text = args.simulation_seeds
    if seed_text is None:
        values = fits.candidate_model_seeds.dropna().unique()
        if len(values) != 1:
            raise ValueError("Fit CSV must specify one candidate seed list.")
        seed_text = str(values[0])
    seeds = [int(s.strip()) for s in seed_text.split(",")]
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        raise ValueError("Invalid simulation seeds.")
    if args.trials_per_seed_tree < 1 or args.batch_size < 1 or args.simulation_random_seed < 0:
        raise ValueError("Invalid trial count, batch size, or simulation seed.")
    name = source.stem.replace("participant_parameter_fits_31", "participant_fitted_simulations_31", 1)
    output = args.output_csv or (SIMULATION_DIR / "participant_fits" /
        f"{name}_n{args.trials_per_seed_tree}_per_seed_tree_simseed{args.simulation_random_seed}.csv")
    output = output.resolve()
    output.relative_to(SIMULATION_DIR.resolve())
    if output.suffix != ".csv":
        raise ValueError("Output must be a CSV inside the simulation root.")
    root = SIMULATION_DIR / "participant_fits" / "shards"
    root.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="sim31_", dir=root))
    snapshot = run / source.name
    shutil.copy2(source, snapshot)
    # Verify the snapshot, not a subsequently updated input table.
    snapshot_fits = read_fits(snapshot)
    if not fits.equals(snapshot_fits):
        raise ValueError("The fit CSV changed while preparing the simulation run; please retry.")
    fits = snapshot_fits
    metadata = dict(
        fit_csv=str(snapshot), original_fit_csv=str(source),
        fit_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        subjects=fits.subject.tolist(), output_csv=str(output),
        simulation_seeds=seeds, trials_per_seed_tree=args.trials_per_seed_tree,
        batch_size=args.batch_size, simulation_random_seed=args.simulation_random_seed,
    )
    (run / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return run


def load_run(run):
    metadata = json.loads((run / "run.json").read_text())
    path = Path(metadata["fit_csv"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["fit_sha256"]:
        raise ValueError("The snapshotted fit CSV has changed.")
    return metadata


def validate_shard(frame, metadata, fit, index):
    subject = metadata["subjects"][index]
    required = {"subject", "tree", "tree_type", "simulation_model_seed", "simulation_trial", "node",
                "best_lambda", "best_alpha", "participant_fit_configuration_hash",
                "simulation_random_seed", "simulation_trials_per_seed_tree", "simulation_participant_index"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Shard {index} is missing required columns.")
    n = metadata["trials_per_seed_tree"]
    seeds = metadata["simulation_seeds"]
    if len(frame) != 4 * len(seeds) * n * 30 or set(frame.subject.astype(str)) != {subject}:
        raise ValueError(f"Wrong participant or row count in shard {index}.")
    expected = {
        "tree": 31, "best_lambda": fit.best_lambda, "best_alpha": fit.best_alpha,
        "participant_fit_configuration_hash": fit.participant_fit_configuration_hash,
        "simulation_random_seed": metadata["simulation_random_seed"],
        "simulation_trials_per_seed_tree": n, "simulation_participant_index": index,
    }
    for column, value in expected.items():
        valid = (np.isclose(pd.to_numeric(frame[column]), float(value), rtol=1e-10, atol=0)
                 if column in {"best_lambda", "best_alpha"} else frame[column].eq(value))
        if not np.all(valid):
            raise ValueError(f"Shard {index} has inconsistent {column}.")
    keys = ["tree_type", "simulation_model_seed", "simulation_trial", "node"]
    expected_keys = pd.MultiIndex.from_product([TREE_TYPES, seeds, range(n), range(1, 31)], names=keys)
    actual = pd.MultiIndex.from_frame(frame[keys])
    if actual.has_duplicates or len(expected_keys.difference(actual)):
        raise ValueError(f"Shard {index} has missing or duplicate trial/node identities.")


def aggregate(run):
    metadata = load_run(run)
    fits = read_fits(Path(metadata["fit_csv"]))
    if fits.subject.tolist() != metadata["subjects"]:
        raise ValueError("Participant manifest does not match the fit snapshot.")
    output = Path(metadata["output_csv"])
    output.resolve().relative_to(SIMULATION_DIR.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, suffix=".csv", delete=False) as handle:
            temporary = Path(handle.name)
            columns = None
            for index, fit in fits.iterrows():
                frame = pd.read_csv(run / f"participant_{index}.csv", dtype={"subject": str})
                validate_shard(frame, metadata, fit, index)
                if columns is not None and columns != list(frame.columns):
                    raise ValueError("Simulation shards have different schemas.")
                columns = list(frame.columns)
                frame.to_csv(handle, index=False, header=index == 0)
                print(f"Aggregated participant {index + 1}/{len(fits)}: {fit.subject}", flush=True)
        temporary.replace(output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print(f"Saved fitted simulations to: {output}", flush=True)
    print_local_scp_command(output, ".local_artifacts/participant_fit_audit")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--fit-csv", type=Path, required=True)
    prepare.add_argument("--simulation-seeds")
    prepare.add_argument("--trials-per-seed-tree", type=int, default=25)
    prepare.add_argument("--batch-size", type=int, default=128)
    prepare.add_argument("--simulation-random-seed", type=int, default=2027)
    prepare.add_argument("--output-csv", type=Path)
    for name in ("count", "worker-args", "aggregate"):
        p = commands.add_parser(name)
        p.add_argument("run", type=Path)
        if name == "worker-args":
            p.add_argument("index", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        print(create_run(args))
    elif args.command == "aggregate":
        aggregate(args.run)
    else:
        metadata = load_run(args.run)
        if args.command == "count":
            print(len(metadata["subjects"]))
        else:
            if not 0 <= args.index < len(metadata["subjects"]):
                raise ValueError("Participant index is outside the manifest.")
            worker_args = ["--fit-csv", metadata["fit_csv"], "--participant-index", str(args.index),
                "--simulation-seeds", ",".join(map(str, metadata["simulation_seeds"])),
                "--trials-per-seed-tree", str(metadata["trials_per_seed_tree"]),
                "--batch-size", str(metadata["batch_size"]),
                "--simulation-random-seed", str(metadata["simulation_random_seed"]),
                "--output-csv", str(args.run / f"participant_{args.index}.csv")]
            print("\n".join(worker_args))


if __name__ == "__main__":
    main()
