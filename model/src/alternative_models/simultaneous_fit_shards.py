"""TF-free validation and storage for parallel simultaneous participant fits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from ..main_vae.participant_fit_shards import (
    canonical_subject, default_behavior_path, read_manifest, write_manifest,
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_identity(args, behavior_path: Path) -> str:
    ignored = {"participant_index", "shard_directory", "aggregate_shards", "behavior_csv"}
    values = {key: value for key, value in vars(args).items() if key not in ignored}
    values["behavior_sha256"] = file_hash(behavior_path)
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()


def write_shard(directory: Path, index: int, subject, identity: str,
                grid: pd.DataFrame, diagnostics: pd.DataFrame | None) -> None:
    """Publish a completion marker last; incomplete tasks are never accepted."""
    target = directory / f"participant_{index}"
    target.mkdir(parents=True, exist_ok=True)
    marker = target / "complete.json"
    marker.unlink(missing_ok=True)
    hashes = {}
    files = [("grid.csv", grid)]
    if diagnostics is not None:
        files.append(("diagnostics.csv", diagnostics))
    for name, frame in files:
        with tempfile.NamedTemporaryFile(mode="w", dir=target, delete=False) as handle:
            temporary = Path(handle.name)
            frame.to_csv(handle, index=False)
        temporary.replace(target / name)
        hashes[name] = file_hash(target / name)
    payload = {"subject": canonical_subject(subject), "identity": identity, "files": hashes}
    temporary = target / "complete.json.tmp"
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(marker)


def read_shards(directory: Path, subjects: list, identity: str,
                cells: list, assignments: pd.DataFrame, *,
                require_diagnostics: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate the requested grid and seed, not merely agreement between shards."""
    expected_subjects = [canonical_subject(value) for value in subjects]
    if read_manifest(directory / "participants.txt") != expected_subjects:
        raise ValueError("Participant manifest differs from the current behavioral cohort.")
    seeds = dict(zip(assignments["subject"].map(canonical_subject), assignments["nll_model_seed"]))
    expected_cells = {(one, two): (lam, alpha) for one, two, lam, alpha, _ in cells}
    frames = {"grid.csv": []}
    if require_diagnostics:
        frames["diagnostics.csv"] = []
    for index, subject in enumerate(expected_subjects):
        target = directory / f"participant_{index}"
        marker = json.loads((target / "complete.json").read_text())
        if marker["identity"] != identity or marker["subject"] != subject:
            raise ValueError(f"Wrong run configuration or participant in {target}.")
        for name in frames:
            path = target / name
            if file_hash(path) != marker["files"][name]:
                raise ValueError(f"Incomplete or modified shard: {path}.")
            frame = pd.read_csv(path, dtype={"subject": str}, float_precision="round_trip")
            if frame.empty or set(frame["subject"].map(canonical_subject)) != {subject}:
                raise ValueError(f"Wrong participant or empty shard: {path}.")
            if set(frame["nll_model_seed"]) != {seeds[subject]}:
                raise ValueError(f"Wrong assigned seed in {path}.")
            keys = ["parameter_one_order", "parameter_two_order"]
            observed = set(map(tuple, frame[keys].to_numpy()))
            if observed != set(expected_cells):
                raise ValueError(f"Incomplete requested parameter grid in {path}.")
            if name == "grid.csv" and frame.duplicated(keys).any():
                raise ValueError(f"Duplicate parameter cells in {path}.")
            for key, group in frame.groupby(keys):
                lam, alpha = expected_cells[key]
                if not (np.allclose(group["lambda"], lam, rtol=1e-12, atol=0)
                        and np.allclose(group["alpha"], alpha, rtol=1e-12, atol=0)):
                    raise ValueError(f"Wrong parameter values in {path}.")
            if frames[name] and list(frame.columns) != list(frames[name][0].columns):
                raise ValueError(f"Inconsistent schema in {path}.")
            frames[name].append(frame)
        if not require_diagnostics:
            print(f"Validated independent participant {index + 1}/{len(subjects)}: {subject}", flush=True)
            continue
        # Ensure raw diagnostics reproduce each participant's likelihood exactly.
        grid = frames["grid.csv"][-1]
        raw = frames["diagnostics.csv"][-1]
        counts = raw.groupby(keys)["path_nll"].agg(["sum", "count"])
        checked = grid.merge(counts, left_on=keys, right_index=True, validate="one_to_one")
        if not np.allclose(checked["path_nll"], checked["sum"], rtol=1e-10, atol=1e-10):
            raise ValueError(f"Diagnostic NLL does not match grid in {target}.")
        if not np.array_equal(checked["path_diagnostic_choice_count"], checked["count"]):
            raise ValueError(f"Diagnostic trial count does not match grid in {target}.")
        print(f"Validated simultaneous participant {index + 1}/{len(subjects)}: {subject}", flush=True)
    grid = pd.concat(frames["grid.csv"], ignore_index=True)
    raw = (pd.concat(frames["diagnostics.csv"], ignore_index=True)
           if require_diagnostics else pd.DataFrame())
    # Restore the source ID types for the downstream simulation merge.
    original_ids = dict(zip(expected_subjects, subjects))
    for frame in (grid, raw):
        if not frame.empty:
            frame["subject"] = frame["subject"].map(canonical_subject).map(original_ids)
    return grid, raw


def prepare(tree: str, directory: Path, behavior_csv: Path | None) -> None:
    source = behavior_csv or default_behavior_path(tree)
    snapshot = directory / "behavior.csv"
    shutil.copyfile(source, snapshot)
    write_manifest(argparse.Namespace(tree=tree, condition="planning",
                                     behavior_csv=snapshot, output=directory / "participants.txt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tree", choices=["7", "13"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--behavior-csv", type=Path)
    options = parser.parse_args()
    prepare(options.tree, options.directory, options.behavior_csv)
