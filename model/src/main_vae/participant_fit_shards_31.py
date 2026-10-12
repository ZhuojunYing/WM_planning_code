"""Validate and combine per-participant mixed 31-node fitting results."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import pandas as pd

from project_paths import SUMMARY_DIR, ensure_directory, print_local_scp_command
from .participant_fit_shards import canonical_subject, read_manifest


def aggregate(shard_directory: Path) -> list[Path]:
    participants = read_manifest(shard_directory / "participants.txt")
    prefixes = (
        "participant_parameter_grid_by_tree_31",
        "participant_parameter_grid_31",
        "participant_parameter_fits_31",
    )
    frames = {prefix: [] for prefix in prefixes}
    filenames: dict[str, str] = {}
    reference_cells = None
    configuration = None
    for index, subject in enumerate(participants):
        participant_frames = {}
        for prefix in prefixes:
            paths = list((shard_directory / f"participant_{index}").glob(f"{prefix}_*.csv"))
            if len(paths) != 1:
                raise ValueError(f"Expected one {prefix} CSV for participant {subject}; found {len(paths)}.")
            path = paths[0]
            if prefix in filenames and filenames[prefix] != path.name:
                raise ValueError("Participant shards have different fit filenames.")
            filenames[prefix] = path.name
            frame = pd.read_csv(path, low_memory=False)
            if frame.empty or set(frame["subject"].map(canonical_subject)) != {subject}:
                raise ValueError(f"Unexpected or missing participant in {path}.")
            identity = frame[["participant_fit_configuration_hash", "participant_fit_artifact_suffix"]].drop_duplicates()
            if len(identity) != 1:
                raise ValueError(f"Inconsistent configuration in {path}.")
            values = tuple(identity.iloc[0])
            if configuration is not None and values != configuration:
                raise ValueError("Participant shards have different configurations.")
            configuration = values
            keys = ["subject"]
            if prefix != prefixes[2]:
                keys += ["lambda", "alpha"]
            if prefix == prefixes[0]:
                keys += ["tree_type"]
            if frame.duplicated(keys).any():
                raise ValueError(f"Duplicate fit rows in {path}.")
            if frames[prefix] and list(frame.columns) != list(frames[prefix][0].columns):
                raise ValueError("Participant shards have different column schemas.")
            participant_frames[prefix] = frame
            frames[prefix].append(frame)
        grid = participant_frames[prefixes[1]]
        cells = set(zip(grid["lambda"], grid["alpha"]))
        if reference_cells is not None and cells != reference_cells:
            raise ValueError(f"Parameter grid differs for participant {subject}.")
        reference_cells = cells
        tree_grid = participant_frames[prefixes[0]]
        if set(zip(tree_grid["lambda"], tree_grid["alpha"])) != cells:
            raise ValueError(f"Per-tree grid is incomplete for participant {subject}.")
        expected_trees = set(tree_grid["tree_type"])
        if any(set(group["tree_type"]) != expected_trees
               for _, group in tree_grid.groupby(["lambda", "alpha"])):
            raise ValueError(f"Missing tree contributions for participant {subject}.")
    # Validate every shard before publishing any combined table.
    destination = ensure_directory(SUMMARY_DIR / "participant_fits")
    outputs = []
    for prefix in prefixes:
        output = destination / filenames[prefix]
        combined = pd.concat(frames[prefix], ignore_index=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=destination, suffix=".csv", delete=False) as handle:
            temporary_path = Path(handle.name)
            combined.to_csv(handle, index=False)
        temporary_path.replace(output)
        outputs.append(output)
        print(f"Saved {len(combined)} rows: {output}", flush=True)
    print(f"Participants aggregated: {len(participants)}", flush=True)
    print_local_scp_command(outputs, ".local_artifacts/participant_fit_audit")
    print("Generate fitted simulations separately using submit_31_fitted_simulations.sh --fit-csv and the best-fit path above.", flush=True)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_directory", type=Path)
    aggregate(parser.parse_args().shard_directory)
