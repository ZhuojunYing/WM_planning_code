"""Central filesystem locations for generated project artifacts.

Importing this module only resolves paths; it never creates directories.
"""

from __future__ import annotations

import hashlib
import os
import shlex
from pathlib import Path
from typing import Sequence


_CONFIG_ROOT = Path(__file__).resolve().parent
_CONFIG = dict(line.split(":", 1) for line in
               (_CONFIG_ROOT / "project_paths.conf").read_text().splitlines()
               if line.strip() and not line.lstrip().startswith("#"))
_CONFIG = {key: value.strip() for key, value in _CONFIG.items()}

def _absolute(value: str | Path, base: Path) -> Path:
    value = Path(value).expanduser()
    return (value if value.is_absolute() else base / value).resolve()

REPO_ROOT = _absolute(os.environ.get("WM_PLANNING_ROOT", _CONFIG["ProjectRoot"]), _CONFIG_ROOT)

def project_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)

def _setting(key: str) -> Path:
    return _absolute(_CONFIG[key], REPO_ROOT)


def get_storage_root() -> Path:
    """Return the configured root for generated artifacts."""
    return _absolute(os.environ.get("WM_PLANNING_STORAGE_ROOT",
        os.environ.get("WM_PLANNING_COMPARISON_STORAGE_ROOT", _CONFIG["StorageRoot"])), REPO_ROOT)


STORAGE_ROOT = get_storage_root()

MODEL_DIR = STORAGE_ROOT / "models"
WEIGHT_DIR = MODEL_DIR / "weights"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

DATA_DIR = STORAGE_ROOT / "data"
SIMULATION_DIR = STORAGE_ROOT / "simulations"

PLOT_DIR = STORAGE_ROOT / "plots"

LOG_DIR = STORAGE_ROOT / "logs"
TRAINING_LOG_DIR = LOG_DIR / "training"
ANALYSIS_LOG_DIR = _setting("AnalysisResults") / "logs"
SLURM_LOG_DIR = LOG_DIR / "slurm"

SUMMARY_DIR = STORAGE_ROOT / "summaries"
RESULT_DIR = STORAGE_ROOT / "results"
CACHE_DIR = STORAGE_ROOT / "caches"
TEMP_DIR = STORAGE_ROOT / "tmp"
ANALYSIS_DATA_DIR = _setting("AnalysisData")
RAW_DATA_DIR = _setting("RawData")
PROCESSED_DATA_DIR = _setting("ProcessedData")
SIMULATION_INPUT_DIR = _setting("SimulationInputs")
PARAMETER_FIT_DIR = _setting("ParameterFits")
RECOVERY_INPUT_DIR = _setting("RecoveryInputs")
ANALYSIS_PLOT_DIR = _setting("Figures")
ANALYSIS_SUMMARY_DIR = _setting("AnalysisResults")
STATS_ROOT = ANALYSIS_SUMMARY_DIR / "stats" / "fitted"
TABLE_DIR = _setting("Tables")
INTEXT_DIR = _setting("Intext")
LEGACY_STATS_DIR = _setting("LegacyStats")

def behavior_path(experiment: str) -> Path:
    suffix = "_beh_new.csv" if experiment in ("exp1", "exp2") else "_beh.csv"
    return PROCESSED_DATA_DIR / (experiment + suffix)


def ensure_directory(path: str | Path) -> Path:
    """Create a directory, including missing parents, and return its path."""
    directory = Path(path).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_parent_directory(path: str | Path) -> Path:
    """Create the parent directory for a file path and return the file path."""
    output_path = Path(path).expanduser()
    ensure_directory(output_path.parent)
    return output_path


def job_temporary_path(
    output_path: str | Path,
    *,
    label: str,
    suffix: str,
) -> Path:
    """Return a short, job-specific temporary path beside an output file.

    Artifact filenames intentionally encode their model configuration and can
    approach a filesystem's per-component length limit. Temporary files must
    not repeat that complete name and then append a job identifier. A digest
    keeps concurrent writes distinct while leaving the final artifact name
    descriptive.
    """
    target_path = Path(output_path).expanduser()
    safe_label = "".join(
        character if character.isalnum() or character in "_-" else "_"
        for character in label
    ).strip("_")
    if not safe_label:
        raise ValueError("Temporary artifact label cannot be empty.")
    if not suffix.startswith("."):
        raise ValueError("Temporary artifact suffix must begin with '.'.")

    run_identifier = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    safe_run_identifier = "".join(
        character if character.isalnum() or character in "_-" else "_"
        for character in run_identifier
    )
    target_digest = hashlib.sha256(
        target_path.name.encode("utf-8")
    ).hexdigest()[:16]
    return target_path.parent / (
        f".{safe_label}_{target_digest}_{safe_run_identifier}{suffix}"
    )


def ensure_output_directories() -> None:
    """Create the complete generated-artifact directory layout."""
    for directory in (
        WEIGHT_DIR,
        CHECKPOINT_DIR,
        DATA_DIR,
        SIMULATION_DIR,
        PLOT_DIR,
        TRAINING_LOG_DIR,
        ANALYSIS_LOG_DIR,
        SLURM_LOG_DIR,
        SUMMARY_DIR,
        RESULT_DIR,
        CACHE_DIR,
        TEMP_DIR,
    ):
        ensure_directory(directory)


def require_existing_path(
    path: str | Path,
    description: str | None = None,
) -> Path:
    """Return an existing path or raise a descriptive FileNotFoundError."""
    required_path = Path(path).expanduser()
    if not required_path.exists():
        label = description or "Required path"
        raise FileNotFoundError(f"{label} does not exist: {required_path}")
    return required_path


def resolve_directory_override(
    path_override: str | Path | None,
    default_directory: str | Path,
) -> Path:
    """Resolve a CLI directory override, with ``default`` selecting the default."""
    if path_override is None:
        return Path(default_directory)

    override_text = str(path_override).strip()
    if not override_text or override_text.lower() == "default":
        return Path(default_directory)

    return Path(override_text).expanduser()


def format_local_scp_command(
    remote_paths: str | Path | Sequence[str | Path],
    local_directory: str | Path,
    *,
    recursive: bool = False,
    remote_host: str = "dtn",
) -> str:
    """Return a copy-pasteable SCP command for the user's local computer."""
    if isinstance(remote_paths, (str, Path)):
        paths = (Path(remote_paths),)
    else:
        paths = tuple(Path(path) for path in remote_paths)
    if not paths:
        raise ValueError("At least one remote path is required.")

    local_text = str(local_directory).rstrip("/")
    if not local_text:
        raise ValueError("The local destination directory cannot be empty.")
    mkdir_command = f"mkdir -p {shlex.quote(local_text)}"
    flags = "-pr" if recursive else "-p"
    remote_arguments = tuple(
        shlex.quote(f"{remote_host}:{path}") for path in paths
    )
    local_argument = shlex.quote(f"{local_text}/")
    if len(remote_arguments) == 1:
        scp_command = (
            f"scp {flags} {remote_arguments[0]} {local_argument}"
        )
    else:
        scp_lines = [f"scp {flags} \\"]
        scp_lines.extend(
            f"  {argument} \\" for argument in remote_arguments
        )
        scp_lines.append(f"  {local_argument}")
        scp_command = "\n".join(scp_lines)
    return "\n".join((mkdir_command, scp_command))


def print_local_scp_command(
    remote_paths: str | Path | Sequence[str | Path],
    local_directory: str | Path,
    *,
    recursive: bool = False,
    remote_host: str = "dtn",
) -> None:
    """Print an SCP command intended to run from the local repository root."""
    print("\nRun this on your local computer from the repository root:")
    print(
        format_local_scp_command(
            remote_paths,
            local_directory,
            recursive=recursive,
            remote_host=remote_host,
        ),
        flush=True,
    )
