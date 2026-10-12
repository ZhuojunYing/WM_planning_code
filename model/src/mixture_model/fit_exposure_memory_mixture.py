"""Fit a discrete Gaussian + uniform recall mixture to exposure responses.

One independent (sigma, guess_rate) fit per subject, using reported rewards,
not binary accuracy. No trained checkpoints or TensorFlow are required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr

CATEGORIES = np.arange(-5, 6, dtype=float)
SIGMA_BOUNDS = (0.2, 5.0)


def memory_probabilities(sigma: float) -> np.ndarray:
    """Rows = true rewards, columns = responses; condition on [-5.5, 5.5].

    This is truncated/renormalized binning, not clipping Gaussian tail mass
    into the endpoint response categories. Use survival probabilities on the
    positive side to avoid subtracting two CDF values rounded to one.
    """
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    distance = CATEGORIES[None, :] - CATEGORIES[:, None]
    lower, upper = (distance - 0.5) / sigma, (distance + 0.5) / sigma
    mass = np.where(lower >= 0, ndtr(-lower) - ndtr(-upper),
                    ndtr(upper) - ndtr(lower))
    return mass / mass.sum(axis=1, keepdims=True)


def mixture_probabilities(sigma: float, guess_rate: float) -> np.ndarray:
    if not 0 <= guess_rate <= 1:
        raise ValueError("guess_rate must be between zero and one")
    return (1 - guess_rate) * memory_probabilities(sigma) + guess_rate / 11


def recall_counts(frame: pd.DataFrame) -> np.ndarray:
    target = pd.to_numeric(frame["node_reward"], errors="raise").to_numpy(float)
    response = pd.to_numeric(frame["estimated_reward"], errors="raise").to_numpy(float)
    if not (np.isin(target, CATEGORIES).all() and np.isin(response, CATEGORIES).all()):
        raise ValueError("True and reported rewards must be integers from -5 to 5")
    counts = np.zeros((11, 11), dtype=np.int64)
    np.add.at(counts, ((target + 5).astype(int), (response + 5).astype(int)), 1)
    return counts


def fit_counts(counts: np.ndarray, sigma_bounds=SIGMA_BOUNDS) -> dict:
    if len(sigma_bounds) != 2 or not np.isfinite(sigma_bounds).all() or not 0 < sigma_bounds[0] < sigma_bounds[1]:
        raise ValueError("Expected finite positive increasing sigma bounds")
    counts = np.asarray(counts)
    if counts.shape != (11, 11) or not np.isfinite(counts).all() or (counts < 0).any() or counts.sum() <= 0:
        raise ValueError("Expected nonnegative, nonempty 11-by-11 counts")

    def objective(parameters):
        probabilities = mixture_probabilities(np.exp(parameters[0]), parameters[1])
        return float(-np.sum(counts * np.log(np.maximum(probabilities, 1e-300))))

    attempts = [minimize(
        objective, [np.log(sigma), guess], method="L-BFGS-B",
        bounds=[tuple(np.log(sigma_bounds)), (0, 1)],
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-6},
    ) for sigma in np.clip((0.3, 1, 3, 10), *sigma_bounds) for guess in (0.05, 0.3, 0.7)]
    valid = [fit for fit in attempts if fit.success and np.isfinite(fit.fun)]
    if not valid:
        raise RuntimeError("No optimization start converged")
    best = min(valid, key=lambda fit: fit.fun)
    sigma, guess = float(np.exp(best.x[0])), float(best.x[1])
    probabilities = mixture_probabilities(sigma, guess)
    n = int(counts.sum())
    sigma_boundary = sigma <= sigma_bounds[0] * 1.001 or sigma >= sigma_bounds[1] * 0.999
    guess_boundary = guess <= 1e-6 or guess >= 1 - 1e-6
    return {
        "n_recall_responses": n,
        "sigma": sigma,
        "precision_inverse_sigma": 1 / sigma,
        "guess_rate": guess,
        "recall_nll": float(best.fun),
        "mean_recall_nll": float(best.fun / n),
        "observed_accuracy": float(np.trace(counts) / n),
        "predicted_accuracy": float(np.dot(counts.sum(axis=1), probabilities.diagonal()) / n),
        "sigma_at_boundary": bool(sigma_boundary),
        "guess_rate_at_boundary": bool(guess_boundary),
        "boundary_fit": bool(sigma_boundary or guess_boundary),
        "precision_unidentified_at_all_guess": bool(guess >= 1 - 1e-6),
        "successful_starts": len(valid),
        "total_starts": len(attempts),
        "optimizer_message": str(best.message),
    }


def fit_file(input_path: Path, output_path: Path) -> None:
    data = pd.read_csv(input_path, dtype={"subject": str})
    # Exp1 preprocessing supplies node_reward; Exp2 retains actual_reward.
    reward_column = "node_reward" if "node_reward" in data.columns else "actual_reward"
    if reward_column not in data.columns:
        raise ValueError("Missing true-reward column: expected node_reward or actual_reward")
    if reward_column == "actual_reward":
        data["node_reward"] = data["actual_reward"]
    required = {"subject", "condition", "node_reward", "estimated_reward"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    exposure = data.loc[data.condition.astype(str).str.strip().str.lower().eq("exposure")]
    selected = exposure.dropna(subset=["subject", "node_reward", "estimated_reward"])
    if selected.empty:
        raise ValueError("No complete exposure recall observations")
    # Validate all observations before starting or writing any results.
    recall_counts(selected)
    rows, failed = [], []
    for subject, frame in selected.groupby("subject", sort=True):
        try:
            result = fit_counts(recall_counts(frame))
            rows.append({"subject": subject, "status": "ok", **result})
            print(f"subject={subject} n={len(frame)} sigma={result['sigma']:.6g} "
                  f"guess_rate={result['guess_rate']:.6g} NLL={result['recall_nll']:.6f} "
                  f"boundary={result['boundary_fit']}", flush=True)
        except RuntimeError as error:
            failed.append(subject)
            rows.append({"subject": subject, "status": "failed", "n_recall_responses": len(frame),
                         "optimizer_message": str(error)})
            print(f"FAILED subject={subject}: {error}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    metadata = {
        "input": str(input_path.resolve()),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "true_reward_column": reward_column,
        "condition": "exposure", "input_rows": len(data), "exposure_rows": len(exposure),
        "complete_recall_rows": len(selected), "dropped_incomplete_exposure_rows": len(exposure) - len(selected),
        "participants": selected.subject.nunique(), "failed_subjects": failed,
        "model": "(1-g)*discretized_truncated_Gaussian(y|true_reward,sigma)+g/11",
        "response_categories": CATEGORIES.astype(int).tolist(),
        "memory_binning": "Integer-centered bins normalized over [-5.5,5.5]; no endpoint tail clipping",
        "sigma_bounds": list(SIGMA_BOUNDS), "guess_rate_bounds": [0, 1],
        "precision_definition": "1/sigma", "recall_nll_scale": 1,
        "uncertainty": "Point estimates only; no confidence intervals or recovery analysis",
        "interpretation": "Guess rate is a mixture weight, not error rate; precision and guessing may be weakly identified",
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {output_path}", flush=True)
    if failed:
        raise RuntimeError(f"Failed fits recorded in output: {failed}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["exp1", "exp2"])
    from project_paths import STORAGE_ROOT, PARAMETER_FIT_DIR, behavior_path
    parser.add_argument("--storage-root", type=Path, default=STORAGE_ROOT)
    parser.add_argument("--behavior-csv", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()
    input_path = args.behavior_csv or (behavior_path(args.experiment) if args.storage_root == STORAGE_ROOT else
        args.storage_root / "data/analysis/processed_data" / f"{args.experiment}_beh_new.csv")
    output_directory = (PARAMETER_FIT_DIR if args.storage_root == STORAGE_ROOT else
                        args.storage_root / "summaries/participant_fits")
    output_path = args.output_csv or output_directory / f"{args.experiment}_exposure_memory_mixture.csv"
    print(f"Input: {input_path}\nOutput: {output_path}", flush=True)
    fit_file(input_path, output_path)


if __name__ == "__main__":
    main()
