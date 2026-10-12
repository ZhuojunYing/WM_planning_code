"""Fit the single simultaneous-observation channel model to participants."""

from ..participant_fitting import run_family


if __name__ == "__main__":
    run_family("simultaneous_channel")
