"""TF-free independent-channel fit shards; reuse common grid/seed validation."""
import hashlib

import numpy as np

from . import simultaneous_fit_shards as shared


def run_identity(args, behavior_path):
    # Distinguish model families even if all CLI settings match.
    identity = shared.run_identity(args, behavior_path)
    return hashlib.sha256(("independent_channel:" + identity).encode()).hexdigest()


def validate_likelihoods(grid):
    columns = ["path_nll", "reconstruction_nll", "scaled_reconstruction_nll",
               "total_nll", "reward_nll_scale"]
    if not np.isfinite(grid[columns].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite independent-channel likelihoods.")
    if (grid[columns] < 0).any().any() or (grid.reward_nll_scale <= 0).any():
        raise ValueError("Invalid independent-channel likelihoods or reward scale.")
    if not np.allclose(grid.scaled_reconstruction_nll,
                       grid.reward_nll_scale * grid.reconstruction_nll, rtol=1e-12, atol=1e-12):
        raise ValueError("Scaled reconstruction NLL does not match reward scale.")
    if not np.allclose(grid.total_nll, grid.path_nll + grid.scaled_reconstruction_nll,
                       rtol=1e-12, atol=1e-12):
        raise ValueError("Total NLL does not match its components.")


def write_shard(directory, index, subject, identity, grid, diagnostics=None):
    validate_likelihoods(grid)
    shared.write_shard(directory, index, subject, identity, grid, None)


def read_shards(directory, subjects, identity, cells, assignments):
    grid, empty = shared.read_shards(directory, subjects, identity, cells, assignments,
                                     require_diagnostics=False)
    validate_likelihoods(grid)
    for one, two, _, _, rho in cells:
        selected = grid.loc[(grid.parameter_one_order == one) & (grid.parameter_two_order == two)]
        if not np.allclose(selected.rho, rho, rtol=1e-12, atol=0, equal_nan=True):
            raise ValueError("Wrong rho values in independent-channel shard.")
    return grid, empty
