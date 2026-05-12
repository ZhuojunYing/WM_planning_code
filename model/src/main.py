"""Run model training and simulation from command-line configuration."""

import os

import config
from model import VariationalRNN, build_decoder, build_encoder
from simulate import run_simulation
from train import train_model


def main() -> None:
    """Create configured models and run training followed by simulation.

    Args:
        None.

    Returns:
        None.
    """
    os.makedirs(config.model_dir_name, exist_ok=True)

    if config.train_mode == "train":
        for lambda_ in config.lambda_values:
            for alpha in config.alpha_values:
                if config.time_steps == 30:
                    model_name = (
                        f"lambda_{lambda_}_alpha_{alpha}_"
                        f"seed_{config.seed}_"
                        f"{config.tree_type}"
                    )
                else:
                    model_name = (
                        f"lambda_{lambda_}_alpha_{alpha}_"
                        f"seed_{config.seed}"
                    )

                encoder = build_encoder(
                    config.rnn_units * 2,
                    config.latent_dim,
                    config.rnn_units,
                )
                decoder = build_decoder(
                    config.latent_dim,
                    2 * config.rnn_units,
                    config.rnn_units,
                )

                vrnn_model = VariationalRNN(
                    encoder=encoder,
                    decoder=decoder,
                    rnn_units=config.rnn_units,
                    latent_dim=config.latent_dim,
                    time_steps=config.time_steps,
                    num_paths=config.num_paths,
                    index_path_map=config.index_path_map,
                    path_map=config.path_map,
                    path_cov_mat=config.path_cov_mat,
                    alpha=alpha,
                    lambda_=lambda_,
                    tree_type=config.tree_type,
                    reward_normalization_constant=(
                        config.reward_normalization_constant
                    ),
                )

                train_model(
                    model=vrnn_model,
                    epochs=config.epochs,
                    trials_per_epoch=config.trials_per_epoch,
                    batch_size=config.batch_size,
                    time_steps=config.time_steps,
                    dir_name=config.model_dir_name,
                    model_name=model_name,
                )

    run_simulation(config)


if __name__ == "__main__":
    main()
