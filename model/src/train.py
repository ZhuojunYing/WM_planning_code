"""Training utilities for the variational recurrent planning model."""

from __future__ import annotations

import os

import tensorflow as tf

import helper


@tf.function
def train_step(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    current_alpha: tf.Tensor,
    current_kl_weight: tf.Tensor,
    current_critic_coef: tf.Tensor,
    input_data: tf.Tensor,
    clip_value: float = 10.0,
) -> tuple[tf.Tensor, ...]:
    """Run one optimization step.

    Args:
        model: Variational recurrent model to train.
        optimizer: Optimizer used to apply gradients.
        current_alpha: Current reconstruction-loss schedule value.
        current_kl_weight: Current KL-loss schedule value.
        current_critic_coef: Current critic-loss schedule value.
        input_data: Batch of reward sequences.
        clip_value: Retained for compatibility with earlier training calls.

    Returns:
        Tuple of loss values used for early stopping.
    """
    first_decoder_params = (
        model.decoder.trainable_variables
        + model.lstm_cell.trainable_variables
        + model.encoder.trainable_variables
        + [model.prior_mu]
        + [model.prior_logvar]
    )
    second_decoder_params = model.reconstruction_head.trainable_variables
    third_decoder_params = model.action_head.trainable_variables
    critic_params = model.critic_head.trainable_variables

    with tf.device("/GPU:0"):
        with tf.GradientTape(persistent=True) as tape:
            (
                _,
                _,
                total_loss,
                first_decoder_loss,
                second_decoder_loss,
                third_decoder_loss,
                critic_loss,
                _,
                _,
                _,
                _,
                _,
            ) = model(
                input_data,
                training=True,
                current_alpha=current_alpha,
                current_kl_weight=current_kl_weight,
                current_critic_coef=current_critic_coef,
            )

    first_decoder_gradients = tape.gradient(
        first_decoder_loss,
        first_decoder_params,
    )
    second_decoder_gradients = tape.gradient(
        second_decoder_loss,
        second_decoder_params,
    )
    third_decoder_gradients = tape.gradient(
        third_decoder_loss,
        third_decoder_params,
    )
    critic_gradients = tape.gradient(critic_loss, critic_params)

    all_gradients = (
        first_decoder_gradients
        + second_decoder_gradients
        + third_decoder_gradients
        + critic_gradients
    )
    all_params = (
        first_decoder_params
        + second_decoder_params
        + third_decoder_params
        + critic_params
    )

    optimizer.apply_gradients(zip(all_gradients, all_params))
    del tape

    return (total_loss,)


def train_model(
    model: tf.keras.Model,
    epochs: int,
    trials_per_epoch: int,
    batch_size: int,
    time_steps: int,
    dir_name: str,
    model_name: str,
) -> None:
    """Train a model with scheduled losses and checkpointing.

    Args:
        model: Variational recurrent model to train.
        epochs: Maximum number of training epochs.
        trials_per_epoch: Number of batches sampled per epoch.
        batch_size: Number of simulated trials per batch.
        time_steps: Number of sequence positions per trial.
        dir_name: Directory prefix used for checkpoints and logs.
        model_name: Base model name used for output files.

    Returns:
        None. Model weights are written to disk.
    """
    total_steps = epochs * trials_per_epoch
    lr_warmup_epochs = 0
    warmup_steps = lr_warmup_epochs * trials_per_epoch

    learning_rate_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-5,
        decay_steps=total_steps,
        alpha=0.01,
        warmup_target=0.0003,
        warmup_steps=warmup_steps,
    )
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate_schedule,
        weight_decay=1e-4,
        clipnorm=20.0,
    )

    dummy_input = tf.zeros((1, time_steps, 1), dtype=tf.float32)
    _ = model(dummy_input, training=False)

    all_trainable_variables = (
        model.encoder.trainable_variables
        + model.decoder.trainable_variables
        + [model.prior_mu]
        + [model.prior_logvar]
        + model.lstm_cell.trainable_variables
        + model.reconstruction_head.trainable_variables
        + model.action_head.trainable_variables
        + model.critic_head.trainable_variables
    )

    base_optimizer = (
        optimizer._optimizer
        if hasattr(optimizer, "_optimizer")
        else optimizer
    )
    base_optimizer.build(all_trainable_variables)

    best_loss = float("inf")
    epochs_without_improvement = 0
    patience = 120
    min_delta = 1e-10
    warmup_epochs = 80
    best_checkpoint_path = dir_name + model_name + "_BEST.weights.h5"

    kl_warmup_epochs = 10
    kl_annealing_epochs = 70
    target_kl_weight = model.lambda_
    critic_warmup_epochs = 80
    critic_annealing_epochs = 120

    if model.time_steps == 6:
        target_critic_coef = 1
    elif model.time_steps == 2:
        target_critic_coef = 0
    elif model.time_steps == 30:
        target_critic_coef = 0
    else:
        target_critic_coef = 0.1

    for epoch in range(epochs):
        epoch_total_loss = 0.0

        current_alpha = 1.0

        if epoch < kl_warmup_epochs:
            current_kl_weight = 0.0
        elif epoch >= (kl_warmup_epochs + kl_annealing_epochs):
            current_kl_weight = target_kl_weight
        else:
            progress = (epoch - kl_warmup_epochs) / kl_annealing_epochs
            current_kl_weight = target_kl_weight * progress

        if epoch < critic_warmup_epochs:
            current_critic_coef = target_critic_coef
        elif epoch >= (critic_warmup_epochs + critic_annealing_epochs):
            current_critic_coef = 0.0
        else:
            progress = (
                (epoch - critic_warmup_epochs) / critic_annealing_epochs
            )
            current_critic_coef = target_critic_coef * (1.0 - progress)

        for _ in range(trials_per_epoch):
            batch_input_data = helper.generate_batch_data(
                batch_size,
                time_steps,
            )

            (loss,) = train_step(
                model=model,
                optimizer=optimizer,
                current_alpha=tf.constant(current_alpha, dtype=tf.float32),
                current_kl_weight=tf.constant(
                    current_kl_weight,
                    dtype=tf.float32,
                ),
                current_critic_coef=tf.constant(
                    current_critic_coef,
                    dtype=tf.float32,
                ),
                input_data=batch_input_data,
            )

            epoch_total_loss += loss

        average_total_loss = epoch_total_loss / trials_per_epoch

        if epoch >= warmup_epochs:
            if average_total_loss < (best_loss - min_delta):
                best_loss = average_total_loss
                epochs_without_improvement = 0

                if os.path.exists(best_checkpoint_path):
                    os.remove(best_checkpoint_path)
                model.save_weights(best_checkpoint_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    try:
                        model.load_weights(best_checkpoint_path)
                    except Exception:
                        pass
                    break

    final_checkpoint_path = dir_name + model_name + ".weights.h5"
    if os.path.exists(final_checkpoint_path):
        os.remove(final_checkpoint_path)
    model.save_weights(final_checkpoint_path)

    if os.path.exists(best_checkpoint_path):
        os.remove(best_checkpoint_path)
