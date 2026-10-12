"""Training utilities for the variational recurrent planning model."""

from __future__ import annotations

import csv
import math
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import tensorflow as tf

from . import helper
from project_paths import job_temporary_path, print_local_scp_command


class _LateLearningRatePlateau(
    tf.keras.optimizers.schedules.LearningRateSchedule
):
    """Cosine decay with a held late phase and a final cosine descent."""

    def __init__(
        self,
        peak_learning_rate: float,
        final_learning_rate: float,
        total_steps: int,
        plateau_start_step: int,
        plateau_end_step: int,
    ) -> None:
        super().__init__()
        self.peak_learning_rate = float(peak_learning_rate)
        self.final_learning_rate = float(final_learning_rate)
        self.total_steps = int(total_steps)
        self.plateau_start_step = int(plateau_start_step)
        self.plateau_end_step = int(plateau_end_step)
        start_progress = self.plateau_start_step / self.total_steps
        self.plateau_learning_rate = self.final_learning_rate + 0.5 * (
            self.peak_learning_rate - self.final_learning_rate
        ) * (1.0 + math.cos(math.pi * start_progress))

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        """Return the scheduled learning rate at one optimizer step."""
        dtype = tf.float32
        step = tf.cast(step, dtype)
        peak = tf.cast(self.peak_learning_rate, dtype)
        final = tf.cast(self.final_learning_rate, dtype)
        total = tf.cast(self.total_steps, dtype)
        plateau_start = tf.cast(self.plateau_start_step, dtype)
        plateau_end = tf.cast(self.plateau_end_step, dtype)
        plateau = tf.cast(self.plateau_learning_rate, dtype)

        initial_progress = tf.clip_by_value(step / total, 0.0, 1.0)
        initial_cosine = 0.5 * (
            1.0 + tf.cos(tf.constant(math.pi, dtype=dtype) * initial_progress)
        )
        initial_rate = final + (peak - final) * initial_cosine

        final_progress = tf.clip_by_value(
            (step - plateau_end) / (total - plateau_end),
            0.0,
            1.0,
        )
        final_cosine = 0.5 * (
            1.0 + tf.cos(tf.constant(math.pi, dtype=dtype) * final_progress)
        )
        final_rate = final + (plateau - final) * final_cosine

        return tf.where(
            step < plateau_start,
            initial_rate,
            tf.where(step < plateau_end, plateau, final_rate),
        )

    def get_config(self) -> dict[str, float | int]:
        """Return a serializable schedule configuration."""
        return {
            "peak_learning_rate": self.peak_learning_rate,
            "final_learning_rate": self.final_learning_rate,
            "total_steps": self.total_steps,
            "plateau_start_step": self.plateau_start_step,
            "plateau_end_step": self.plateau_end_step,
        }


def _training_reward_values(time_steps: int) -> tuple[int, ...]:
    """Return the scalar rewards sampled by the training-data generator."""
    if time_steps == 2:
        return (0, 1)
    return (-4, -3, -2, -1, 1, 2, 3, 4)


def _reward_column_suffix(reward: int) -> str:
    """Return a CSV-safe label for a signed integer reward."""
    if reward < 0:
        return f"minus{abs(reward)}"
    if reward > 0:
        return f"plus{reward}"
    return "zero"


def _gradient_global_norm(gradients: list[tf.Tensor | None]) -> tf.Tensor:
    """Return the global norm of available gradients."""
    available_gradients = [
        gradient for gradient in gradients if gradient is not None
    ]
    if not available_gradients:
        return tf.constant(0.0, dtype=tf.float32)
    return tf.linalg.global_norm(available_gradients)


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
        Tuple of losses, policy diagnostics, pre-clipping gradient norms, and
        final-state reconstruction diagnostics plus per-reward probability,
        modal-correct, endpoint-probability, and observation sums for
        epoch-level aggregation.
    """
    first_decoder_params = (
        model.shared_representation_trainable_variables
    )
    second_decoder_params = model.reconstruction_head.trainable_variables
    third_decoder_params = model.action_head.trainable_variables
    critic_params = model.critic_trainable_variables

    with tf.device("/CPU:0"):
        with tf.GradientTape(persistent=True) as tape:
            model_outputs = model(
                input_data,
                training=True,
                current_alpha=current_alpha,
                current_kl_weight=current_kl_weight,
                current_critic_coef=current_critic_coef,
            )
            (
                category_outputs,
                action_probs,
                total_loss,
                first_decoder_loss,
                second_decoder_loss,
                third_decoder_loss,
                critic_loss,
                information_loss,
                action_loss,
                reconstruction_loss,
                _,
                _,
                final_hidden_state,
            ) = model_outputs[:13]

            if (
                model.reconstruction_head_type
                == "nonlinear_location_scale"
            ):
                if getattr(
                    model,
                    "returns_reconstruction_location_scale_parameters",
                    False,
                ):
                    reconstruction_locations = model_outputs[13]
                    reconstruction_scales = model_outputs[14]
                else:
                    reconstruction_locations, reconstruction_scales = (
                        model.reconstruction_location_scale_parameters(
                            final_hidden_state
                        )
                    )
                reconstruction_location_mean = tf.reduce_mean(
                    reconstruction_locations
                )
                reconstruction_location_squared_mean = tf.reduce_mean(
                    tf.square(reconstruction_locations)
                )
                reconstruction_location_absolute_mean = tf.reduce_mean(
                    tf.abs(reconstruction_locations)
                )
                reconstruction_location_min = tf.reduce_min(
                    reconstruction_locations
                )
                reconstruction_location_max = tf.reduce_max(
                    reconstruction_locations
                )
                location_saturation_threshold = 0.95 * max(
                    abs(float(model.reconstruction_min_reward)),
                    abs(float(model.reconstruction_max_reward)),
                )
                reconstruction_location_saturation_fraction = (
                    tf.reduce_mean(
                        tf.cast(
                            tf.abs(reconstruction_locations)
                            >= location_saturation_threshold,
                            tf.float32,
                        )
                    )
                )
                reconstruction_scale_mean = tf.reduce_mean(
                    reconstruction_scales
                )
                reconstruction_scale_squared_mean = tf.reduce_mean(
                    tf.square(reconstruction_scales)
                )
                reconstruction_scale_min = tf.reduce_min(
                    reconstruction_scales
                )
                reconstruction_scale_max = tf.reduce_max(
                    reconstruction_scales
                )
            else:
                unavailable = tf.constant(float("nan"), dtype=tf.float32)
                reconstruction_location_mean = unavailable
                reconstruction_location_squared_mean = unavailable
                reconstruction_location_absolute_mean = unavailable
                reconstruction_location_min = unavailable
                reconstruction_location_max = unavailable
                reconstruction_location_saturation_fraction = unavailable
                reconstruction_scale_mean = unavailable
                reconstruction_scale_squared_mean = unavailable
                reconstruction_scale_min = unavailable
                reconstruction_scale_max = unavailable

    # These diagnostics use the final recurrent state's reconstruction of all
    # observed rewards, regardless of which temporal outputs receive loss.
    # True-category probability is expected exact accuracy for one draw and
    # avoids adding response-sampling noise to the training log.
    final_category_probabilities = category_outputs[:, -1, :, :]
    probability_plus5 = final_category_probabilities[:, :, 0]
    probability_minus5 = final_category_probabilities[:, :, -1]
    reconstruction_probability_plus5_mean = tf.reduce_mean(
        probability_plus5
    )
    reconstruction_probability_minus5_mean = tf.reduce_mean(
        probability_minus5
    )
    target_onehot = tf.squeeze(
        helper.scalar_to_categorical(input_data, model.num_categories),
        axis=2,
    )
    true_category_probabilities = tf.reduce_sum(
        final_category_probabilities * target_onehot,
        axis=-1,
    )
    target_category_indices = tf.argmax(
        target_onehot,
        axis=-1,
        output_type=tf.int32,
    )
    modal_category_indices = tf.argmax(
        final_category_probabilities,
        axis=-1,
        output_type=tf.int32,
    )
    flat_target_indices = tf.reshape(target_category_indices, [-1])
    reward_observation_counts = tf.math.unsorted_segment_sum(
        tf.ones_like(flat_target_indices, dtype=tf.float32),
        flat_target_indices,
        model.num_categories,
    )
    reward_true_probability_sums = tf.math.unsorted_segment_sum(
        tf.reshape(true_category_probabilities, [-1]),
        flat_target_indices,
        model.num_categories,
    )
    reward_modal_correct_sums = tf.math.unsorted_segment_sum(
        tf.cast(
            tf.reshape(
                tf.equal(modal_category_indices, target_category_indices),
                [-1],
            ),
            tf.float32,
        ),
        flat_target_indices,
        model.num_categories,
    )
    reward_probability_plus5_sums = tf.math.unsorted_segment_sum(
        tf.reshape(probability_plus5, [-1]),
        flat_target_indices,
        model.num_categories,
    )
    reward_probability_minus5_sums = tf.math.unsorted_segment_sum(
        tf.reshape(probability_minus5, [-1]),
        flat_target_indices,
        model.num_categories,
    )
    if model.reconstruction_head_type == "nonlinear_location_scale":
        reward_location_sums = tf.math.unsorted_segment_sum(
            tf.reshape(reconstruction_locations, [-1]),
            flat_target_indices,
            model.num_categories,
        )
        reward_scale_sums = tf.math.unsorted_segment_sum(
            tf.reshape(reconstruction_scales, [-1]),
            flat_target_indices,
            model.num_categories,
        )
    else:
        reward_location_sums = tf.fill(
            [model.num_categories],
            tf.constant(float("nan"), dtype=tf.float32),
        )
        reward_scale_sums = tf.fill(
            [model.num_categories],
            tf.constant(float("nan"), dtype=tf.float32),
        )

    first_decoder_gradients = tape.gradient(
        first_decoder_loss,
        first_decoder_params,
    )
    second_decoder_gradients = tape.gradient(
        second_decoder_loss,
        second_decoder_params,
    )
    action_loss_enabled = bool(
        getattr(model, "action_loss_enabled", True)
    )
    if action_loss_enabled:
        third_decoder_gradients = tape.gradient(
            third_decoder_loss,
            third_decoder_params,
        )
    else:
        third_decoder_gradients = [None for _ in third_decoder_params]
    if model.critic_loss_enabled:
        critic_gradients = tape.gradient(critic_loss, critic_params)
    else:
        critic_gradients = [None for _ in critic_params]

    mean_action_confidence = tf.reduce_mean(
        tf.reduce_max(action_probs, axis=-1)
    )
    mean_action_gradient_scale = (
        tf.reduce_mean(model.action_gradient_scale(action_probs))
        if action_loss_enabled
        else tf.constant(0.0, dtype=tf.float32)
    )
    shared_gradient_norm = _gradient_global_norm(first_decoder_gradients)
    reconstruction_head_gradient_norm = _gradient_global_norm(
        second_decoder_gradients
    )
    action_head_gradient_norm = _gradient_global_norm(
        third_decoder_gradients
    )
    critic_head_gradient_norm = _gradient_global_norm(critic_gradients)

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

    gradient_parameter_pairs = [
        (gradient, parameter)
        for gradient, parameter in zip(all_gradients, all_params)
        if gradient is not None
    ]
    optimizer.apply_gradients(gradient_parameter_pairs)
    del tape

    return (
        total_loss,
        information_loss,
        action_loss,
        reconstruction_loss,
        critic_loss,
        mean_action_confidence,
        mean_action_gradient_scale,
        shared_gradient_norm,
        reconstruction_head_gradient_norm,
        action_head_gradient_norm,
        critic_head_gradient_norm,
        _gradient_global_norm(all_gradients),
        reconstruction_location_mean,
        reconstruction_location_squared_mean,
        reconstruction_location_absolute_mean,
        reconstruction_location_saturation_fraction,
        reconstruction_scale_mean,
        reconstruction_scale_squared_mean,
        reconstruction_probability_plus5_mean,
        reconstruction_probability_minus5_mean,
        reconstruction_location_min,
        reconstruction_location_max,
        reconstruction_scale_min,
        reconstruction_scale_max,
        reward_true_probability_sums,
        reward_modal_correct_sums,
        reward_observation_counts,
        reward_probability_plus5_sums,
        reward_probability_minus5_sums,
        reward_location_sums,
        reward_scale_sums,
    )


def train_model(
    model: tf.keras.Model,
    epochs: int,
    trials_per_epoch: int,
    batch_size: int,
    time_steps: int,
    weight_dir: str | Path,
    checkpoint_dir: str | Path,
    training_log_dir: str | Path,
    model_name: str,
    final_learning_rate: float = 3e-4,
    late_lr_plateau_start_epoch: int | None = None,
    late_lr_plateau_end_epoch: int | None = None,
    initial_critic_coefficient: float | None = None,
    batch_data_generator: Callable[[int, int], tf.Tensor] | None = None,
    progress_timing_label: str | None = None,
) -> None:
    """Train a model with scheduled losses and save the final epoch.

    Args:
        model: Variational recurrent model to train.
        epochs: Maximum number of training epochs.
        trials_per_epoch: Number of batches sampled per epoch.
        batch_size: Number of simulated trials per batch.
        time_steps: Number of sequence positions per trial.
        weight_dir: Directory for final model weights.
        checkpoint_dir: Directory that may contain stale checkpoints from
            older versions. Best-checkpoint selection is currently disabled.
        training_log_dir: Directory for epoch-level training diagnostics.
        model_name: Base model name used for output files.
        final_learning_rate: Terminal value of the cosine learning-rate
            schedule. The peak learning rate remains 3e-4.
        late_lr_plateau_start_epoch: Optional epoch boundary after which the
            ordinary cosine learning rate is held constant.
        late_lr_plateau_end_epoch: Optional epoch boundary at which a final
            cosine decay from the held rate to ``final_learning_rate`` begins.
        initial_critic_coefficient: Optional override for the critic
            coefficient before its epoch-80 annealing begins. ``None`` uses
            the model's tree-dependent default.
        batch_data_generator: Optional model-family-specific vectorized reward
            generator. The established Python generator is used when omitted.
        progress_timing_label: Optional label enabling timestamped, flushed
            runtime diagnostics. Timing is summarized every ten epochs.

    Returns:
        None. Model weights are written to disk.
    """
    weight_dir = Path(weight_dir)
    checkpoint_dir = Path(checkpoint_dir)
    training_log_dir = Path(training_log_dir)
    weight_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    training_log_dir.mkdir(parents=True, exist_ok=True)
    generate_batch_data = (
        helper.generate_batch_data
        if batch_data_generator is None
        else batch_data_generator
    )
    timing_enabled = bool(progress_timing_label)

    def progress_log(message: str) -> None:
        if not timing_enabled:
            return
        timestamp = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )
        print(
            f"[{timestamp}] {progress_timing_label}: {message}",
            flush=True,
        )

    training_timer_start = time.perf_counter()
    progress_log(
        "trainer setup started; "
        f"epochs={epochs}, batches_per_epoch={trials_per_epoch}, "
        f"batch_size={batch_size}, total_optimizer_steps="
        f"{epochs * trials_per_epoch}"
    )

    total_steps = epochs * trials_per_epoch
    peak_learning_rate = 3e-4
    if not 0.0 < final_learning_rate <= peak_learning_rate:
        raise ValueError(
            "final_learning_rate must be positive and no greater than 3e-4."
        )
    plateau_epochs = (
        late_lr_plateau_start_epoch,
        late_lr_plateau_end_epoch,
    )
    if (late_lr_plateau_start_epoch is None) != (
        late_lr_plateau_end_epoch is None
    ):
        raise ValueError(
            "Both late learning-rate plateau epochs must be provided together."
        )
    if late_lr_plateau_start_epoch is None:
        learning_rate_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-5,
            decay_steps=total_steps,
            alpha=final_learning_rate / peak_learning_rate,
            warmup_target=peak_learning_rate,
            warmup_steps=0,
        )
        learning_rate_schedule_name = "cosine"
        plateau_learning_rate: float | None = None
    else:
        if not (
            0 < late_lr_plateau_start_epoch
            < late_lr_plateau_end_epoch
            < epochs
        ):
            raise ValueError(
                "Late learning-rate plateau epochs must satisfy "
                "0 < start < end < epochs."
            )
        learning_rate_schedule = _LateLearningRatePlateau(
            peak_learning_rate=peak_learning_rate,
            final_learning_rate=final_learning_rate,
            total_steps=total_steps,
            plateau_start_step=(
                late_lr_plateau_start_epoch * trials_per_epoch
            ),
            plateau_end_step=late_lr_plateau_end_epoch * trials_per_epoch,
        )
        learning_rate_schedule_name = "late_plateau"
        plateau_learning_rate = (
            learning_rate_schedule.plateau_learning_rate
        )
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate_schedule,
        weight_decay=1e-4,
        clipnorm=20.0,
    )

    dummy_input = tf.zeros((1, time_steps, 1), dtype=tf.float32)
    _ = model(dummy_input, training=False)

    all_trainable_variables = (
        model.shared_representation_trainable_variables
        + model.reconstruction_head.trainable_variables
        + model.action_head.trainable_variables
        + model.critic_trainable_variables
    )

    base_optimizer = (
        optimizer._optimizer
        if hasattr(optimizer, "_optimizer")
        else optimizer
    )
    base_optimizer.build(all_trainable_variables)
    progress_log(
        "model variables and optimizer initialized in "
        f"{time.perf_counter() - training_timer_start:.2f}s; "
        f"trainable_variables={len(all_trainable_variables)}"
    )

    training_log_path = training_log_dir / (model_name + ".csv")
    final_weight_path = weight_dir / (model_name + ".weights.h5")
    temporary_training_log_path = job_temporary_path(
        training_log_path,
        label="training_log",
        suffix=".csv.in_progress",
    )
    temporary_final_weight_path = job_temporary_path(
        final_weight_path,
        label="final_weight",
        suffix=".weights.h5",
    )
    temporary_training_log_path.unlink(missing_ok=True)
    temporary_final_weight_path.unlink(missing_ok=True)
    kl_warmup_epochs = 10
    kl_annealing_epochs = 70
    target_kl_weight = model.lambda_
    critic_warmup_epochs = 80
    critic_annealing_epochs = 120

    if initial_critic_coefficient is not None and (
        not math.isfinite(initial_critic_coefficient)
        or initial_critic_coefficient < 0.0
    ):
        raise ValueError(
            "initial_critic_coefficient must be finite and non-negative."
        )
    action_loss_enabled = bool(
        getattr(model, "action_loss_enabled", True)
    )
    if not action_loss_enabled:
        configured_initial_critic_coefficient = 0.0
    elif initial_critic_coefficient is not None:
        configured_initial_critic_coefficient = float(
            initial_critic_coefficient
        )
    elif model.time_steps in {6, 12}:
        configured_initial_critic_coefficient = 1.0
    elif model.time_steps in {2, 30}:
        configured_initial_critic_coefficient = 0.0
    else:
        configured_initial_critic_coefficient = 0.1
    target_critic_coef = (
        configured_initial_critic_coefficient
        if model.critic_loss_enabled
        else 0.0
    )

    metric_names = (
        "total_loss",
        "information_loss",
        "action_loss",
        "reconstruction_loss",
        "critic_loss",
        "action_confidence",
        "action_gradient_scale",
        "shared_gradient_norm",
        "reconstruction_head_gradient_norm",
        "action_head_gradient_norm",
        "critic_head_gradient_norm",
        "all_gradient_norm",
        "reconstruction_location_mean",
        "reconstruction_location_squared_mean",
        "reconstruction_location_absolute_mean",
        "reconstruction_location_saturation_fraction",
        "reconstruction_scale_mean",
        "reconstruction_scale_squared_mean",
        "reconstruction_probability_plus5_mean",
        "reconstruction_probability_minus5_mean",
    )
    extrema_metric_names = (
        "reconstruction_location_min",
        "reconstruction_location_max",
        "reconstruction_scale_min",
        "reconstruction_scale_max",
    )
    derived_diagnostic_fieldnames = (
        "reconstruction_location_std",
        "reconstruction_scale_std",
        "reconstruction_probability_plus5_given_actual_plus4",
        "reconstruction_probability_minus5_given_actual_minus4",
        "reconstruction_endpoint_leakage_mean",
        "reconstruction_location_mean_given_actual_plus4",
        "reconstruction_location_mean_given_actual_minus4",
        "reconstruction_scale_mean_given_actual_plus4",
        "reconstruction_scale_mean_given_actual_minus4",
    )
    training_reward_values = _training_reward_values(time_steps)
    reward_metric_fieldnames = tuple(
        fieldname
        for reward in training_reward_values
        for fieldname in (
            "reconstruction_expected_accuracy_reward_"
            + _reward_column_suffix(reward),
            "reconstruction_modal_accuracy_reward_"
            + _reward_column_suffix(reward),
            "reconstruction_observations_reward_"
            + _reward_column_suffix(reward),
        )
    )
    log_fieldnames = (
        "epoch",
        *metric_names,
        *extrema_metric_names,
        *derived_diagnostic_fieldnames,
        *reward_metric_fieldnames,
        "learning_rate",
        "final_learning_rate",
        "learning_rate_schedule",
        "late_lr_plateau_start_epoch",
        "late_lr_plateau_end_epoch",
        "late_lr_plateau_learning_rate",
        "kl_weight",
        "reconstruction_weight",
        "critic_coefficient",
        "requested_epochs",
        "model_architecture",
        "model_condition",
        "action_loss_enabled",
        "action_loss_weight",
        "reconstruction_head_type",
        "reconstruction_hidden_units",
        "reconstruction_loss_scope",
        "channel_kl_aggregation",
        "channel_kl_time_weighting",
        "rnn_units",
        "latent_dim",
        "reconstruction_min_reward",
        "reconstruction_max_reward",
        "reconstruction_category_count",
        "critic_loss_enabled",
        "max_action_gradient_scale",
        "parameter_multiplier",
        "parameter_multiplier_explicit",
        "initial_critic_coefficient",
        "initial_critic_coefficient_explicit",
        "effective_initial_critic_coefficient",
        "action_entropy_bonus",
        "reward_normalization_constant",
        "reward_normalization_scheme",
        "checkpoint_policy",
        "checkpoint_eligible",
        "saved_best_checkpoint",
        "selected_for_final_weights",
    )

    with temporary_training_log_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as log_file:
        log_writer = csv.DictWriter(log_file, fieldnames=log_fieldnames)
        log_writer.writeheader()
        log_file.flush()
        progress_log(
            "training loop started; temporary_log="
            f"{temporary_training_log_path}"
        )

        window_start = time.perf_counter()
        window_epoch_count = 0
        window_generation_seconds = 0.0
        window_train_step_seconds = 0.0
        window_aggregation_seconds = 0.0

        for epoch in range(epochs):
            epoch_start = time.perf_counter()
            epoch_generation_seconds = 0.0
            epoch_train_step_seconds = 0.0
            epoch_aggregation_seconds = 0.0
            epoch_metric_totals = {
                metric_name: tf.constant(0.0, dtype=tf.float32)
                for metric_name in metric_names
            }
            location_scale_diagnostics_available = (
                model.reconstruction_head_type
                == "nonlinear_location_scale"
            )
            if location_scale_diagnostics_available:
                epoch_extrema = {
                    "reconstruction_location_min": tf.constant(
                        float("inf"), dtype=tf.float32
                    ),
                    "reconstruction_location_max": tf.constant(
                        float("-inf"), dtype=tf.float32
                    ),
                    "reconstruction_scale_min": tf.constant(
                        float("inf"), dtype=tf.float32
                    ),
                    "reconstruction_scale_max": tf.constant(
                        float("-inf"), dtype=tf.float32
                    ),
                }
            else:
                epoch_extrema = {
                    metric_name: float("nan")
                    for metric_name in extrema_metric_names
                }
            epoch_reward_true_probability_sums = tf.zeros(
                [model.num_categories],
                dtype=tf.float32,
            )
            epoch_reward_modal_correct_sums = tf.zeros(
                [model.num_categories],
                dtype=tf.float32,
            )
            epoch_reward_observation_counts = tf.zeros(
                [model.num_categories],
                dtype=tf.float32,
            )
            epoch_reward_probability_plus5_sums = tf.zeros(
                [model.num_categories],
                dtype=tf.float32,
            )
            epoch_reward_probability_minus5_sums = tf.zeros(
                [model.num_categories],
                dtype=tf.float32,
            )
            epoch_reward_location_sums = tf.zeros(
                [model.num_categories],
                dtype=tf.float32,
            )
            epoch_reward_scale_sums = tf.zeros(
                [model.num_categories],
                dtype=tf.float32,
            )

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

            for batch_index in range(trials_per_epoch):
                phase_start = time.perf_counter()
                batch_input_data = generate_batch_data(
                    batch_size,
                    time_steps,
                )
                epoch_generation_seconds += time.perf_counter() - phase_start

                phase_start = time.perf_counter()
                step_outputs = train_step(
                    model=model,
                    optimizer=optimizer,
                    current_alpha=tf.constant(
                        current_alpha,
                        dtype=tf.float32,
                    ),
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
                train_step_duration = time.perf_counter() - phase_start
                epoch_train_step_seconds += train_step_duration
                if epoch == 0 and batch_index == 0:
                    progress_log(
                        "first TensorFlow training step completed in "
                        f"{train_step_duration:.2f}s; this includes initial "
                        "tf.function graph tracing"
                    )

                phase_start = time.perf_counter()
                step_metrics = step_outputs[:len(metric_names)]
                extrema_start = len(metric_names)
                extrema_end = extrema_start + len(extrema_metric_names)
                step_extrema = step_outputs[extrema_start:extrema_end]
                (
                    reward_true_probability_sums,
                    reward_modal_correct_sums,
                    reward_observation_counts,
                    reward_probability_plus5_sums,
                    reward_probability_minus5_sums,
                    reward_location_sums,
                    reward_scale_sums,
                ) = step_outputs[extrema_end:]

                for metric_name, metric_value in zip(
                    metric_names,
                    step_metrics,
                ):
                    epoch_metric_totals[metric_name] += metric_value
                if location_scale_diagnostics_available:
                    step_extrema_values = {
                        metric_name: metric_value
                        for metric_name, metric_value in zip(
                            extrema_metric_names,
                            step_extrema,
                        )
                    }
                    for metric_name in (
                        "reconstruction_location_min",
                        "reconstruction_scale_min",
                    ):
                        epoch_extrema[metric_name] = tf.minimum(
                            epoch_extrema[metric_name],
                            step_extrema_values[metric_name],
                        )
                    for metric_name in (
                        "reconstruction_location_max",
                        "reconstruction_scale_max",
                    ):
                        epoch_extrema[metric_name] = tf.maximum(
                            epoch_extrema[metric_name],
                            step_extrema_values[metric_name],
                        )
                epoch_reward_true_probability_sums += (
                    reward_true_probability_sums
                )
                epoch_reward_modal_correct_sums += reward_modal_correct_sums
                epoch_reward_observation_counts += reward_observation_counts
                epoch_reward_probability_plus5_sums += (
                    reward_probability_plus5_sums
                )
                epoch_reward_probability_minus5_sums += (
                    reward_probability_minus5_sums
                )
                epoch_reward_location_sums += reward_location_sums
                epoch_reward_scale_sums += reward_scale_sums
                epoch_aggregation_seconds += (
                    time.perf_counter() - phase_start
                )

            epoch_metrics = {
                metric_name: float(metric_total.numpy()) / trials_per_epoch
                for metric_name, metric_total in epoch_metric_totals.items()
            }
            if location_scale_diagnostics_available:
                location_variance = max(
                    0.0,
                    epoch_metrics[
                        "reconstruction_location_squared_mean"
                    ]
                    - epoch_metrics["reconstruction_location_mean"] ** 2,
                )
                scale_variance = max(
                    0.0,
                    epoch_metrics["reconstruction_scale_squared_mean"]
                    - epoch_metrics["reconstruction_scale_mean"] ** 2,
                )
                reconstruction_location_std = math.sqrt(location_variance)
                reconstruction_scale_std = math.sqrt(scale_variance)
            else:
                reconstruction_location_std = float("nan")
                reconstruction_scale_std = float("nan")
            if location_scale_diagnostics_available:
                epoch_extrema_for_log = {
                    metric_name: float(metric_value.numpy())
                    for metric_name, metric_value in epoch_extrema.items()
                }
            else:
                epoch_extrema_for_log = epoch_extrema
            epoch_reward_metrics: dict[str, float | int] = {}
            for reward in training_reward_values:
                category_index = model.reconstruction_max_reward - reward
                suffix = _reward_column_suffix(reward)
                observation_count = int(
                    round(
                        float(
                            epoch_reward_observation_counts[
                                category_index
                            ].numpy()
                        )
                    )
                )
                epoch_reward_metrics[
                    f"reconstruction_observations_reward_{suffix}"
                ] = observation_count
                if observation_count == 0:
                    expected_accuracy = float("nan")
                    modal_accuracy = float("nan")
                else:
                    expected_accuracy = float(
                        epoch_reward_true_probability_sums[
                            category_index
                        ].numpy()
                    ) / observation_count
                    modal_accuracy = float(
                        epoch_reward_modal_correct_sums[
                            category_index
                        ].numpy()
                    ) / observation_count
                epoch_reward_metrics[
                    f"reconstruction_expected_accuracy_reward_{suffix}"
                ] = expected_accuracy
                epoch_reward_metrics[
                    f"reconstruction_modal_accuracy_reward_{suffix}"
                ] = modal_accuracy

            plus4_category_index = model.reconstruction_max_reward - 4
            minus4_category_index = model.reconstruction_max_reward - (-4)
            plus4_observation_count = float(
                epoch_reward_observation_counts[
                    plus4_category_index
                ].numpy()
            )
            minus4_observation_count = float(
                epoch_reward_observation_counts[
                    minus4_category_index
                ].numpy()
            )
            if plus4_observation_count == 0:
                probability_plus5_given_actual_plus4 = float("nan")
            else:
                probability_plus5_given_actual_plus4 = float(
                    epoch_reward_probability_plus5_sums[
                        plus4_category_index
                    ].numpy()
                ) / plus4_observation_count
            if minus4_observation_count == 0:
                probability_minus5_given_actual_minus4 = float("nan")
            else:
                probability_minus5_given_actual_minus4 = float(
                    epoch_reward_probability_minus5_sums[
                        minus4_category_index
                    ].numpy()
                ) / minus4_observation_count
            if (
                plus4_observation_count == 0
                or minus4_observation_count == 0
            ):
                endpoint_leakage_mean = float("nan")
            else:
                endpoint_leakage_mean = 0.5 * (
                    probability_plus5_given_actual_plus4
                    + probability_minus5_given_actual_minus4
                )
            if (
                location_scale_diagnostics_available
                and plus4_observation_count > 0
                and minus4_observation_count > 0
            ):
                location_mean_given_actual_plus4 = float(
                    epoch_reward_location_sums[
                        plus4_category_index
                    ].numpy()
                ) / plus4_observation_count
                location_mean_given_actual_minus4 = float(
                    epoch_reward_location_sums[
                        minus4_category_index
                    ].numpy()
                ) / minus4_observation_count
                scale_mean_given_actual_plus4 = float(
                    epoch_reward_scale_sums[
                        plus4_category_index
                    ].numpy()
                ) / plus4_observation_count
                scale_mean_given_actual_minus4 = float(
                    epoch_reward_scale_sums[
                        minus4_category_index
                    ].numpy()
                ) / minus4_observation_count
            else:
                location_mean_given_actual_plus4 = float("nan")
                location_mean_given_actual_minus4 = float("nan")
                scale_mean_given_actual_plus4 = float("nan")
                scale_mean_given_actual_minus4 = float("nan")
            selected_for_final_weights = epoch == epochs - 1
            if selected_for_final_weights:
                model.save_weights(str(temporary_final_weight_path))
                temporary_final_weight_path.replace(final_weight_path)

            learning_rate = float(
                learning_rate_schedule(optimizer.iterations).numpy()
            )
            log_writer.writerow(
                {
                    "epoch": epoch + 1,
                    **epoch_metrics,
                    **epoch_extrema_for_log,
                    "reconstruction_location_std": (
                        reconstruction_location_std
                    ),
                    "reconstruction_scale_std": reconstruction_scale_std,
                    "reconstruction_probability_plus5_given_actual_plus4": (
                        probability_plus5_given_actual_plus4
                    ),
                    "reconstruction_probability_minus5_given_actual_minus4": (
                        probability_minus5_given_actual_minus4
                    ),
                    "reconstruction_endpoint_leakage_mean": (
                        endpoint_leakage_mean
                    ),
                    "reconstruction_location_mean_given_actual_plus4": (
                        location_mean_given_actual_plus4
                    ),
                    "reconstruction_location_mean_given_actual_minus4": (
                        location_mean_given_actual_minus4
                    ),
                    "reconstruction_scale_mean_given_actual_plus4": (
                        scale_mean_given_actual_plus4
                    ),
                    "reconstruction_scale_mean_given_actual_minus4": (
                        scale_mean_given_actual_minus4
                    ),
                    **epoch_reward_metrics,
                    "learning_rate": learning_rate,
                    "final_learning_rate": final_learning_rate,
                    "learning_rate_schedule": (
                        learning_rate_schedule_name
                    ),
                    "late_lr_plateau_start_epoch": (
                        ""
                        if plateau_epochs[0] is None
                        else plateau_epochs[0]
                    ),
                    "late_lr_plateau_end_epoch": (
                        ""
                        if plateau_epochs[1] is None
                        else plateau_epochs[1]
                    ),
                    "late_lr_plateau_learning_rate": (
                        ""
                        if plateau_learning_rate is None
                        else plateau_learning_rate
                    ),
                    "kl_weight": float(current_kl_weight),
                    "reconstruction_weight": float(
                        model.alpha * current_alpha
                    ),
                    "critic_coefficient": float(current_critic_coef),
                    "requested_epochs": epochs,
                    "model_architecture": model.model_architecture,
                    "model_condition": getattr(
                        model,
                        "model_condition",
                        "planning",
                    ),
                    "action_loss_enabled": action_loss_enabled,
                    "action_loss_weight": getattr(
                        model,
                        "action_loss_weight",
                        0.0,
                    ),
                    "reconstruction_head_type": (
                        model.reconstruction_head_type
                    ),
                    "reconstruction_hidden_units": (
                        model.reconstruction_hidden_units
                    ),
                    "reconstruction_loss_scope": (
                        model.reconstruction_loss_scope
                    ),
                    "channel_kl_aggregation": getattr(
                        model,
                        "kl_aggregation",
                        "not_applicable",
                    ),
                    "channel_kl_time_weighting": getattr(
                        model,
                        "kl_time_weighting",
                        "not_applicable",
                    ),
                    "rnn_units": model.rnn_units,
                    "latent_dim": model.latent_dim,
                    "reconstruction_min_reward": (
                        model.reconstruction_min_reward
                    ),
                    "reconstruction_max_reward": (
                        model.reconstruction_max_reward
                    ),
                    "reconstruction_category_count": model.num_categories,
                    "critic_loss_enabled": model.critic_loss_enabled,
                    "max_action_gradient_scale": (
                        model.max_action_gradient_scale
                    ),
                    "parameter_multiplier": getattr(
                        model,
                        "parameter_multiplier",
                        getattr(model, "parameter_scale", 1.0),
                    ),
                    "parameter_multiplier_explicit": (
                        getattr(
                            model,
                            "parameter_multiplier_explicit",
                            False,
                        )
                    ),
                    "initial_critic_coefficient": (
                        configured_initial_critic_coefficient
                    ),
                    "initial_critic_coefficient_explicit": (
                        initial_critic_coefficient is not None
                    ),
                    "effective_initial_critic_coefficient": (
                        target_critic_coef
                    ),
                    "action_entropy_bonus": model.action_entropy_bonus,
                    "reward_normalization_constant": (
                        model.reward_normalization_constant
                    ),
                    "reward_normalization_scheme": getattr(
                        model,
                        "reward_normalization_scheme",
                        "legacy_time_step_expected_max",
                    ),
                    "checkpoint_policy": "final_epoch",
                    "checkpoint_eligible": False,
                    "saved_best_checkpoint": False,
                    "selected_for_final_weights": (
                        selected_for_final_weights
                    ),
                }
            )
            log_file.flush()
            completed_epoch = epoch + 1
            epoch_total_seconds = time.perf_counter() - epoch_start
            epoch_finalize_seconds = max(
                0.0,
                epoch_total_seconds
                - epoch_generation_seconds
                - epoch_train_step_seconds
                - epoch_aggregation_seconds,
            )
            window_epoch_count += 1
            window_generation_seconds += epoch_generation_seconds
            window_train_step_seconds += epoch_train_step_seconds
            window_aggregation_seconds += epoch_aggregation_seconds
            if completed_epoch % 10 == 0 or completed_epoch == epochs:
                if timing_enabled:
                    window_total_seconds = time.perf_counter() - window_start
                    cumulative_seconds = (
                        time.perf_counter() - training_timer_start
                    )
                    average_epoch_seconds = (
                        cumulative_seconds / completed_epoch
                    )
                    eta_seconds = average_epoch_seconds * (
                        epochs - completed_epoch
                    )
                    window_finalize_seconds = max(
                        0.0,
                        window_total_seconds
                        - window_generation_seconds
                        - window_train_step_seconds
                        - window_aggregation_seconds,
                    )
                    progress_log(
                        f"completed epoch {completed_epoch}/{epochs}; "
                        f"last_{window_epoch_count}_epochs="
                        f"{window_total_seconds:.2f}s "
                        f"({window_total_seconds / window_epoch_count:.2f}"
                        "s/epoch); breakdown: "
                        f"batch_generation={window_generation_seconds:.2f}s, "
                        f"tensorflow_train_step={window_train_step_seconds:.2f}s, "
                        f"metric_aggregation={window_aggregation_seconds:.2f}s, "
                        f"epoch_finalize={window_finalize_seconds:.2f}s; "
                        f"cumulative={cumulative_seconds:.2f}s; "
                        f"estimated_remaining={eta_seconds:.2f}s; "
                        f"latest_epoch_total={epoch_total_seconds:.2f}s, "
                        f"latest_epoch_finalize={epoch_finalize_seconds:.2f}s"
                    )
                    window_start = time.perf_counter()
                    window_epoch_count = 0
                    window_generation_seconds = 0.0
                    window_train_step_seconds = 0.0
                    window_aggregation_seconds = 0.0
                else:
                    print(
                        f"Completed epoch {completed_epoch}/{epochs}.",
                        flush=True,
                    )

    temporary_training_log_path.replace(training_log_path)
    progress_log(
        "training loop and final weight selection completed in "
        f"{time.perf_counter() - training_timer_start:.2f}s"
    )
    print_local_scp_command(
        training_log_path,
        ".local_artifacts/logs/training",
    )
