"""Single-channel VAE that observes every node reward simultaneously.

The model converts every scalar reward to a one-hot category vector, keeps
the node order by concatenating those vectors, and presents the complete
trial to one zero-state RNN step. One encoder, one latent sample, and one
decoder reconstruct a single hidden state. The shared action and
reconstruction heads and the optional critic all read that state. The
objective contains exactly one KL divergence and one fully observed critic
target per trial.
"""

from __future__ import annotations

import math
from typing import Any

import tensorflow as tf
from tensorflow.keras import layers

from ...main_vae import helper
from ...main_vae.model import (
    NoForgetLSTMCell,
    build_decoder,
    build_encoder,
)
from ..independent_channel.model import IndependentNodeChannelModel


class SimultaneousObservationChannelModel(IndependentNodeChannelModel):
    """Variational planning model with one joint simultaneous channel."""

    def __init__(
        self,
        rnn_units: int,
        latent_dim: int,
        time_steps: int,
        num_paths: int,
        index_path_map: dict[Any, list[int]],
        alpha: float = 0.0,
        lambda_: float = 1.0,
        reward_normalization_constant: float | None = None,
        action_gradient_confidence_threshold: float = 0.90,
        max_action_gradient_scale: float = 1.0,
        reconstruction_head_type: str = "nonlinear_location_scale",
        reconstruction_hidden_units: int = 128,
        critic_loss_enabled: bool = True,
        parameter_multiplier: float | None = None,
        action_entropy_bonus: float = 0.0,
    ) -> None:
        """Initialize one joint bottleneck and its global readouts."""
        # This class reuses only architecture-independent readout and actor
        # methods from IndependentNodeChannelModel. Calling Model directly
        # avoids constructing any of the parent's per-node channels.
        tf.keras.Model.__init__(
            self,
            name="simultaneous_observation_channel_model",
        )

        if rnn_units < 1 or latent_dim < 1 or time_steps < 1:
            raise ValueError(
                "rnn_units, latent_dim, and time_steps must be positive."
            )
        if num_paths < 1:
            raise ValueError("num_paths must be positive.")
        if not 0.0 <= action_gradient_confidence_threshold < 1.0:
            raise ValueError(
                "action_gradient_confidence_threshold must be in [0, 1)."
            )
        if max_action_gradient_scale < 1.0:
            raise ValueError("max_action_gradient_scale must be at least 1.")
        if reconstruction_head_type not in {
            "nonlinear_categorical",
            "nonlinear_location_scale",
        }:
            raise ValueError(
                "reconstruction_head_type must be nonlinear_categorical "
                "or nonlinear_location_scale."
            )
        if reconstruction_hidden_units < 1:
            raise ValueError(
                "reconstruction_hidden_units must be a positive integer."
            )

        self.parameter_multiplier_explicit = parameter_multiplier is not None
        parameter_scale = (
            self._parameter_scale_for_time_steps(time_steps)
            if parameter_multiplier is None
            else float(parameter_multiplier)
        )
        if not math.isfinite(parameter_scale) or parameter_scale <= 0.0:
            raise ValueError(
                "parameter_multiplier must be finite and positive."
            )
        if (
            not math.isfinite(action_entropy_bonus)
            or action_entropy_bonus < 0.0
        ):
            raise ValueError(
                "action_entropy_bonus must be finite and non-negative."
            )

        self.rnn_units = int(rnn_units)
        self.latent_dim = int(latent_dim)
        self.time_steps = int(time_steps)
        self.num_paths = int(num_paths)
        self.index_path_map = index_path_map
        self.alpha = float(alpha) * parameter_scale
        self.lambda_ = float(lambda_) * parameter_scale
        self.action_loss_weight = parameter_scale
        self.parameter_scale = parameter_scale
        self.parameter_multiplier = parameter_scale
        self.action_entropy_bonus = float(action_entropy_bonus)
        if reward_normalization_constant is not None and (
            not math.isfinite(reward_normalization_constant)
            or reward_normalization_constant <= 0.0
        ):
            raise ValueError(
                "reward_normalization_constant must be finite and positive."
            )
        self.reward_normalization_constant = reward_normalization_constant
        self.reward_normalization_scheme = (
            "shared_31_expected_max"
            if self.time_steps == 30
            else "legacy_time_step_expected_max"
        )
        self.action_gradient_confidence_threshold = float(
            action_gradient_confidence_threshold
        )
        self.max_action_gradient_scale = float(max_action_gradient_scale)
        self.reconstruction_head_type = reconstruction_head_type
        self.reconstruction_hidden_units = int(reconstruction_hidden_units)
        self.reconstruction_loss_scope = "all_nodes"
        self.reconstruction_min_reward = helper.RECONSTRUCTION_MIN_REWARD
        self.reconstruction_max_reward = helper.RECONSTRUCTION_MAX_REWARD
        self.num_categories = helper.NUM_RECONSTRUCTION_CATEGORIES
        self.critic_loss_enabled = bool(critic_loss_enabled)
        self.model_architecture = "single_simultaneous_channel"
        self.returns_reconstruction_location_scale_parameters = True
        self.kl_terms_per_trial = 1

        # The flattened input retains node identity through a fixed block for
        # every node: [node 1 categories, ..., node N categories].
        self.rnn_cell = NoForgetLSTMCell(
            self.rnn_units,
            kernel_initializer="orthogonal",
            recurrent_initializer="orthogonal",
            unit_forget_bias=False,
            name="simultaneous_rnn",
        )
        self.encoder = build_encoder(
            2 * self.rnn_units,
            self.latent_dim,
            self.rnn_units,
        )
        self.decoder = build_decoder(
            self.latent_dim,
            2 * self.rnn_units,
            self.rnn_units,
        )

        reconstruction_output_units = self.time_steps * (
            self.num_categories
            if reconstruction_head_type == "nonlinear_categorical"
            else 2
        )
        reconstruction_output_name = (
            "logits"
            if reconstruction_head_type == "nonlinear_categorical"
            else "distribution_parameters"
        )
        reconstruction_output_initializer = (
            "glorot_uniform"
            if reconstruction_head_type == "nonlinear_categorical"
            else tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        )
        self.reconstruction_head = tf.keras.Sequential(
            [
                layers.Dense(
                    self.reconstruction_hidden_units,
                    activation="relu",
                    name="hidden",
                ),
                layers.Dense(
                    reconstruction_output_units,
                    activation="linear",
                    kernel_initializer=reconstruction_output_initializer,
                    bias_initializer="zeros",
                    name=reconstruction_output_name,
                ),
            ],
            name=f"{reconstruction_head_type}_reconstruction_head",
        )
        self.action_head = layers.Dense(
            self.num_paths,
            activation=None,
            kernel_initializer="glorot_uniform",
            name="action_head",
        )
        self.critic_head = (
            layers.Dense(
                1,
                activation=None,
                kernel_initializer="glorot_uniform",
                name="critic_head",
            )
            if self.critic_loss_enabled
            else None
        )
        self.prior_mu = self.add_weight(
            name="prior_mu",
            shape=(self.latent_dim,),
            initializer="zeros",
            trainable=True,
        )
        self.prior_logvar = self.add_weight(
            name="prior_logvar",
            shape=(self.latent_dim,),
            initializer="zeros",
            trainable=True,
        )

    @property
    def shared_representation_trainable_variables(
        self,
    ) -> list[tf.Variable]:
        """Return the one RNN, encoder, decoder, and prior variables."""
        return (
            self.rnn_cell.trainable_variables
            + self.encoder.trainable_variables
            + self.decoder.trainable_variables
            + [self.prior_mu, self.prior_logvar]
        )

    @property
    def critic_trainable_variables(self) -> list[tf.Variable]:
        """Return the optional fully observed critic-head variables."""
        if self.critic_head is None:
            return []
        return self.critic_head.trainable_variables

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = True,
        current_alpha: float | tf.Tensor = 1.0,
        current_kl_weight: float | tf.Tensor = 1.0,
        current_critic_coef: float | tf.Tensor = 1.0,
    ) -> tuple[tf.Tensor, ...]:
        """Encode all rewards together and charge one KL divergence."""
        batch_size = tf.shape(inputs)[0]
        categories_onehot = helper.scalar_to_categorical(
            inputs,
            self.num_categories,
        )
        category_vectors = tf.squeeze(categories_onehot, axis=2)
        simultaneous_input = tf.reshape(
            category_vectors,
            [batch_size, self.time_steps * self.num_categories],
        )

        _, recurrent_state = self.rnn_cell.call_from_zero_state(
            simultaneous_input,
            training=training,
        )
        encoder_input = tf.concat(recurrent_state, axis=-1)
        latent_mean, latent_log_variance, latent_sample = self.encoder(
            encoder_input,
            training=training,
        )
        decoded_state = self.decoder(latent_sample, training=training)
        reconstructed_hidden_state, _ = tf.split(
            decoded_state,
            num_or_size_splits=2,
            axis=-1,
        )

        (
            reconstruction_probabilities,
            reconstruction_locations,
            reconstruction_scales,
        ) = self._reconstruction_probabilities(
            reconstructed_hidden_state
        )
        action_logits = self.action_head(reconstructed_hidden_state)
        action_probabilities = tf.nn.softmax(action_logits, axis=-1)

        target_categories = category_vectors
        raw_cross_entropy = -tf.reduce_sum(
            target_categories
            * tf.math.log(reconstruction_probabilities + 1e-7),
            axis=-1,
        )
        reconstruction_loss = tf.reduce_mean(
            raw_cross_entropy
        ) / tf.math.log(tf.cast(self.num_categories, tf.float32))

        prior_mean = tf.broadcast_to(
            self.prior_mu,
            [batch_size, self.latent_dim],
        )
        prior_variance = tf.exp(
            tf.broadcast_to(
                self.prior_logvar,
                [batch_size, self.latent_dim],
            )
        )
        information_cost = self._calculate_kl_loss(
            latent_mean,
            latent_log_variance,
            prior_mean,
            prior_variance,
        )
        information_loss = information_cost
        action_loss = self.compute_final_actor_loss(
            inputs,
            action_probabilities,
        )
        critic_loss = tf.constant(0.0, dtype=tf.float32)
        if self.critic_loss_enabled:
            value_prediction = self.critic_head(
                reconstructed_hidden_state
            )
            actual_path_rewards, _ = self._prepare_path_rewards(inputs)
            value_target = tf.stop_gradient(
                tf.reduce_max(
                    actual_path_rewards,
                    axis=1,
                    keepdims=True,
                )
            )
            critic_loss = tf.reduce_mean(
                tf.square(value_prediction - value_target)
            )
            critic_loss = tf.where(
                tf.math.is_finite(critic_loss),
                critic_loss,
                tf.constant(0.0, dtype=tf.float32),
            )

        total_loss = tf.constant(0.0, dtype=tf.float32)
        first_decoder_loss = tf.constant(0.0, dtype=tf.float32)
        second_decoder_loss = tf.constant(0.0, dtype=tf.float32)
        third_decoder_loss = tf.constant(0.0, dtype=tf.float32)
        if training:
            weighted_information_loss = (
                information_loss * current_kl_weight
            )
            weighted_action_loss = action_loss * self.action_loss_weight
            weighted_reconstruction_loss = (
                reconstruction_loss * self.alpha * current_alpha
            )
            self.add_loss(weighted_information_loss)
            self.add_loss(weighted_action_loss)
            self.add_loss(weighted_reconstruction_loss)
            total_loss = tf.add_n(self.losses)
            first_decoder_loss = (
                weighted_information_loss
                + weighted_action_loss
                + weighted_reconstruction_loss
                + critic_loss
                * self.action_loss_weight
                * current_critic_coef
            )
            second_decoder_loss = reconstruction_loss
            third_decoder_loss = action_loss

        # A single output step preserves the trainer/simulator convention that
        # the last readout reconstructs every observed reward.
        category_outputs = reconstruction_probabilities[:, tf.newaxis, :, :]
        all_latent_means = latent_mean[:, tf.newaxis, :]
        return (
            category_outputs,
            action_probabilities,
            total_loss,
            first_decoder_loss,
            second_decoder_loss,
            third_decoder_loss,
            critic_loss,
            information_loss,
            action_loss,
            reconstruction_loss,
            information_cost,
            all_latent_means,
            reconstructed_hidden_state,
            reconstruction_locations,
            reconstruction_scales,
        )
