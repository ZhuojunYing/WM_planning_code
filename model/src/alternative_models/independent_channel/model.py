"""Independent per-node bottleneck model for architecture comparisons.

Unlike the sequential model in ``model.py``, this comparison model gives every
reward node its own one-step RNN -> encoder -> decoder channel. A channel sees
only its node's reward and never receives another channel's recurrent or
latent state. The reconstructed hidden states are concatenated only after all
independent bottlenecks, and the shared action and reconstruction heads read
that concatenation.

Consequently, both task losses backpropagate through every channel, while each
channel also contributes its own KL divergence. By default, each channel KL is
multiplied by the inclusive number of timesteps from its presentation through
the terminal readout, then the weighted channel costs are summed. This adds a
sequential maintenance cost without recurrent communication between channels.
Uniform temporal weights and mean aggregation remain available for reproducing
earlier runs. No critic layer or critic loss is constructed.
"""

from __future__ import annotations

import math
from typing import Any

import tensorflow as tf
from tensorflow.keras import layers, models

from ...main_vae import helper
from ...main_vae.model import NoForgetLSTMCell, sampling


def _build_channel_encoder(
    channel_index: int,
    input_dim: int,
    latent_dim: int,
    rnn_units: int,
) -> tf.keras.Model:
    """Build one node-specific posterior encoder."""
    encoder_inputs = layers.Input(
        shape=(input_dim,),
        name=f"node_{channel_index + 1}_encoder_input",
    )
    hidden = layers.Dense(
        rnn_units,
        name=f"node_{channel_index + 1}_encoder_dense",
    )(encoder_inputs)
    hidden = layers.LayerNormalization(
        name=f"node_{channel_index + 1}_encoder_normalization",
    )(hidden)
    hidden = layers.Activation(
        "relu",
        name=f"node_{channel_index + 1}_encoder_activation",
    )(hidden)
    latent_mean = layers.Dense(
        latent_dim,
        name=f"node_{channel_index + 1}_latent_mean",
    )(hidden)
    latent_log_variance = layers.Dense(
        latent_dim,
        name=f"node_{channel_index + 1}_latent_log_variance",
    )(hidden)
    latent_sample = layers.Lambda(
        sampling,
        output_shape=(latent_dim,),
        name=f"node_{channel_index + 1}_latent_sample",
    )([latent_mean, latent_log_variance])
    return models.Model(
        encoder_inputs,
        [latent_mean, latent_log_variance, latent_sample],
        name=f"node_{channel_index + 1}_encoder",
    )


def _build_channel_decoder(
    channel_index: int,
    latent_dim: int,
    output_dim: int,
    rnn_units: int,
) -> tf.keras.Model:
    """Build one node-specific latent-to-recurrent-state decoder."""
    latent_inputs = layers.Input(
        shape=(latent_dim,),
        name=f"node_{channel_index + 1}_decoder_input",
    )
    hidden = layers.Dense(
        rnn_units,
        name=f"node_{channel_index + 1}_decoder_dense",
    )(latent_inputs)
    hidden = layers.LayerNormalization(
        name=f"node_{channel_index + 1}_decoder_normalization",
    )(hidden)
    hidden = layers.Activation(
        "relu",
        name=f"node_{channel_index + 1}_decoder_activation",
    )(hidden)
    outputs = layers.Dense(
        output_dim,
        activation="linear",
        name=f"node_{channel_index + 1}_decoded_state",
    )(hidden)
    return models.Model(
        latent_inputs,
        outputs,
        name=f"node_{channel_index + 1}_decoder",
    )


class IndependentNodeChannelModel(tf.keras.Model):
    """Variational planning model with one isolated channel per reward node."""

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
        kl_aggregation: str = "sum",
        kl_time_weighting: str = "remaining_steps",
        parameter_multiplier: float | None = None,
        action_entropy_bonus: float = 0.0,
    ) -> None:
        """Initialize independent node channels and the two global readouts.

        Args:
            rnn_units: Hidden and cell width within every node channel.
            latent_dim: Latent bottleneck width within every node channel.
            time_steps: Number of reward nodes and therefore channel count.
            num_paths: Number of available path actions.
            index_path_map: One-based node indices for each candidate path.
            alpha: Reconstruction-loss weight before tree-size scaling.
            lambda_: Total channel-KL weight before tree-size scaling when
                ``kl_aggregation`` is ``sum``.
            reward_normalization_constant: Actor reward normalizer.
            action_gradient_confidence_threshold: Confidence at which optional
                backward-only actor-gradient scaling begins.
            max_action_gradient_scale: Maximum actor-gradient multiplier.
            reconstruction_head_type: ``nonlinear_location_scale`` or
                ``nonlinear_categorical``.
            reconstruction_hidden_units: Hidden width of the global
                reconstruction head.
            kl_aggregation: ``sum`` for total independent-channel KL or
                ``mean`` to reproduce the earlier per-channel average.
            kl_time_weighting: ``remaining_steps`` multiplies successive
                channel KL values by ``time_steps, ..., 1``; ``uniform``
                assigns every channel weight 1 to reproduce earlier runs.
            parameter_multiplier: Optional common multiplier for
                reconstruction, total KL, and action loss. ``None`` uses the
                tree-size default.
            action_entropy_bonus: Non-negative entropy-bonus coefficient in
                the actor objective.
        """
        super().__init__(name="independent_node_channel_model")

        if time_steps < 1:
            raise ValueError("time_steps must be positive.")
        if rnn_units < 1 or latent_dim < 1:
            raise ValueError("rnn_units and latent_dim must be positive.")
        if num_paths < 1:
            raise ValueError("num_paths must be positive.")
        if not 0.0 <= action_gradient_confidence_threshold < 1.0:
            raise ValueError(
                "action_gradient_confidence_threshold must be in [0, 1)."
            )
        if max_action_gradient_scale < 1.0:
            raise ValueError("max_action_gradient_scale must be at least 1.")
        valid_head_types = {
            "nonlinear_categorical",
            "nonlinear_location_scale",
        }
        if reconstruction_head_type not in valid_head_types:
            raise ValueError(
                "reconstruction_head_type must be nonlinear_categorical "
                "or nonlinear_location_scale."
            )
        if reconstruction_hidden_units < 1:
            raise ValueError(
                "reconstruction_hidden_units must be a positive integer."
            )
        if kl_aggregation not in {"sum", "mean"}:
            raise ValueError("kl_aggregation must be sum or mean.")
        if kl_time_weighting not in {"remaining_steps", "uniform"}:
            raise ValueError(
                "kl_time_weighting must be remaining_steps or uniform."
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
        self.num_channels = int(time_steps)
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
        self.kl_aggregation = kl_aggregation
        self.kl_time_weighting = kl_time_weighting
        if kl_time_weighting == "remaining_steps":
            self.channel_kl_time_weights = tuple(
                range(self.num_channels, 0, -1)
            )
        else:
            self.channel_kl_time_weights = tuple(
                1 for _ in range(self.num_channels)
            )
        self._channel_kl_time_weights_tensor = tf.constant(
            self.channel_kl_time_weights,
            dtype=tf.float32,
        )
        self.reconstruction_loss_scope = "all_channels"
        self.reconstruction_min_reward = helper.RECONSTRUCTION_MIN_REWARD
        self.reconstruction_max_reward = helper.RECONSTRUCTION_MAX_REWARD
        self.num_categories = helper.NUM_RECONSTRUCTION_CATEGORIES
        self.critic_loss_enabled = False
        self.model_architecture = "independent_node_channels"
        self.returns_reconstruction_location_scale_parameters = True

        self.channel_rnn_cells = [
            NoForgetLSTMCell(
                self.rnn_units,
                kernel_initializer="orthogonal",
                recurrent_initializer="orthogonal",
                unit_forget_bias=False,
                name=f"node_{channel_index + 1}_rnn",
            )
            for channel_index in range(self.num_channels)
        ]
        self.channel_encoders = [
            _build_channel_encoder(
                channel_index,
                input_dim=2 * self.rnn_units,
                latent_dim=self.latent_dim,
                rnn_units=self.rnn_units,
            )
            for channel_index in range(self.num_channels)
        ]
        self.channel_decoders = [
            _build_channel_decoder(
                channel_index,
                latent_dim=self.latent_dim,
                output_dim=2 * self.rnn_units,
                rnn_units=self.rnn_units,
            )
            for channel_index in range(self.num_channels)
        ]

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

        # Every row is a different channel prior; no prior parameter is shared
        # between nodes even though the variables are stored compactly.
        self.prior_mu = self.add_weight(
            name="channel_prior_mu",
            shape=(self.num_channels, self.latent_dim),
            initializer="zeros",
            trainable=True,
        )
        self.prior_logvar = self.add_weight(
            name="channel_prior_logvar",
            shape=(self.num_channels, self.latent_dim),
            initializer="zeros",
            trainable=True,
        )

    @staticmethod
    def _parameter_scale_for_time_steps(time_steps: int) -> float:
        """Return the current tree-size loss scaling."""
        if time_steps == 12:
            return 1.0
        if time_steps == 30:
            return 100.0
        return 1.0

    @property
    def shared_representation_trainable_variables(
        self,
    ) -> list[tf.Variable]:
        """Return all independent channel and prior variables."""
        variables: list[tf.Variable] = []
        for channel_layers in (
            self.channel_rnn_cells,
            self.channel_encoders,
            self.channel_decoders,
        ):
            for channel_layer in channel_layers:
                variables.extend(channel_layer.trainable_variables)
        return variables + [self.prior_mu, self.prior_logvar]

    @property
    def critic_trainable_variables(self) -> list[tf.Variable]:
        """Return no variables because this architecture has no critic."""
        return []

    def _decode_location_scale_parameters(
        self,
        reconstruction_outputs: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Convert global head outputs to per-node location and scale."""
        batch_size = tf.shape(reconstruction_outputs)[0]
        distribution_parameters = tf.reshape(
            reconstruction_outputs,
            [batch_size, self.time_steps, 2],
        )
        location = float(self.reconstruction_max_reward) * tf.math.tanh(
            distribution_parameters[:, :, 0:1]
        )
        scale = tf.exp(distribution_parameters[:, :, 1:2]) + 1e-4
        return location, scale

    def reconstruction_location_scale_parameters(
        self,
        concatenated_hidden_states: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Return per-node parameters from all decoded channel states."""
        if self.reconstruction_head_type != "nonlinear_location_scale":
            raise ValueError(
                "Location and scale are available only for the "
                "nonlinear_location_scale reconstruction head."
            )
        outputs = self.reconstruction_head(concatenated_hidden_states)
        return self._decode_location_scale_parameters(outputs)

    def _reconstruction_probabilities(
        self,
        concatenated_hidden_states: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Decode probabilities and expose existing location-scale outputs."""
        reconstruction_outputs = self.reconstruction_head(
            concatenated_hidden_states
        )
        batch_size = tf.shape(concatenated_hidden_states)[0]
        if self.reconstruction_head_type == "nonlinear_categorical":
            category_logits = tf.reshape(
                reconstruction_outputs,
                [batch_size, self.time_steps, self.num_categories],
            )
            unavailable = tf.constant(float("nan"), dtype=tf.float32)
            return (
                tf.nn.softmax(category_logits, axis=-1),
                unavailable,
                unavailable,
            )

        location, scale = self._decode_location_scale_parameters(
            reconstruction_outputs
        )
        bin_edges = tf.range(
            float(self.reconstruction_min_reward) - 0.5,
            float(self.reconstruction_max_reward) + 1.5,
            delta=1.0,
            dtype=tf.float32,
        )
        bin_edges = tf.reshape(
            bin_edges,
            [1, 1, self.num_categories + 1],
        )
        cdf_at_edges = tf.math.sigmoid((bin_edges - location) / scale)
        probabilities = cdf_at_edges[:, :, 1:] - cdf_at_edges[:, :, :-1]
        probabilities = tf.reverse(probabilities, axis=[-1]) + 1e-6
        return (
            probabilities
            / tf.reduce_sum(
                probabilities,
                axis=-1,
                keepdims=True,
            ),
            location,
            scale,
        )

    def _channel_prior(
        self,
        channel_index: int,
        batch_size: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Return one channel's independently learned Gaussian prior."""
        prior_mean = tf.broadcast_to(
            self.prior_mu[channel_index],
            [batch_size, self.latent_dim],
        )
        prior_variance = tf.exp(
            tf.broadcast_to(
                self.prior_logvar[channel_index],
                [batch_size, self.latent_dim],
            )
        )
        return prior_mean, prior_variance

    @staticmethod
    def _calculate_kl_loss(
        latent_mean: tf.Tensor,
        latent_log_variance: tf.Tensor,
        prior_mean: tf.Tensor,
        prior_variance: tf.Tensor,
        epsilon: float = 1e-6,
    ) -> tf.Tensor:
        """Calculate one channel's batch-averaged Gaussian KL divergence."""
        prior_variance = prior_variance + epsilon
        prior_log_variance = tf.math.log(prior_variance)
        latent_variance = tf.exp(latent_log_variance) + epsilon
        log_variance_ratio = latent_log_variance - prior_log_variance
        per_example_kl = -0.5 * tf.reduce_mean(
            1
            + log_variance_ratio
            - (
                (
                    tf.square(latent_mean - prior_mean)
                    + latent_variance
                )
                / prior_variance
            ),
            axis=1,
        )
        return tf.reduce_mean(per_example_kl)

    @staticmethod
    def _stacked_dense(
        inputs: tf.Tensor,
        dense_layers: list[layers.Dense],
    ) -> tf.Tensor:
        """Apply channel-specific dense layers in one batched operation.

        The kernels and biases remain owned by the original per-channel Keras
        layers. Stacking them here creates only transient tensors, so saved
        weight names, shapes, ordering, and gradient destinations are
        unchanged.
        """
        kernels = tf.stack(
            [
                tf.convert_to_tensor(dense_layer.kernel)
                for dense_layer in dense_layers
            ],
            axis=0,
        )
        outputs = tf.einsum("bci,cio->bco", inputs, kernels)
        if dense_layers[0].use_bias:
            biases = tf.stack(
                [
                    tf.convert_to_tensor(dense_layer.bias)
                    for dense_layer in dense_layers
                ],
                axis=0,
            )
            outputs = outputs + biases[tf.newaxis, :, :]
        return outputs

    @staticmethod
    def _stacked_layer_normalization(
        inputs: tf.Tensor,
        normalization_layers: list[layers.LayerNormalization],
    ) -> tf.Tensor:
        """Apply independent channel LayerNorms without a Python loop."""
        reference_layer = normalization_layers[0]
        mean, variance = tf.nn.moments(
            inputs,
            axes=[-1],
            keepdims=True,
        )
        normalized = (inputs - mean) * tf.math.rsqrt(
            variance
            + tf.cast(reference_layer.epsilon, inputs.dtype)
        )
        if reference_layer.scale:
            gamma = tf.stack(
                [
                    tf.convert_to_tensor(layer.gamma)
                    for layer in normalization_layers
                ],
                axis=0,
            )
            normalized = normalized * gamma[tf.newaxis, :, :]
        if reference_layer.center:
            beta = tf.stack(
                [
                    tf.convert_to_tensor(layer.beta)
                    for layer in normalization_layers
                ],
                axis=0,
            )
            normalized = normalized + beta[tf.newaxis, :, :]
        return normalized

    def _stacked_zero_state_rnn(
        self,
        channel_inputs: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Evaluate all one-step zero-state RNN channels together."""
        input_shape = tf.TensorShape([None, self.num_categories])
        for cell in self.channel_rnn_cells:
            if not cell.built:
                cell.build(input_shape)

        kernels = tf.stack(
            [
                tf.convert_to_tensor(cell.kernel)
                for cell in self.channel_rnn_cells
            ],
            axis=0,
        )
        input_kernels, _, candidate_kernels, output_kernels = tf.split(
            kernels,
            4,
            axis=-1,
        )
        active_kernels = tf.concat(
            [input_kernels, candidate_kernels, output_kernels],
            axis=-1,
        )
        gate_preactivations = tf.einsum(
            "bci,cio->bco",
            channel_inputs,
            active_kernels,
        )

        if self.channel_rnn_cells[0].use_bias:
            biases = tf.stack(
                [
                    tf.convert_to_tensor(cell.bias)
                    for cell in self.channel_rnn_cells
                ],
                axis=0,
            )
            input_biases, _, candidate_biases, output_biases = tf.split(
                biases,
                4,
                axis=-1,
            )
            active_biases = tf.concat(
                [input_biases, candidate_biases, output_biases],
                axis=-1,
            )
            gate_preactivations = (
                gate_preactivations
                + active_biases[tf.newaxis, :, :]
            )

        (
            input_preactivation,
            candidate_preactivation,
            output_preactivation,
        ) = tf.split(gate_preactivations, 3, axis=-1)
        reference_cell = self.channel_rnn_cells[0]
        input_gate = reference_cell.recurrent_activation(
            input_preactivation
        )
        cell_state = input_gate * reference_cell.activation(
            candidate_preactivation
        )
        hidden_state = reference_cell.recurrent_activation(
            output_preactivation
        ) * reference_cell.activation(cell_state)
        return hidden_state, cell_state

    def _vectorized_channel_forward(
        self,
        categories_onehot: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Run every independent bottleneck with transient stacked weights."""
        channel_inputs = tf.squeeze(categories_onehot, axis=2)
        hidden_state, cell_state = self._stacked_zero_state_rnn(
            channel_inputs
        )
        encoder_inputs = tf.concat(
            [hidden_state, cell_state],
            axis=-1,
        )

        encoder_dense_layers = [
            encoder.get_layer(f"node_{channel_index + 1}_encoder_dense")
            for channel_index, encoder in enumerate(self.channel_encoders)
        ]
        encoder_normalization_layers = [
            encoder.get_layer(
                f"node_{channel_index + 1}_encoder_normalization"
            )
            for channel_index, encoder in enumerate(self.channel_encoders)
        ]
        latent_mean_layers = [
            encoder.get_layer(f"node_{channel_index + 1}_latent_mean")
            for channel_index, encoder in enumerate(self.channel_encoders)
        ]
        latent_log_variance_layers = [
            encoder.get_layer(
                f"node_{channel_index + 1}_latent_log_variance"
            )
            for channel_index, encoder in enumerate(self.channel_encoders)
        ]

        encoder_hidden = self._stacked_dense(
            encoder_inputs,
            encoder_dense_layers,
        )
        encoder_hidden = self._stacked_layer_normalization(
            encoder_hidden,
            encoder_normalization_layers,
        )
        encoder_hidden = tf.nn.relu(encoder_hidden)
        latent_mean = self._stacked_dense(
            encoder_hidden,
            latent_mean_layers,
        )
        latent_log_variance = self._stacked_dense(
            encoder_hidden,
            latent_log_variance_layers,
        )
        epsilon = tf.random.normal(
            shape=tf.shape(latent_mean),
            dtype=latent_mean.dtype,
        )
        latent_sample = (
            latent_mean
            + tf.exp(0.5 * latent_log_variance) * epsilon
        )

        decoder_dense_layers = [
            decoder.get_layer(f"node_{channel_index + 1}_decoder_dense")
            for channel_index, decoder in enumerate(self.channel_decoders)
        ]
        decoder_normalization_layers = [
            decoder.get_layer(
                f"node_{channel_index + 1}_decoder_normalization"
            )
            for channel_index, decoder in enumerate(self.channel_decoders)
        ]
        decoded_state_layers = [
            decoder.get_layer(f"node_{channel_index + 1}_decoded_state")
            for channel_index, decoder in enumerate(self.channel_decoders)
        ]
        decoder_hidden = self._stacked_dense(
            latent_sample,
            decoder_dense_layers,
        )
        decoder_hidden = self._stacked_layer_normalization(
            decoder_hidden,
            decoder_normalization_layers,
        )
        decoder_hidden = tf.nn.relu(decoder_hidden)
        decoded_state = self._stacked_dense(
            decoder_hidden,
            decoded_state_layers,
        )
        reconstructed_hidden_state = decoded_state[:, :, : self.rnn_units]

        prior_variance = tf.exp(self.prior_logvar)[tf.newaxis, :, :] + 1e-6
        prior_log_variance = tf.math.log(prior_variance)
        latent_variance = tf.exp(latent_log_variance) + 1e-6
        per_example_channel_kl = -0.5 * tf.reduce_mean(
            1
            + latent_log_variance
            - prior_log_variance
            - (
                (
                    tf.square(
                        latent_mean - self.prior_mu[tf.newaxis, :, :]
                    )
                    + latent_variance
                )
                / prior_variance
            ),
            axis=-1,
        )
        mean_channel_kl = tf.reduce_mean(per_example_channel_kl, axis=0)
        information_cost = tf.reduce_sum(
            mean_channel_kl
            * tf.cast(
                self._channel_kl_time_weights_tensor,
                mean_channel_kl.dtype,
            )
        )
        return latent_mean, reconstructed_hidden_state, information_cost

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = True,
        current_alpha: float | tf.Tensor = 1.0,
        current_kl_weight: float | tf.Tensor = 1.0,
        current_critic_coef: float | tf.Tensor = 0.0,
    ) -> tuple[tf.Tensor, ...]:
        """Run all independent channels and compute comparison losses."""
        del current_critic_coef
        batch_size = tf.shape(inputs)[0]
        categories_onehot = helper.scalar_to_categorical(
            inputs,
            self.num_categories,
        )
        (
            all_latent_means,
            reconstructed_hidden_states,
            information_cost,
        ) = self._vectorized_channel_forward(categories_onehot)
        concatenated_hidden_states = tf.reshape(
            reconstructed_hidden_states,
            [batch_size, self.num_channels * self.rnn_units],
        )
        (
            reconstruction_probabilities,
            reconstruction_locations,
            reconstruction_scales,
        ) = self._reconstruction_probabilities(concatenated_hidden_states)
        action_logits = self.action_head(concatenated_hidden_states)
        action_probabilities = tf.nn.softmax(action_logits, axis=-1)

        target_categories = tf.squeeze(categories_onehot, axis=2)
        raw_cross_entropy = -tf.reduce_sum(
            target_categories
            * tf.math.log(reconstruction_probabilities + 1e-7),
            axis=-1,
        )
        reconstruction_loss = tf.reduce_mean(raw_cross_entropy) / tf.math.log(
            tf.cast(self.num_categories, tf.float32)
        )
        action_loss = self.compute_final_actor_loss(
            inputs,
            action_probabilities,
        )
        if self.kl_aggregation == "sum":
            information_loss = information_cost
        else:
            information_loss = information_cost / tf.cast(
                self.num_channels,
                tf.float32,
            )
        critic_loss = tf.constant(0.0, dtype=tf.float32)

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
            )
            second_decoder_loss = reconstruction_loss
            third_decoder_loss = action_loss

        # A singleton readout dimension preserves the existing inference
        # contract: callers obtain the joint reconstruction with ``[:, -1]``.
        category_outputs = reconstruction_probabilities[:, tf.newaxis, :, :]
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
            concatenated_hidden_states,
            reconstruction_locations,
            reconstruction_scales,
        )

    def calculate_path_rewards(self, rewards: tf.Tensor) -> tf.Tensor:
        """Return summed rewards for each configured candidate path."""
        batch_size = tf.shape(rewards)[0]
        path_rewards = []
        for node_indices in self.index_path_map.values():
            zero_based_indices = (
                tf.convert_to_tensor(node_indices, dtype=tf.int32) - 1
            )
            zero_based_indices = tf.clip_by_value(
                zero_based_indices,
                0,
                tf.shape(rewards)[1] - 1,
            )
            tiled_indices = tf.tile(
                zero_based_indices[tf.newaxis, :],
                [batch_size, 1],
            )
            gathered_rewards = tf.gather(
                rewards,
                tiled_indices,
                axis=1,
                batch_dims=1,
            )
            path_rewards.append(tf.reduce_sum(gathered_rewards, axis=1))
        return tf.stack(path_rewards, axis=0)

    def _prepare_path_rewards(
        self,
        inputs: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Prepare finite raw and normalized path rewards for the actor."""
        finite_inputs = tf.where(
            tf.math.is_finite(inputs),
            inputs,
            tf.zeros_like(inputs),
        )
        batch_size = tf.shape(finite_inputs)[0]
        path_rewards = self.calculate_path_rewards(finite_inputs)
        path_rewards = tf.transpose(path_rewards, perm=[1, 0, 2])
        path_rewards = tf.reshape(
            path_rewards,
            [batch_size, self.num_paths],
        )
        return path_rewards, path_rewards

    def action_gradient_scale(self, action_probs: tf.Tensor) -> tf.Tensor:
        """Return the existing confidence-dependent actor gradient scale."""
        confidence = tf.reduce_max(action_probs, axis=-1, keepdims=True)
        remaining_probability = tf.maximum(
            1.0 - confidence,
            tf.constant(1e-6, dtype=action_probs.dtype),
        )
        threshold_remaining = tf.cast(
            1.0 - self.action_gradient_confidence_threshold,
            action_probs.dtype,
        )
        scale = threshold_remaining / remaining_probability
        return tf.clip_by_value(
            scale,
            tf.cast(1.0, action_probs.dtype),
            tf.cast(self.max_action_gradient_scale, action_probs.dtype),
        )

    def scale_confident_action_gradients(
        self,
        action_probs: tf.Tensor,
    ) -> tf.Tensor:
        """Keep action values fixed while optionally scaling their gradient."""
        stopped_probabilities = tf.stop_gradient(action_probs)
        scale = tf.stop_gradient(self.action_gradient_scale(action_probs))
        return stopped_probabilities + scale * (
            action_probs - stopped_probabilities
        )

    def compute_final_actor_loss(
        self,
        inputs: tf.Tensor,
        action_probs: tf.Tensor,
    ) -> tf.Tensor:
        """Compute the same expected-path-reward objective as the main model."""
        _, path_rewards = self._prepare_path_rewards(inputs)
        gradient_scaled_probabilities = (
            self.scale_confident_action_gradients(action_probs)
        )
        expected_reward = tf.reduce_sum(
            gradient_scaled_probabilities * path_rewards,
            axis=1,
            keepdims=True,
        )
        mean_expected_reward = tf.reduce_mean(expected_reward)
        entropy = -tf.reduce_sum(
            action_probs * tf.math.log(action_probs + 1e-8),
            axis=1,
            keepdims=True,
        )

        reward_normalizer = self.reward_normalization_constant
        if reward_normalizer is None:
            if self.time_steps == 30:
                if self.num_paths not in {16, 25}:
                    raise ValueError(
                        "A 31-node actor must have either 16 deep-tree "
                        "paths or 25 wide-tree paths."
                    )
                reward_normalizer = 8.157443089523166
            else:
                reward_normalizer = {
                    2: 0.75,
                    6: 3.58,
                    12: 5.11,
                    39: 8.0682,
                }.get(self.time_steps, 8.157443089523166)
        normalized_expected_reward = mean_expected_reward / reward_normalizer
        action_loss = (1.0 - normalized_expected_reward) - (
            self.action_entropy_bonus * tf.reduce_mean(entropy)
        )
        return tf.where(
            tf.math.is_finite(action_loss),
            action_loss,
            tf.constant(0.0, dtype=tf.float32),
        )
