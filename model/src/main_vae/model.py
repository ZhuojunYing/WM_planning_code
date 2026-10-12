"""Neural network components for the variational recurrent planning model."""

from __future__ import annotations

import math
from typing import Any

import tensorflow as tf
from tensorflow.keras import layers, models

from . import helper


def sampling(args: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    """Sample latent states with the reparameterization trick.

    Args:
        args: Tuple containing the latent mean and log variance tensors.

    Returns:
        A sampled latent tensor with the same shape as the latent mean.
    """
    latent_mean, latent_log_variance = args
    batch_size = tf.shape(latent_mean)[0]
    latent_dim = tf.shape(latent_mean)[1]
    epsilon = tf.random.normal(shape=(batch_size, latent_dim))
    return latent_mean + tf.exp(0.5 * latent_log_variance) * epsilon


def build_encoder(
    input_dim: int,
    latent_dim: int,
    rnn_units: int,
) -> tf.keras.Model:
    """Build the encoder network for the latent posterior.

    Args:
        input_dim: Number of input features to the encoder.
        latent_dim: Number of latent dimensions.
        rnn_units: Number of hidden units in the dense projection.

    Returns:
        A Keras model that outputs latent mean, log variance, and sample.
    """
    encoder_inputs = layers.Input(shape=(input_dim,))

    hidden = layers.Dense(rnn_units)(encoder_inputs)
    hidden = layers.LayerNormalization()(hidden)
    hidden = layers.Activation("relu")(hidden)

    latent_mean = layers.Dense(latent_dim)(hidden)
    latent_log_variance = layers.Dense(latent_dim)(hidden)
    latent_sample = layers.Lambda(
        sampling,
        output_shape=(latent_dim,),
    )([latent_mean, latent_log_variance])

    return models.Model(
        encoder_inputs,
        [latent_mean, latent_log_variance, latent_sample],
        name="encoder",
    )


def build_decoder(
    latent_dim: int,
    output_dim: int,
    rnn_units: int,
) -> tf.keras.Model:
    """Build the decoder network for recurrent hidden-state updates.

    Args:
        latent_dim: Number of latent dimensions.
        output_dim: Number of output features produced by the decoder.
        rnn_units: Number of hidden units in the dense projection.

    Returns:
        A Keras model that maps latent samples to decoder outputs.
    """
    latent_inputs = layers.Input(shape=(latent_dim,))

    hidden = layers.Dense(rnn_units)(latent_inputs)
    hidden = layers.LayerNormalization()(hidden)
    hidden = layers.Activation("relu")(hidden)

    outputs = layers.Dense(output_dim, activation="linear")(hidden)
    return models.Model(latent_inputs, outputs, name="decoder")


class NoForgetLSTMCell(tf.keras.layers.LSTMCell):
    """LSTM cell whose forget multiplier is fixed to one.

    The standard four-gate weight layout is retained so the surrounding model
    and optimizer continue to treat this as an LSTM cell. The learned forget
    preactivation is deliberately ignored, giving the cell-state update
    ``c_t = c_(t-1) + i_t * candidate_t``.
    """

    def call(
        self,
        inputs: tf.Tensor,
        states: tuple[tf.Tensor, tf.Tensor],
        training: bool = False,
    ) -> tuple[tf.Tensor, list[tf.Tensor]]:
        """Process one recurrent step without multiplicative forgetting."""
        del training

        previous_hidden_state, previous_cell_state = states
        gate_preactivations = (
            tf.matmul(inputs, self.kernel)
            + tf.matmul(previous_hidden_state, self.recurrent_kernel)
        )
        if self.use_bias:
            gate_preactivations = tf.nn.bias_add(
                gate_preactivations,
                self.bias,
            )

        (
            input_preactivation,
            _forget_preactivation,
            candidate_preactivation,
            output_preactivation,
        ) = tf.split(gate_preactivations, 4, axis=1)

        input_gate = self.recurrent_activation(input_preactivation)
        candidate_state = self.activation(candidate_preactivation)
        output_gate = self.recurrent_activation(output_preactivation)

        cell_state = (
            previous_cell_state
            + input_gate * candidate_state
        )
        hidden_state = output_gate * self.activation(cell_state)

        return hidden_state, [hidden_state, cell_state]

    def call_from_zero_state(
        self,
        inputs: tf.Tensor,
        training: bool = False,
    ) -> tuple[tf.Tensor, list[tf.Tensor]]:
        """Process the first and only step without zero-valued operations.

        Independent node channels always start from exactly zero hidden and
        cell states and process one observation. Their recurrent-kernel
        product, previous-cell addition, and unused forget preactivation are
        therefore mathematically irrelevant. This method retains the ordinary
        cell variables for saved-weight compatibility while skipping those
        operations.
        """
        del training
        if not self.built:
            self.build(inputs.shape)

        (
            input_kernel,
            _forget_kernel,
            candidate_kernel,
            output_kernel,
        ) = tf.split(self.kernel, 4, axis=1)
        active_kernel = tf.concat(
            [input_kernel, candidate_kernel, output_kernel],
            axis=1,
        )
        gate_preactivations = tf.matmul(inputs, active_kernel)
        if self.use_bias:
            (
                input_bias,
                _forget_bias,
                candidate_bias,
                output_bias,
            ) = tf.split(self.bias, 4, axis=0)
            active_bias = tf.concat(
                [input_bias, candidate_bias, output_bias],
                axis=0,
            )
            gate_preactivations = tf.nn.bias_add(
                gate_preactivations,
                active_bias,
            )

        (
            input_preactivation,
            candidate_preactivation,
            output_preactivation,
        ) = tf.split(gate_preactivations, 3, axis=1)
        input_gate = self.recurrent_activation(input_preactivation)
        candidate_state = self.activation(candidate_preactivation)
        output_gate = self.recurrent_activation(output_preactivation)
        cell_state = input_gate * candidate_state
        hidden_state = output_gate * self.activation(cell_state)
        return hidden_state, [hidden_state, cell_state]


class VariationalRNN(tf.keras.Model):
    """Variational recurrent planning model."""

    def __init__(
        self,
        encoder: tf.keras.Model,
        decoder: tf.keras.Model,
        rnn_units: int,
        latent_dim: int,
        time_steps: int,
        num_paths: int,
        index_path_map: dict[Any, list[int]],
        path_map: dict[Any, Any],
        path_cov_mat: Any,
        alpha: float = 0.0,
        lambda_: float = 1.0,
        tree_type: str = "deep",
        reward_normalization_constant: float | None = None,
        action_gradient_confidence_threshold: float = 0.90,
        max_action_gradient_scale: float = 1.0,
        reconstruction_head_type: str = "nonlinear_location_scale",
        reconstruction_hidden_units: int = 128,
        reconstruction_loss_scope: str = "final_timestep",
        critic_loss_enabled: bool = True,
        parameter_multiplier: float | None = None,
        action_entropy_bonus: float = 0.0,
        model_condition: str = "planning",
    ) -> None:
        """Initialize the variational recurrent planning model.

        Args:
            encoder: Encoder network that parameterizes the latent posterior.
            decoder: Decoder network that updates recurrent hidden states.
            rnn_units: Number of hidden units in the recurrent state.
            latent_dim: Number of latent dimensions.
            time_steps: Number of observed sequence positions.
            num_paths: Number of candidate paths in the tree.
            index_path_map: Mapping from path identifiers to node indices.
            path_map: Mapping that defines the tree paths.
            path_cov_mat: Path covariance matrix retained for compatibility.
            alpha: Reconstruction loss weight before time-step scaling.
            lambda_: KL/information loss weight before time-step scaling.
            tree_type: Tree topology label retained for compatibility.
            reward_normalization_constant: Expected maximum path reward used
                to normalize the actor objective.
            action_gradient_confidence_threshold: Path-policy confidence above
                which the backward actor gradient is amplified.
            max_action_gradient_scale: Maximum backward-only actor-gradient
                multiplier.
            reconstruction_head_type: Reconstruction output family. The
                default ``nonlinear_location_scale`` uses a discretized
                logistic location and scale; ``nonlinear_categorical`` uses
                eleven unrestricted logits.
            reconstruction_hidden_units: Width of the reconstruction head's
                ReLU hidden layer.
            reconstruction_loss_scope: Whether reconstruction loss supervises
                every observed-prefix output or only the final recurrent
                output.
            critic_loss_enabled: Whether the auxiliary critic loss trains the
                shared representation and critic head.
            parameter_multiplier: Optional explicit common multiplier for the
                reconstruction, KL, action, and shared critic weights. ``None``
                uses the model's tree-size-dependent default.
            action_entropy_bonus: Non-negative coefficient multiplying path
                policy entropy in the actor objective.
            model_condition: ``planning`` trains the actor and optional critic;
                ``exposure`` trains only reconstruction and KL objectives.
        """
        super().__init__()

        self.parameter_multiplier_explicit = parameter_multiplier is not None
        if parameter_multiplier is None:
            parameter_scale = self._parameter_scale_for_time_steps(time_steps)
        else:
            parameter_scale = float(parameter_multiplier)
            if not math.isfinite(parameter_scale) or parameter_scale <= 0.0:
                raise ValueError(
                    "parameter_multiplier must be finite and positive."
                )

        self.encoder = encoder
        self.decoder = decoder
        self.rnn_units = rnn_units
        self.time_steps = time_steps
        self.num_paths = num_paths
        self.latent_dim = latent_dim
        self.alpha = alpha * parameter_scale
        self.lambda_ = lambda_ * parameter_scale
        if model_condition not in {"planning", "exposure"}:
            raise ValueError(
                "model_condition must be planning or exposure."
            )
        self.model_condition = model_condition
        self.action_loss_enabled = model_condition == "planning"
        self.action_loss_weight = (
            1.0 * parameter_scale if self.action_loss_enabled else 0.0
        )
        self.parameter_scale = parameter_scale
        self.parameter_multiplier = parameter_scale
        if (
            not math.isfinite(action_entropy_bonus)
            or action_entropy_bonus < 0.0
        ):
            raise ValueError(
                "action_entropy_bonus must be finite and non-negative."
            )
        self.action_entropy_bonus = float(action_entropy_bonus)
        if not self.action_loss_enabled and self.action_entropy_bonus != 0.0:
            raise ValueError(
                "action_entropy_bonus must be 0 for exposure models."
            )
        self.reconstruction_min_reward = helper.RECONSTRUCTION_MIN_REWARD
        self.reconstruction_max_reward = helper.RECONSTRUCTION_MAX_REWARD
        self.num_categories = helper.NUM_RECONSTRUCTION_CATEGORIES
        self.tree_type = tree_type
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
        if not 0.0 <= action_gradient_confidence_threshold < 1.0:
            raise ValueError(
                "action_gradient_confidence_threshold must be in [0, 1)."
            )
        if max_action_gradient_scale < 1.0:
            raise ValueError("max_action_gradient_scale must be at least 1.")
        self.action_gradient_confidence_threshold = float(
            action_gradient_confidence_threshold
        )
        self.max_action_gradient_scale = float(max_action_gradient_scale)
        self.critic_loss_enabled = bool(
            critic_loss_enabled and self.action_loss_enabled
        )
        self.model_architecture = "sequential_shared_rnn"
        self.returns_reconstruction_location_scale_parameters = True
        self.index_path_map = index_path_map
        self.path_map = path_map
        self.path_cov_mat = path_cov_mat

        valid_reconstruction_head_types = {
            "nonlinear_categorical",
            "nonlinear_location_scale",
        }
        if reconstruction_head_type not in valid_reconstruction_head_types:
            raise ValueError(
                "reconstruction_head_type must be nonlinear_categorical "
                "or nonlinear_location_scale."
            )
        self.reconstruction_head_type = reconstruction_head_type
        if (
            isinstance(reconstruction_hidden_units, bool)
            or not isinstance(reconstruction_hidden_units, int)
            or reconstruction_hidden_units < 1
        ):
            raise ValueError(
                "reconstruction_hidden_units must be a positive integer."
            )
        valid_reconstruction_loss_scopes = {
            "all_timesteps",
            "final_timestep",
        }
        if reconstruction_loss_scope not in valid_reconstruction_loss_scopes:
            raise ValueError(
                "reconstruction_loss_scope must be all_timesteps or "
                "final_timestep."
            )
        self.reconstruction_hidden_units = reconstruction_hidden_units
        self.reconstruction_loss_scope = reconstruction_loss_scope
        reconstruction_output_units = time_steps * (
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
            else tf.keras.initializers.RandomNormal(
                mean=0.0,
                stddev=0.01,
            )
        )
        self.reconstruction_head = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    reconstruction_hidden_units,
                    activation="relu",
                    name="hidden",
                ),
                tf.keras.layers.Dense(
                    reconstruction_output_units,
                    activation="linear",
                    kernel_initializer=reconstruction_output_initializer,
                    bias_initializer="zeros",
                    name=reconstruction_output_name,
                ),
            ],
            name=f"{reconstruction_head_type}_reconstruction_head",
        )
        self.action_head = tf.keras.layers.Dense(
            num_paths,
            activation=None,
            kernel_initializer="glorot_uniform",
        )
        self.lstm_cell = NoForgetLSTMCell(
            self.rnn_units,
            kernel_initializer="orthogonal",
            recurrent_initializer="orthogonal",
            unit_forget_bias=False,
        )
        self.critic_head = tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer="glorot_uniform",
            name="critic_head",
        )

        self.prior_mu = self.add_weight(
            name="prior_mu",
            shape=(time_steps, latent_dim),
            initializer="zeros",
            trainable=True,
        )
        self.prior_logvar = self.add_weight(
            name="prior_logvar",
            shape=(time_steps, latent_dim),
            initializer="zeros",
            trainable=True,
        )

    @property
    def shared_representation_trainable_variables(
        self,
    ) -> list[tf.Variable]:
        """Return variables trained through the shared representation loss."""
        return (
            self.decoder.trainable_variables
            + self.lstm_cell.trainable_variables
            + self.encoder.trainable_variables
            + [self.prior_mu, self.prior_logvar]
        )

    @property
    def critic_trainable_variables(self) -> list[tf.Variable]:
        """Return critic variables through the generic trainer interface."""
        return self.critic_head.trainable_variables

    @staticmethod
    def _parameter_scale_for_time_steps(time_steps: int) -> float:
        """Return the parameter scaling factor for a sequence length.

        Args:
            time_steps: Number of observed sequence positions.

        Returns:
            Multiplicative scale applied to model loss weights.
        """
        if time_steps == 12:
            return 1.0
        if time_steps == 30:
            return 100.0
        return 1.0

    def _decode_location_scale_parameters(
        self,
        reconstruction_outputs: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Convert raw reconstruction outputs to location and scale tensors."""
        batch_size = tf.shape(reconstruction_outputs)[0]
        distribution_params = tf.reshape(
            reconstruction_outputs,
            [batch_size, self.time_steps, 2],
        )
        location = float(self.reconstruction_max_reward) * tf.math.tanh(
            distribution_params[:, :, 0:1]
        )
        scale = tf.exp(distribution_params[:, :, 1:2]) + 1e-4
        return location, scale

    def reconstruction_location_scale_parameters(
        self,
        hidden_state: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Return decoded parameters for a shared recurrent state.

        This method is used by training diagnostics so the logged parameters
        are transformed by exactly the same logic as the forward pass.
        """
        if self.reconstruction_head_type != "nonlinear_location_scale":
            raise ValueError(
                "Location and scale are available only for the "
                "nonlinear_location_scale reconstruction head."
            )
        reconstruction_outputs = self.reconstruction_head(hidden_state)
        return self._decode_location_scale_parameters(
            reconstruction_outputs
        )

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = True,
        current_alpha: float | tf.Tensor = 1.0,
        current_kl_weight: float | tf.Tensor = 1.0,
        current_critic_coef: float | tf.Tensor = 1.0,
    ) -> tuple[tf.Tensor, ...]:
        """Run the model forward pass and compute training losses.

        Args:
            inputs: Reward tensor with shape ``[batch, time_steps, 1]``.
            training: Whether to add and return training losses.
            current_alpha: Current reconstruction schedule value.
            current_kl_weight: Current KL schedule value.
            current_critic_coef: Current critic-loss schedule value.

        Returns:
            Tuple containing reconstructed category probabilities, action
            probabilities, losses, information cost, latent means, and the
            final shared recurrent representation used by both output heads.
        """
        batch_size = tf.shape(inputs)[0]
        categories_onehot = helper.scalar_to_categorical(
            inputs,
            self.num_categories,
        )

        all_z_means = []
        all_z_log_vars = []
        all_category_outputs = []
        unavailable_reconstruction_parameter = tf.constant(
            float("nan"),
            dtype=tf.float32,
        )
        reconstruction_locations = unavailable_reconstruction_parameter
        reconstruction_scales = unavailable_reconstruction_parameter

        prior_categories = tf.zeros(
            [batch_size, self.time_steps, self.num_categories],
            dtype=tf.float32,
        )

        total_loss = 0
        first_decoder_loss = 0
        second_decoder_loss = 0
        third_decoder_loss = 0
        information_cost = 0
        cumulative_critic_loss = tf.constant(0.0, dtype=tf.float32)

        latent_state = tf.zeros(
            [batch_size, self.latent_dim],
            dtype=tf.float32,
        )
        recurrent_state = self.lstm_cell.get_initial_state(
            batch_size=batch_size,
        )

        for time_index in range(self.time_steps):
            category_at_time = categories_onehot[:, time_index, 0, :]
            one_hot_time = tf.one_hot(time_index, self.time_steps)
            one_hot_time = tf.tile(
                tf.reshape(one_hot_time, [1, -1]),
                [batch_size, 1],
            )

            if time_index == 0:
                hidden_state = tf.zeros((batch_size, self.rnn_units))

            hidden_state_flat = tf.reshape(hidden_state, [batch_size, -1])
            latent_state = tf.reshape(latent_state, [batch_size, -1])

            lstm_input = tf.concat([category_at_time, one_hot_time], axis=1)
            if time_index == 0:
                next_hidden_state = recurrent_state[0]
                next_cell_state = recurrent_state[1]
            else:
                next_hidden_state = hidden_state_flat
                next_cell_state = cell_state_flat

            _, recurrent_state = self.lstm_cell(
                lstm_input,
                states=(next_hidden_state, next_cell_state),
            )
            encoder_input = tf.concat(recurrent_state, axis=-1)
            z_mean, z_log_var, z_sampled = self.encoder(encoder_input)

            latent_state = z_sampled
            all_z_means.append(z_mean)
            all_z_log_vars.append(z_log_var)

            decoder_output = self.decoder(tf.concat([latent_state], axis=1))
            hidden_state_flat, cell_state_flat = tf.split(
                decoder_output,
                num_or_size_splits=2,
                axis=-1,
            )
            hidden_state = tf.reshape(
                hidden_state_flat,
                [batch_size, 1, self.rnn_units],
            )

            reconstruction_outputs = self.reconstruction_head(
                hidden_state_flat
            )
            if self.reconstruction_head_type == "nonlinear_categorical":
                category_logits = tf.reshape(
                    reconstruction_outputs,
                    [batch_size, self.time_steps, self.num_categories],
                )
                category_slice = tf.nn.softmax(
                    category_logits[:, : time_index + 1, :],
                    axis=-1,
                )
            else:
                (
                    reconstruction_locations,
                    reconstruction_scales,
                ) = self._decode_location_scale_parameters(
                    reconstruction_outputs
                )
                location = reconstruction_locations[
                    :, : time_index + 1, :
                ]
                scale = reconstruction_scales[:, : time_index + 1, :]
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
                cdf_at_edges = tf.math.sigmoid(
                    (bin_edges - location) / scale
                )
                category_slice = (
                    cdf_at_edges[:, :, 1:]
                    - cdf_at_edges[:, :, :-1]
                )
                category_slice = tf.reverse(category_slice, axis=[-1])
                category_slice = category_slice + 1e-6
                category_slice = category_slice / tf.reduce_sum(
                    category_slice,
                    axis=-1,
                    keepdims=True,
                )

            category_prior_slice = prior_categories[:, time_index + 1:, :]
            category_output = tf.concat(
                [category_slice, category_prior_slice],
                axis=1,
            )
            all_category_outputs.append(category_output)

            step_value_pred = self.critic_head(hidden_state_flat)
            if self.critic_loss_enabled:
                step_value_target = self.compute_best_achievable_value_target(
                    inputs,
                    time_index,
                )
                step_critic_loss = self.compute_critic_loss(
                    step_value_pred,
                    step_value_target,
                )
                cumulative_critic_loss += step_critic_loss

            prior_mean, prior_var = self.compute_time_conditional_prior(
                time_index,
                batch_size,
            )
            kl_loss = self.calculate_kl_loss(
                z_mean,
                z_log_var,
                prior_mean,
                prior_var,
            )
            information_cost += kl_loss

        final_action_decoder_input = tf.concat([hidden_state_flat], axis=1)
        action_logits = self.action_head(final_action_decoder_input)
        action_output = tf.nn.softmax(action_logits, axis=-1)

        action_loss = (
            self.compute_final_actor_loss(inputs, action_output)
            if self.action_loss_enabled
            else tf.constant(0.0, dtype=tf.float32)
        )
        critic_loss = cumulative_critic_loss / tf.cast(
            self.time_steps,
            tf.float32,
        )

        all_z_means = tf.stack(all_z_means, axis=1)
        all_z_log_vars = tf.stack(all_z_log_vars, axis=1)

        information_loss = information_cost / self.time_steps
        reconstruction_loss = self.compute_categorical_cross_entropy_loss(
            categories_onehot,
            all_category_outputs,
        )

        if training:
            self.add_loss(information_loss * current_kl_weight)
            if self.action_loss_enabled:
                self.add_loss(action_loss * self.action_loss_weight)
            self.add_loss(
                self.alpha * current_alpha * reconstruction_loss
            )

            total_loss += sum(self.losses)

            first_decoder_loss += (
                information_loss * current_kl_weight
                + reconstruction_loss * self.alpha * current_alpha
            )
            if self.action_loss_enabled:
                first_decoder_loss += (
                    action_loss * self.action_loss_weight
                )
            if self.critic_loss_enabled:
                first_decoder_loss += (
                    critic_loss
                    * self.action_loss_weight
                    * current_critic_coef
                )
            second_decoder_loss += reconstruction_loss
            if self.action_loss_enabled:
                third_decoder_loss += action_loss

        category_outputs = tf.stack(all_category_outputs, axis=1)

        return (
            category_outputs,
            action_output,
            total_loss,
            first_decoder_loss,
            second_decoder_loss,
            third_decoder_loss,
            critic_loss,
            information_loss,
            action_loss,
            reconstruction_loss,
            information_cost,
            all_z_means,
            hidden_state_flat,
            reconstruction_locations,
            reconstruction_scales,
        )

    def compute_time_conditional_prior(
        self,
        time_index: int,
        batch_size: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Compute the learned Gaussian prior for a specific time step.

        Args:
            time_index: Zero-based time-step index.
            batch_size: Dynamic batch size tensor.

        Returns:
            A tuple containing the prior mean and variance tensors.
        """
        prior_mean_at_time = self.prior_mu[time_index]
        prior_logvar_at_time = self.prior_logvar[time_index]

        prior_mean = tf.broadcast_to(
            prior_mean_at_time,
            [batch_size, self.latent_dim],
        )
        prior_var = tf.exp(
            tf.broadcast_to(
                prior_logvar_at_time,
                [batch_size, self.latent_dim],
            ),
        )

        return prior_mean, prior_var

    def calculate_kl_loss(
        self,
        z_means: tf.Tensor,
        z_log_vars: tf.Tensor,
        prior_mean: tf.Tensor,
        prior_var: tf.Tensor,
        epsilon: float = 1e-6,
    ) -> tf.Tensor:
        """Calculate KL divergence between posterior and time prior.

        Args:
            z_means: Posterior latent means.
            z_log_vars: Posterior latent log variances.
            prior_mean: Prior latent means.
            prior_var: Prior latent variances.
            epsilon: Numerical stability constant.

        Returns:
            Batch-averaged KL divergence.
        """
        prior_var = prior_var + epsilon
        prior_log_var = tf.math.log(prior_var)
        z_var = tf.exp(z_log_vars) + epsilon
        log_var_ratio = z_log_vars - prior_log_var

        kl_loss = -0.5 * tf.reduce_mean(
            1 + log_var_ratio
            - ((tf.square(z_means - prior_mean) + z_var) / prior_var),
            axis=1,
        )
        return tf.reduce_mean(kl_loss)

    def calculate_path_rewards(self, rewards: tf.Tensor) -> tf.Tensor:
        """Calculate summed rewards for each candidate path.

        Args:
            rewards: Reward tensor with shape ``[batch, time_steps, 1]``.

        Returns:
            Tensor of path rewards with shape ``[num_paths, batch, 1]``.
        """
        num_paths = len(self.index_path_map)
        batch_size = tf.shape(rewards)[0]
        path_rewards = tf.TensorArray(
            dtype=tf.float32,
            size=num_paths,
            dynamic_size=False,
        )

        path_items = enumerate(self.index_path_map.values())
        for path_index, node_indices in path_items:
            try:
                node_indices_tensor = (
                    tf.convert_to_tensor(node_indices, dtype=tf.int32) - 1
                )
                node_indices_tensor = tf.clip_by_value(
                    node_indices_tensor,
                    0,
                    tf.shape(rewards)[1] - 1,
                )
                node_indices_tensor = tf.tile(
                    tf.expand_dims(node_indices_tensor, 0),
                    [batch_size, 1],
                )
                gathered_rewards = tf.gather(
                    rewards,
                    node_indices_tensor,
                    axis=1,
                    batch_dims=1,
                )
                path_reward = tf.reduce_sum(gathered_rewards, axis=1)
                path_reward = tf.where(
                    tf.math.is_finite(path_reward),
                    path_reward,
                    tf.zeros_like(path_reward),
                )
                path_rewards = path_rewards.write(path_index, path_reward)
            except Exception:
                path_rewards = path_rewards.write(
                    path_index,
                    tf.zeros([batch_size], dtype=tf.float32),
                )

        path_reward_tensor = path_rewards.stack()
        return tf.where(
            tf.math.is_finite(path_reward_tensor),
            path_reward_tensor,
            tf.zeros_like(path_reward_tensor),
        )

    def compute_categorical_cross_entropy_loss(
        self,
        target_categories: tf.Tensor,
        category_outputs: list[tf.Tensor],
    ) -> tf.Tensor:
        """Compute masked categorical reconstruction loss over time.

        Args:
            target_categories: One-hot target categories for each time step.
            category_outputs: Predicted category probabilities from each step.

        Returns:
            Normalized masked categorical cross-entropy loss.
        """
        batch_size = tf.shape(target_categories)[0]
        target_category_onehot = tf.squeeze(target_categories, axis=2)
        stacked_predictions = tf.stack(category_outputs, axis=0)
        if self.reconstruction_loss_scope == "all_timesteps":
            time_mask = tf.linalg.band_part(
                tf.ones([self.time_steps, self.time_steps]),
                -1,
                0,
            )
        else:
            time_mask = tf.concat(
                [
                    tf.zeros(
                        [self.time_steps - 1, self.time_steps],
                        dtype=tf.float32,
                    ),
                    tf.ones([1, self.time_steps], dtype=tf.float32),
                ],
                axis=0,
            )

        target_expanded = tf.tile(
            target_category_onehot[None, :, :, :],
            [self.time_steps, 1, 1, 1],
        )

        # Adding epsilon rather than clipping preserves corrective gradients
        # for very small predicted probabilities.
        safe_probabilities = stacked_predictions + 1e-7
        raw_cross_entropy = -tf.reduce_sum(
            target_expanded * tf.math.log(safe_probabilities),
            axis=-1,
        )
        masked_cross_entropy = raw_cross_entropy * time_mask[:, None, :]

        total_loss = tf.reduce_sum(masked_cross_entropy) / tf.math.log(
            tf.cast(self.num_categories, tf.float32),
        )
        valid_count = tf.reduce_sum(time_mask) * tf.cast(
            batch_size,
            tf.float32,
        )
        return total_loss / valid_count

    def _prepare_path_rewards(
        self,
        inputs: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Prepare finite path-reward tensors for actor and critic losses.

        Args:
            inputs: Reward tensor with shape ``[batch, time_steps, 1]``.

        Returns:
            Tuple containing raw path rewards and normalized path rewards.
        """
        finite_inputs = tf.convert_to_tensor(inputs)
        finite_inputs = tf.where(
            tf.math.is_finite(finite_inputs),
            finite_inputs,
            tf.zeros_like(finite_inputs),
        )

        batch_size = tf.shape(finite_inputs)[0]
        actual_path_rewards = self.calculate_path_rewards(finite_inputs)
        actual_path_rewards = tf.where(
            tf.math.is_finite(actual_path_rewards),
            actual_path_rewards,
            tf.zeros_like(actual_path_rewards),
        )
        actual_path_rewards = tf.transpose(actual_path_rewards, perm=[1, 0, 2])
        actual_path_rewards = tf.reshape(
            actual_path_rewards,
            [batch_size, self.num_paths],
        )

        normalized_path_rewards = actual_path_rewards
        return actual_path_rewards, normalized_path_rewards

    def compute_final_actor_loss(
        self,
        inputs: tf.Tensor,
        action_probs: tf.Tensor,
    ) -> tf.Tensor:
        """Compute actor loss from expected path reward and entropy.

        Args:
            inputs: Reward tensor with shape ``[batch, time_steps, 1]``.
            action_probs: Path-selection probabilities.

        Returns:
            Scalar actor loss tensor.
        """
        _, normalized_path_rewards = self._prepare_path_rewards(inputs)
        gradient_scaled_action_probs = (
            self.scale_confident_action_gradients(action_probs)
        )
        expected_reward = tf.reduce_sum(
            gradient_scaled_action_probs * normalized_path_rewards,
            axis=1,
            keepdims=True,
        )
        mean_expected_reward = tf.reduce_mean(expected_reward)

        entropy = -tf.reduce_sum(
            action_probs * tf.math.log(action_probs + 1e-8),
            axis=1,
            keepdims=True,
        )
        mean_entropy = tf.reduce_mean(entropy)

        reward_normalizer = self.reward_normalization_constant
        if reward_normalizer is None:
            if self.time_steps == 30:
                configured_tree_types = {
                    "deep_breadth",
                    "deep_depth",
                    "wide_breadth",
                    "wide_depth",
                }
                if self.tree_type not in configured_tree_types:
                    raise ValueError(
                        "A 31-node actor needs a configured deep or wide "
                        "tree type."
                    )
                reward_normalizer = 8.157443089523166
            else:
                reward_normalizer = {
                    2: 0.75,
                    6: 3.58,
                    12: 5.11,
                    39: 8.0682,
                }.get(self.time_steps, 8.157443089523166)
        mean_expected_reward = mean_expected_reward / reward_normalizer

        action_loss = (1.0 - mean_expected_reward) - (
            self.action_entropy_bonus * mean_entropy
        )
        return tf.where(
            tf.math.is_finite(action_loss),
            action_loss,
            tf.constant(0.0, dtype=tf.float32),
        )

    def action_gradient_scale(self, action_probs: tf.Tensor) -> tf.Tensor:
        """Return a per-trial scale for saturated path-policy gradients.

        The multiplier stays at one through the configured confidence
        threshold, then grows inversely with the remaining probability mass.
        It is capped to avoid unstable updates.
        """
        confidence = tf.reduce_max(action_probs, axis=-1, keepdims=True)
        remaining_probability = tf.maximum(
            1.0 - confidence,
            tf.constant(1e-6, dtype=action_probs.dtype),
        )
        threshold_remaining_probability = tf.cast(
            1.0 - self.action_gradient_confidence_threshold,
            action_probs.dtype,
        )
        scale = threshold_remaining_probability / remaining_probability
        return tf.clip_by_value(
            scale,
            clip_value_min=tf.cast(1.0, action_probs.dtype),
            clip_value_max=tf.cast(
                self.max_action_gradient_scale,
                action_probs.dtype,
            ),
        )

    def scale_confident_action_gradients(
        self,
        action_probs: tf.Tensor,
    ) -> tf.Tensor:
        """Keep path probabilities unchanged while scaling their gradients."""
        stopped_probabilities = tf.stop_gradient(action_probs)
        scale = tf.stop_gradient(self.action_gradient_scale(action_probs))
        return stopped_probabilities + scale * (
            action_probs - stopped_probabilities
        )

    def compute_best_achievable_value_target(
        self,
        inputs: tf.Tensor,
        time_index: int,
    ) -> tf.Tensor:
        """Compute the critic target from the best partial path reward.

        Args:
            inputs: Reward tensor with shape ``[batch, time_steps, 1]``.
            time_index: Zero-based current time-step index.

        Returns:
            Stop-gradient critic target tensor with shape ``[batch, 1]``.
        """
        batch_size = tf.shape(inputs)[0]
        time_mask = tf.concat(
            [
                tf.ones([batch_size, time_index + 1, 1], dtype=tf.float32),
                tf.zeros(
                    [
                        batch_size,
                        self.time_steps - (time_index + 1),
                        1,
                    ],
                    dtype=tf.float32,
                ),
            ],
            axis=1,
        )

        masked_inputs = inputs * time_mask
        actual_path_rewards, _ = self._prepare_path_rewards(masked_inputs)
        best_partial_reward = tf.reduce_max(
            actual_path_rewards,
            axis=1,
            keepdims=True,
        )

        value_target = tf.stop_gradient(best_partial_reward)
        return value_target

    def compute_critic_loss(
        self,
        value_pred: tf.Tensor,
        value_target: tf.Tensor,
    ) -> tf.Tensor:
        """Compute mean-squared-error critic loss.

        Args:
            value_pred: Predicted critic values.
            value_target: Stop-gradient critic targets.

        Returns:
            Scalar critic loss tensor.
        """
        critic_loss = tf.reduce_mean(tf.square(value_pred - value_target))
        return tf.where(
            tf.math.is_finite(critic_loss),
            critic_loss,
            tf.constant(0.0, dtype=tf.float32),
        )
