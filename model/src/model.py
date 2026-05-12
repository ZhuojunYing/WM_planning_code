"""Neural network components for the variational recurrent planning model."""

from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow.keras import layers, models

import helper


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
            reward_normalization_constant: Optional reward normalizer retained
                for compatibility with the command-line training entry point.
        """
        super().__init__()

        parameter_scale = self._parameter_scale_for_time_steps(time_steps)

        self.encoder = encoder
        self.decoder = decoder
        self.rnn_units = rnn_units
        self.time_steps = time_steps
        self.num_paths = num_paths
        self.latent_dim = latent_dim
        self.alpha = alpha * parameter_scale
        self.lambda_ = lambda_ * parameter_scale
        self.action_loss_weight = 1.0 * parameter_scale
        self.parameter_scale = parameter_scale
        self.num_categories = 9
        self.tree_type = tree_type
        self.reward_normalization_constant = reward_normalization_constant
        self.index_path_map = index_path_map
        self.path_map = path_map
        self.path_cov_mat = path_cov_mat

        self.reconstruction_head = tf.keras.layers.Dense(
            time_steps * 2,
            activation="linear",
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0,
                stddev=0.01,
            ),
            bias_initializer="zeros",
        )
        self.action_head = tf.keras.layers.Dense(
            num_paths,
            activation=None,
            kernel_initializer="glorot_uniform",
        )
        self.lstm_cell = tf.keras.layers.LSTMCell(
            self.rnn_units,
            kernel_initializer="orthogonal",
            recurrent_initializer="orthogonal",
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

    @staticmethod
    def _parameter_scale_for_time_steps(time_steps: int) -> float:
        """Return the parameter scaling factor for a sequence length.

        Args:
            time_steps: Number of observed sequence positions.

        Returns:
            Multiplicative scale applied to model loss weights.
        """
        if time_steps == 12:
            return 10.0
        if time_steps == 30:
            return 100.0
        return 1.0

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
            probabilities, losses, information cost, and latent means.
        """
        batch_size = tf.shape(inputs)[0]
        categories_onehot = helper.scalar_to_categorical(
            inputs,
            self.num_categories,
        )

        all_z_means = []
        all_z_log_vars = []
        all_category_outputs = []

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
        cumulative_action_loss = tf.constant(0.0, dtype=tf.float32)

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

            step_action_logits = self.action_head(hidden_state_flat)
            step_action_probs = tf.nn.softmax(step_action_logits, axis=-1)

            time_mask = tf.concat(
                [
                    tf.ones(
                        [batch_size, time_index + 1, 1],
                        dtype=tf.float32,
                    ),
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
            step_action_loss = self.compute_final_actor_loss(
                masked_inputs,
                step_action_probs,
            )
            cumulative_action_loss += step_action_loss

            distribution_params = self.reconstruction_head(hidden_state_flat)
            distribution_params = tf.reshape(
                distribution_params,
                [batch_size, self.time_steps, 2],
            )
            location_raw = distribution_params[:, : time_index + 1, 0:1]
            location = 5 * tf.math.tanh(location_raw)
            scale = tf.exp(
                distribution_params[:, : time_index + 1, 1:2],
            ) + 1e-4

            bin_edges = tf.constant(
                [-4.5, -3.5, -2.5, -1.5, -0.5,
                 0.5, 1.5, 2.5, 3.5, 4.5],
                dtype=tf.float32,
            )
            bin_edges = tf.reshape(bin_edges, [1, 1, 10])

            cdf_at_edges = tf.math.sigmoid((bin_edges - location) / scale)
            category_slice_raw = (
                cdf_at_edges[:, :, 1:10] - cdf_at_edges[:, :, 0:9]
            )
            category_slice_raw = tf.reverse(category_slice_raw, axis=[-1])
            category_slice_raw = category_slice_raw + 1e-6
            category_slice = category_slice_raw / tf.reduce_sum(
                category_slice_raw,
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

        action_loss = self.compute_final_actor_loss(inputs, action_output)
        auxiliary_action_loss = cumulative_action_loss / tf.cast(
            self.time_steps,
            tf.float32,
        )
        critic_loss = cumulative_critic_loss / tf.cast(
            self.time_steps,
            tf.float32,
        )

        value_pred = self.critic_head(final_action_decoder_input)
        value_target = self.compute_best_achievable_value_target(
            inputs,
            self.time_steps - 1,
        )

        all_z_means = tf.stack(all_z_means, axis=1)
        all_z_log_vars = tf.stack(all_z_log_vars, axis=1)

        information_loss = information_cost / self.time_steps
        reconstruction_loss = self.compute_categorical_cross_entropy_loss(
            categories_onehot,
            all_category_outputs,
        )

        if training:
            self.add_loss(information_loss * self.lambda_)
            self.add_loss(action_loss * self.action_loss_weight)
            self.add_loss(self.alpha * reconstruction_loss)

            total_loss += sum(self.losses)

            first_decoder_loss += (
                information_loss * current_kl_weight
                + action_loss * self.action_loss_weight
                + critic_loss * self.action_loss_weight * current_critic_coef
                + reconstruction_loss * self.alpha
            )
            second_decoder_loss += reconstruction_loss
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
        time_mask = tf.linalg.band_part(
            tf.ones([self.time_steps, self.time_steps]),
            -1,
            0,
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
        expected_reward = tf.reduce_sum(
            action_probs * normalized_path_rewards,
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
        entropy_weight = 0.05 if self.time_steps == 12 else 0

        reward_normalizer = self.reward_normalization_constant
        if reward_normalizer is None:
            reward_normalizer = {
                2: 0.75,
                6: 3.58,
                12: 5.11,
                30: 8.1574,
                39: 8.0682,
            }.get(self.time_steps, 8.1574)
        mean_expected_reward = mean_expected_reward / reward_normalizer

        action_loss = (1.0 - mean_expected_reward) - (
            entropy_weight * mean_entropy
        )
        return tf.where(
            tf.math.is_finite(action_loss),
            action_loss,
            tf.constant(0.0, dtype=tf.float32),
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
