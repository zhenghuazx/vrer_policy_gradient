import random
from collections import deque

import numpy as np
import tensorflow as tf
from gym.spaces.discrete import Discrete
from vrer_policy_gradient import A2C
from tensorflow_probability.python.distributions import (
    Categorical, MultivariateNormalDiag)


class A2CVRER(A2C):
    """
    Asynchronous Methods for Deep Reinforcement Learning
    https://arxiv.org/abs/1602.01783
    """

    def __init__(
        self,
        envs,
        model,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        grad_norm=0.5,
        num_reuse_each_iter=3,
        buffer_size=100,
        **kwargs,
    ):
        """
        Initialize A2C agent.
        Args:
            envs: A list of gym environments.
            model: tf.keras.models.Model that is expected to be compiled
                with an optimizer before training starts.
            entropy_coef: Entropy coefficient used for entropy loss calculation.
            value_loss_coef: Value coefficient used for value loss calculation.
            grad_norm: Gradient clipping value passed to tf.clip_by_global_norm()
            **kwargs: kwargs Passed to super classes.
        """
        super(A2CVRER, self).__init__(envs, model, **kwargs)
        self.num_reuse_each_iter = num_reuse_each_iter
        self.reuse_set_size = tf.Variable(0)
        self.m_squared_sum, self.v_sum, self.relative_var = tf.Variable(0.0), tf.Variable(0.0), tf.Variable(0.0)
        self.model_history = deque(maxlen=buffer_size)
        self.buffers = deque(maxlen=buffer_size)
        self.old_actor = tf.keras.models.clone_model(self.model)
        
    def compute_relative_variance(self):
        m_squared_sum, v_sum = 0, 0
        # Loop through all trainable variables in the model
        iterations = int(self.model.optimizer.iterations.numpy())
        config = self.model.optimizer.get_config()
        beta_1_power = tf.pow(config['beta_1'], iterations + 1)
        beta_2_power = tf.pow(config['beta_2'], iterations + 1)
        for var in self.model.trainable_variables:
            # Get the first and second moments
            m = self.model.optimizer.get_slot(var, "m") / (1 - beta_1_power)
            v = self.model.optimizer.get_slot(var, "v") / (1 - beta_1_power)

            m_squared_sum += tf.reduce_sum(tf.square(m))
            v_sum += tf.reduce_sum(v)
        return m_squared_sum, v_sum, (v_sum - m_squared_sum) / m_squared_sum

    def calculate_kl_div(self, states, actions, old_weights, new_distribution):
        """
        Calculate probability distribution of both new and old actor models
        and calculate Kullback–Leibler divergence.
        Args:
            states: States tensor expected by the actor models.
            old_weights: Trainable weights of an old policy

        Returns:
            Mean KL divergence, old distribution and new distribution.
        """
        self.old_actor.set_weights(old_weights)
        old_actor_output = self.get_model_outputs(
            states, [self.old_actor], actions,
        )[4]
        old_distribution = self.get_distribution(old_actor_output)
        return (
            tf.reduce_mean(old_distribution.kl_divergence(new_distribution)),
            old_distribution,
        )

    def get_likelihood_ratio(self, states, actions, old_weights, new_distribution, new_log_probs):
        # calculate likelihood ratio and KL divergence
        (
            kl_divergence,
            old_distribution,
        ) = self.calculate_kl_div(states, actions, old_weights, new_distribution)

        ratios = tf.exp(
            new_log_probs - old_distribution.log_prob(actions)
        )
        ratio = tf.reduce_mean(ratios)
        return ratio, kl_divergence

    def np_train_step(self):
        """
        Perform the batching and return calculation in numpy.
        """
        (
            states,
            rewards,
            actions,
            values,
            dones,
            log_probs,
            entropies,
            actor_output,
        ) = [np.asarray(item, np.float32) for item in self.get_batch()]
        returns = self.calculate_returns(rewards, dones)
        batch = [states, actions, returns, values]
        states_flatten, actions_flatten = self.concat_step_batches(states, actions)
        
        new_actor_output = self.get_model_outputs(states_flatten, self.output_models)[4]
        new_distribution = self.get_distribution(new_actor_output)
        new_log_probs = new_distribution.log_prob(actions_flatten)

        all_states = states.copy()
        all_actions = actions.copy()
        all_returns = returns.copy()
        all_values = values.copy()
        num_reuse = 0
        sel_constant = self.c * int(self.model.optimizer.iterations.numpy()) / (
                    1 + int(self.model.optimizer.iterations.numpy()))
        for i, buffer_batch in enumerate(self.buffers):
            old_states, old_actions, old_returns, old_values = buffer_batch
            ratio, kl_divergence = self.get_likelihood_ratio(states_flatten, actions_flatten, self.model_history[i],
                                                             new_distribution, new_log_probs)
            var_ratio = ratio  # * np.exp(abs(ratio - 1) * v_sum**2 / m_squared_sum**2)
            indices = random.choices(range(self.n_steps), k=self.num_reuse_each_iter)
            if var_ratio < 1 + (sel_constant - 1) * self.relative_var.numpy() / (self.relative_var.numpy() + 1):
                all_states = np.concatenate([all_states, old_states[indices]])
                all_actions = np.concatenate([all_actions, old_actions[indices]])
                all_returns = np.concatenate([all_returns, old_returns[indices]])
                all_values = np.concatenate([all_values, old_values[indices]])
                num_reuse += 1
        print('number of reuse: ', num_reuse)
        self.buffers.append(batch)
        self.reuse_set_size.assign(num_reuse)
        self.model_history.append(self.model.get_weights())

        return self.concat_step_batches(all_states, all_returns, all_actions, all_values)

    @tf.function
    def train_step(self):
        """
        Perform 1 step which controls action_selection, interaction with environments
        in self.envs, batching and gradient updates.

        Returns:
            None
        """
        states, returns, actions, old_values = tf.numpy_function(
            self.np_train_step, [], 4 * [tf.float32]
        )
        advantages = returns - old_values
        with tf.GradientTape() as tape:
            _, log_probs, critic_output, entropy, actor_output = self.get_model_outputs(
                states, self.output_models, actions=actions
            )
            entropy = tf.reduce_mean(entropy)
            pg_loss = -tf.reduce_mean(advantages * log_probs)
            value_loss = tf.reduce_mean(tf.square(critic_output - returns))
            loss = (
                pg_loss
                - entropy * self.entropy_coef
                + value_loss * self.value_loss_coef
            )
        grads = tape.gradient(loss, self.model.trainable_variables)
        if self.grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.grad_norm)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
