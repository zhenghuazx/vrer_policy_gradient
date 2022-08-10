
import pybullet_envs
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import MultivariateNormalDiag
from tensorflow.keras.optimizers import Adam, SGD
import numpy as np
import time
import matplotlib.pyplot as plt
import gym
import os


class PolicyGradientNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=32, fc2_dims=32):
        super(PolicyGradientNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.pi = Dense(n_actions, activation='linear')

    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)
        pi = self.pi(value)
        return pi



class Agent:
    def __init__(self, alpha=0.003, gamma=0.99, n_actions=4, n_k=4, num_episodes=2000, layer1_size=32, layer2_size=32,
                 c=1.5):
        self.c = c
        self.gamma = gamma
        self.lr = alpha
        self.n_actions = n_actions
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.state_memory_full = []
        self.action_memory_full = []
        self.reward_memory_full = []
        self.G_memory_full = []
        self.state_memory = []
        self.model_memory = []
        self.reuses = []
        self.variance = []
        self.time_elapsed = []
        self.gradient_norm = []
        self.num_episodes = num_episodes
        self.loglik = np.zeros((num_episodes, n_k))
        self.policy = PolicyGradientNetwork(n_actions=n_actions)
        # self.policy.compile(optimizer=SGD(learning_rate=self.lr, decay=0.0))
        self.policy.compile(optimizer=Adam(learning_rate=self.lr))
        self._policy_hist = PolicyGradientNetwork(n_actions=n_actions)
        # self._policy_hist.compile(optimizer=SGD(learning_rate=self.lr))

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        probs = self.policy(state)
        action_probs = MultivariateNormalDiag(probs)
        action = action_probs.sample()
        # action = tf.squeeze(action)
        # print(action.numpy())
        action = np.nan_to_num(action)
        return action

    def store_transition(self, observation, action, reward):
        # (iter, r, H)
        self.state_memory = observation
        self.action_memory = action
        self.reward_memory = reward

    def compute_ilr(self):
        return

    def gradient_compute(self, model, i, j, p, n_k, H):
        with tf.GradientTape(persistent=True) as tape:
            cur_likelihood = 0
            loss = 0
            score = 0
            for idx, (g, state) in enumerate(zip(self.G_memory_full[i][j][:], self.state_memory_full[i][j])):
                state = tf.convert_to_tensor([state], dtype=tf.float32)
                if model == None:
                    model = self._policy_hist.set_weights(self.model_memory[p])
                probs = model(state)
                action_probs = MultivariateNormalDiag(probs)
                log_prob = action_probs.log_prob(self.action_memory_full[i][j][idx])
                # loss[j, idx] = -g * tf.squeeze(log_prob)
                loss += -g * tf.squeeze(log_prob)  # negative loss
                score += tf.squeeze(log_prob)  # postive log probability
                cur_likelihood += tf.squeeze(log_prob)
        grad = tape.gradient(loss, model.trainable_variables)
        score = tape.gradient(score, model.trainable_variables)
        del tape
        return grad, score, cur_likelihood

    def compute_likelihood(self, model, i, j, p, n_k, H):
        cur_likelihood = 0
        for idx, (g, state) in enumerate(zip(self.G_memory_full[i][j][:], self.state_memory_full[i][j])):
            state = tf.convert_to_tensor([state], dtype=tf.float32)
            if model == None:
                model = self._policy_hist.set_weights(self.model_memory[p])
            probs = model(state)
            action_probs = MultivariateNormalDiag(probs)
            log_prob = action_probs.log_prob(self.action_memory_full[i][j][idx])
            log_prob = tf.where(tf.math.is_nan(log_prob), tf.zeros_like(log_prob), log_prob)
            cur_likelihood += tf.squeeze(log_prob)
        return cur_likelihood

    def mixture_gradient_compute(self, reuse, n_k, num_iters, loglikelihoods):
        with tf.GradientTape(persistent=True) as tape:
            loss = 0
            for i in reuse:
                for j in range(n_k):
                    # numerator = np.exp(loglikelihoods[i, j])
                    # reuse_mixture = [k for k in reuse if k >= i]
                    # denominator = np.sum(np.exp(self.loglikelihoods[i, j, [k for k in reuse if k >= i]])) / len(reuse_mixture)
                    # denominator = np.exp(self.loglik[i, j])
                    ratio = np.exp(loglikelihoods[i, j] - self.loglik[i, j])
                    for idx, (g, state) in enumerate(zip(self.G_memory_full[i][j][:], self.state_memory_full[i][j])):
                        state = tf.convert_to_tensor([state], dtype=tf.float32)
                        probs = self.policy(state)
                        action_probs = MultivariateNormalDiag(probs)
                        log_prob = action_probs.log_prob(self.action_memory_full[i][j][idx])
                        log_prob = tf.where(tf.math.is_nan(log_prob), tf.zeros_like(log_prob), log_prob)
                        loss += - np.clip(ratio, 0.5, 2) * g * log_prob
            loss = loss / (len(reuse) * n_k)
        grad = tape.gradient(loss, self.policy.trainable_variables)
        del tape
        return grad

    def learn(self):
        # actions = tf.convert_to_tensor(self.action_memory, dtype=tf.float32)
        # rewards = np.array(self.reward_memory)
        n_k = len(self.reward_memory)
        loglikelihoods = np.zeros((self.num_episodes, n_k))
        G = {}
        for j in range(n_k):
            rewards = self.reward_memory[j]
            H = len(self.reward_memory[j])
            G_j = np.zeros_like(rewards)
            for t in range(H):
                G_sum = 0
                discount = 1
                for k in range(t, H):
                    G_sum += rewards[k] * discount
                    discount *= self.gamma
                G_j[t] = G_sum
            G[j] = G_j
        # store cur info to full
        self.G_memory_full.append(G)
        self.state_memory_full.append(self.state_memory)
        self.action_memory_full.append(self.action_memory)
        self.reward_memory_full.append(self.reward_memory)
        num_iters = len(self.reward_memory_full)
        # loss = np.zeros((n_k, H), dtype = 'float32') # tf.zeros((n_k, H))
        # cur_likelihood = np.zeros((n_k, H))
        grad_agg = []
        timer1 = time.time()
        scores = []
        for j in range(n_k):
            grad, score, ll = self.gradient_compute(self.policy, -1, j, -1, n_k, H)
            grad_numpy = [g.numpy().flatten() for g in grad]
            grad_numpy = np.concatenate(grad_numpy)
            grad_agg.append(grad_numpy)
            score_numpy = [s.numpy().flatten() for s in score]
            score_numpy = np.concatenate(score_numpy)
            scores.append(score_numpy)
            loglikelihoods[num_iters - 1, j] = ll
            self.loglik[num_iters - 1, j] = ll
        cur_pg_variance = np.stack(grad_agg, axis=0)
        # print(cur_pg_variance)
        # compute the current policy gradient total variance
        cur_pg_var = np.mean(np.linalg.norm(cur_pg_variance - cur_pg_variance.mean(axis=0), ord=2, axis=1) ** 2)
        self.variance.append(cur_pg_var)
        timer2 = time.time()
        scores = np.array(scores)
        fisher = []
        for j in range(n_k):
            fisher.append(np.outer(scores[j,], scores[j,]))

        # print(scores.shape)

        reuse_iter = [num_iters - 1]
        variance_ratios = []
        for i in range(num_iters - 1):
            theta_diff = np.subtract(np.array(self.model_memory[-1]), np.array(self.model_memory[i]))
            theta_diff_numpy = [s.flatten() for s in theta_diff]
            theta_diff = np.concatenate(theta_diff_numpy)
            variance_ratio = 0
            for j in range(n_k):
                a = np.inner(theta_diff, scores[j,])
                variance_ratio += np.exp(a + a ** 2) * np.linalg.norm(cur_pg_variance[j,]) ** 2

            variance_ratio = (variance_ratio / n_k - np.linalg.norm(cur_pg_variance.mean(axis=0)) ** 2) / cur_pg_var
            variance_ratios.append(variance_ratio)
            # print(variance_ratio)
            if variance_ratio <= self.c:
                reuse_iter.append(i)
        # ind = np.argsort(variance_ratios, axis=None)
        # print('reuse: ', reuse_iter, ind[:int(0.75 * len(variance_ratios))])
        # if num_iters > 10:
        #   reuse_iter = list(ind[:int(0.75 * len(variance_ratios))])
        #   if num_iters - 1 not in reuse_iter:
        #     reuse_iter = reuse_iter + [num_iters-1]
        #   reuse_iter.sort()

        # # compute the nested likelihood ratio
        # timer2 = time.time()
        # gradient = np.zeros((num_iters, n_k, policy_param_size))
        # for j in range(n_k):
        #     gradient[num_iters - 1, j, :] = grad_agg[j]
        # # i-th iter
        for i in reuse_iter[:-1]:
            # j-th replicate data
            for j in range(n_k):
                ll = self.compute_likelihood(self.policy, i, j, -1, n_k, H)
                loglikelihoods[i, j] = ll
        # loss_ilr_i_j = np.zeros((n_k, policy_param_size))
        # reuse_iter = []
        # for i in range(num_iters):
        #     for j in range(n_k):
        #         numerator = np.exp(self.loglikelihoods[i,j,num_iters-1])
        #         denominator = np.exp(self.loglikelihoods[i,j,i])
        #         loss_ilr_i_j[j, :] = numerator / denominator * gradient[i, j, :]
        #     cur_ilr_variance = np.mean(np.linalg.norm(loss_ilr_i_j, ord=2, axis=1)) # ith
        #     if cur_ilr_variance <= self.c * cur_pg_variance:
        #         reuse_iter.append(i)
        timer3 = time.time()
        self.reuses.append(reuse_iter)
        gradient = self.mixture_gradient_compute(reuse_iter, n_k, num_iters, loglikelihoods)
        ##
        grad_agg = []
        grad_numpy = [g.numpy().flatten() for g in gradient]
        grad_numpy = np.concatenate(grad_numpy)
        grad_agg.append(grad_numpy)
        cur_pg_variance = np.stack(grad_agg, axis=0)
        self.gradient_norm.append(
            np.mean(np.linalg.norm(cur_pg_variance - cur_pg_variance.mean(axis=0), ord=2, axis=1) ** 2))
        self.policy.optimizer.apply_gradients(zip(gradient, self.policy.trainable_variables))
        timer4 = time.time()
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.time_elapsed.append([timer2 - timer1, timer3 - timer2, timer4 - timer3])


if __name__ == '__main__':
    c = 1.5
    index = 1
    seed = 2021 + index
    n_k = 4
    lr = 0.0015
    num_episodes = 500  # iteraction
    problem = "InvertedPendulumBulletEnv-v0"  # "LunarLander-v2"
    env = gym.make(problem)
    # env._max_episode_steps = 200
    num_states = env.observation_space.shape[0]
    print("Size of State Space ->  {}".format(num_states))
    num_actions = env.action_space.shape[0]
    print("Size of Action Space ->  {}".format(num_actions))
    path = "VPGVRER/{}/approx-2nd-seed-lr-{}-{}-n_k-{}-id-{}-c-{}".format(problem, lr, seed, n_k, index, c)
    # Log details
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)

    env.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    agent = Agent(alpha=lr, gamma=0.99, n_actions=num_actions, n_k=4, num_episodes=500)
    score_history = []
    for i in range(num_episodes):
        score = 0
        old_weights = agent.policy.get_weights()
        model = [old_weights]
        observations = {}
        actions = {}
        rewards = {}
        for j in range(n_k):
            observations[j] = []
            actions[j] = []
            rewards[j] = []
            done = False
            observation = env.reset()
            while not done:
                observation = tf.squeeze(observation)
                action = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                observations[j].append(observation)
                actions[j].append(action)
                rewards[j].append(reward)
                observation = observation_
                score += reward
            # print(rewards)
        agent.store_transition(observations, actions, rewards)
        score_history.append(score / n_k)
        agent.model_memory.append(agent.policy.get_weights())
        agent.learn()
        agent.policy.save_weights(path + "/model-{}".format(i))
        # Update running reward to check condition for solving
        avg_score = np.mean(score_history[-100:])
        print('episode: ', i, 'score: %.1f' % (score / n_k),
              'average score %.1f' % avg_score)
        # template = "reuse window: {}"
        # print(template.format(agent.reuses[-1]))
        if avg_score >= 995:  # Condition to consider the task solved
            print("Solved at episode {}!".format(i))

    with open(path + '/reuses.txt', 'w') as f:
        for _list in agent.reuses:
            for i in range(len(_list)):
                # f.seek(0)
                if i == len(_list) - 1:
                    f.write(str(_list[i]) + '\n')
                else:
                    f.write(str(_list[i]) + ',')
    np.save(path + '/variance', agent.variance)
    np.save(path + '/time_elapsed', agent.time_elapsed)
    np.save(path + "/score_history", score_history)
    np.save(path + "/gradient_norm", agent.gradient_norm)