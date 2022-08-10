import gym
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam, SGD

class PolicyGradientNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=32, fc2_dims=32):
        super(PolicyGradientNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.pi = Dense(n_actions, activation='softmax')

    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)

        pi = self.pi(value)

        return pi


class Agent:
    def __init__(self, alpha=0.003, gamma=0.99, n_actions=4,
                 layer1_size=32, layer2_size=32):

        self.gamma = gamma
        self.lr = alpha
        self.n_actions = n_actions
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.state_memory_full = []
        self.action_memory_full = []
        self.reward_memory_full = []
        self.policy = PolicyGradientNetwork(n_actions=n_actions)
        self.policy.compile(optimizer=Adam(learning_rate=self.lr))
        # self.policy.compile(optimizer=SGD(learning_rate=self.lr, decay=0.0))

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        probs = self.policy(state)
        action_probs = tfp.distributions.Categorical(probs=probs)
        action = action_probs.sample()

        return action.numpy()[0]

    def store_transition(self, observation, action, reward):
        # (iter, r, H)
        self.state_memory = observation
        self.action_memory = action
        self.reward_memory = reward
    def compute_ilr(self):
        return


    def learn(self):
        # actions = tf.convert_to_tensor(self.action_memory, dtype=tf.float32)
        n_k = len(self.reward_memory)
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

        with tf.GradientTape() as tape:
            loss = 0
            for j in range(n_k):
                for idx, (g, state) in enumerate(zip(G[j][:], self.state_memory[j])):
                    state = tf.convert_to_tensor([state], dtype=tf.float32)
                    probs = self.policy(state)
                    action_probs = tfp.distributions.Categorical(probs=probs)
                    log_prob = action_probs.log_prob(actions[j][idx])
                    loss += -g * tf.squeeze(log_prob)
            loss = loss / n_k
        gradient = tape.gradient(loss, self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(zip(gradient, self.policy.trainable_variables))

        self.state_memory_full.append(self.state_memory)
        self.action_memory_full.append(self.action_memory)
        self.reward_memory_full.append(self.reward_memory)

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

if __name__ == '__main__':
    index = 1
    seed = 2021 + index
    n_k = 4
    problem = "CartPole-v0"
    env = gym.make(problem)
    num_states = env.observation_space.shape[0]
    print("Size of State Space ->  {}".format(num_states))
    num_actions = env.action_space.shape
    print("Size of Action Space ->  {}".format(num_actions))

    score_history = []
    macro = 10
    n_k = 4
    num_episodes = 500
    env.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    for m in range(macro):
        agent = Agent(alpha=0.005, gamma=0.99, n_actions=2)
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

            agent.learn()
            avg_score = np.mean(score_history[-100:])
            print('episode: ', i,'score: %.1f' % (score / n_k),
                'average score %.1f' % avg_score)
            if avg_score >= 195:  # Condition to consider the task solved
                print("Solved at episode {}!".format(i))
#                 break
        np.save('pg_out-5e_3-seed-{0}-m-{1}'.format(seed,m), np.array(score_history))