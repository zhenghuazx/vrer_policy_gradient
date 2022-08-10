from tensorflow.keras.optimizers import Adam
import os
from vrer_policy_gradient import PPO, TRPOVRER
from vrer_policy_gradient.utils.common import ModelReader, create_envs
from vrer_policy_gradient.utils.buffers import ReplayBuffer1
import vrer_policy_gradient

for i in range(5):
    seed = i + 2021
    n_envs = 4
    n_steps = 128
    clip_norm = 0.2
    entropy_coef = 0.01
    mini_batches = 128
    lr = 0.0003
    max_steps = 400000
    buffer_size = 100
    target_reward = 201
    problem = 'CartPole-v0'
    checkpoints = ['trpovrer-actor-problem-{}-buffer_size-{}-seed-{}.tf'.format(problem, buffer_size, i),
                   'trpovrer-critic-problem-{}-buffer_size-{}-seed-{}.tf'.format(problem, buffer_size, i)]
    history_checkpoint = 'trpovrer-problem-{}-buffer_size-{}-seed-{}.parquet'.format(problem, buffer_size, i)
    path = "trpovrer/{}/approx-2nd-buffer_size-{}-seed-{}-id-{}/".format(problem, buffer_size, seed, i)
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
    checkpoints = [path + i for i in checkpoints]
    history_checkpoint = path + history_checkpoint

    envs = create_envs(problem, n_envs, False)

    actor_model_cfg = vrer_policy_gradient.agents['trpovrer']['actor_model']['ann'][0]
    critic_model_cfg = vrer_policy_gradient.agents['trpovrer']['critic_model']['ann'][0]
    optimizer = Adam(learning_rate=lr)
    actor_model = ModelReader(
        actor_model_cfg,
        output_units=[envs[0].action_space.n],
        input_shape=envs[0].observation_space.shape,
        optimizer=optimizer,
    ).build_model()
    critic_model = ModelReader(
        actor_model_cfg,
        output_units=[1],
        input_shape=envs[0].observation_space.shape,
        optimizer=optimizer,
    ).build_model()

    agent = TRPOVRER(envs, actor_model, critic_model, seed=seed, n_steps=n_steps, entropy_coef=entropy_coef, mini_batches=mini_batches,
                 clip_norm=clip_norm, checkpoints=checkpoints, history_checkpoint=history_checkpoint, log_frequency=4, buffer_size=buffer_size)
    agent.fit(target_reward=target_reward, max_steps=max_steps)