from tensorflow.keras.optimizers import Adam
import pybullet_envs
import os
from vrer_policy_gradient import PPO, PPOVRER
from vrer_policy_gradient.utils.common import ModelReader, create_envs
import vrer_policy_gradient

for i in range(5):
    seed = i + 2021
    n_envs = 4
    n_steps = 128
    clip_norm = 0.2
    entropy_coef = 0.01
    mini_batches = 128
    lr = 0.0003
    max_steps = 2000000
    target_reward = 10000
    problem = 'InvertedPendulumBulletEnv-v0'
    buffer_size = 100
    checkpoints = 'ppo-problem-{}-buffer_size-{}-seed-{}.tf'.format(problem, buffer_size, i)
    history_checkpoint = 'ppo-problem-{}-buffer_size-{}-seed-{}.parquet'.format(problem, buffer_size, i)
    path = "PPO/{}/approx-2nd-buffer_size-{}-seed-{}-id-{}/".format(problem, buffer_size, seed, i)
    isExist = os.path.exists(path)
    # if not isExist:
    #     # Create a new directory because it does not exist
    #     os.makedirs(path)
    checkpoints = path + checkpoints
    history_checkpoint = path + history_checkpoint

    envs = create_envs(problem, n_envs, False)
    model_cfg = vrer_policy_gradient.agents['ppovrer']['model']['ann'][0]
    # model_cfg = 'ppo/model/ann-actor-critic.cfg'
    optimizer = Adam(learning_rate=lr)
    model = ModelReader(
        model_cfg,
        seed=seed,
        output_units=[envs[0].action_space.shape[0], 1],
        input_shape=envs[0].observation_space.shape,
        optimizer=optimizer,
    ).build_model()
    agent = PPOVRER(envs, model, seed=seed, n_steps=n_steps, entropy_coef=entropy_coef, mini_batches=mini_batches, clip_norm=clip_norm, checkpoints=[checkpoints], history_checkpoint=history_checkpoint, log_frequency=4, buffer_size=buffer_size)
    agent.fit(target_reward=target_reward, max_steps=max_steps)