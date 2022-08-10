from tensorflow.keras.optimizers import Adam
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
    buffer_size = 50
    max_steps = 400000
    num_reuse_each_iter = 2
    target_reward = -70
    problem = 'Acrobot-v1'
    checkpoints = 'ppovrer-acrobot-buffer_size-{}-seed-{}.tf'.format(buffer_size, i)
    history_checkpoint = 'ppovrer-acrobot-buffer_size-{}-seed-{}.parquet'.format(buffer_size, i)
    path = "PPOVRER/{}/approx-2nd-buffer_size-{}-seed-{}-id-{}/".format(problem, buffer_size, seed, i)
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
    checkpoints = path + checkpoints
    history_checkpoint = path + history_checkpoint

    envs = create_envs(problem, n_envs, False)
    model_cfg = vrer_policy_gradient.agents['ppovrer']['model']['ann'][0]
    optimizer = Adam(learning_rate=lr)
    model = ModelReader(
        model_cfg,
        seed=seed,
        output_units=[envs[0].action_space.n, 1],
        input_shape=envs[0].observation_space.shape,
        optimizer=optimizer,
    ).build_model()
    agent = PPOVRER(envs, model, seed=seed, n_steps=n_steps, entropy_coef=entropy_coef, mini_batches=mini_batches, clip_norm=clip_norm, checkpoints=[checkpoints], history_checkpoint=history_checkpoint, log_frequency=4, buffer_size=buffer_size, num_reuse_each_iter=num_reuse_each_iter)
    agent.fit(target_reward=target_reward, max_steps=max_steps)