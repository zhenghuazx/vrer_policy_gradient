from tensorflow.keras.optimizers import Adam
import os
from vrer_policy_gradient import PPO
from vrer_policy_gradient.utils.common import ModelReader, create_envs

for i in range(5):
    seed = i + 2021
    n_envs = 4
    n_steps = 128
    clip_norm = 0.2
    entropy_coef = 0.01
    mini_batches = 128
    lr = 0.0003
    max_steps = 1000000
    target_reward = -70
    problem = 'Acrobot-v1'
    checkpoints = 'ppo-acrobot-v1-seed-{}.tf'.format(i)
    history_checkpoint = 'ppo-acrobot-v1-seed-{}.parquet'.format(i)
    path = "PPO/{}/approx-2nd-seed-{}-id-{}/".format(problem, seed, i)
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
    checkpoints = path + checkpoints
    history_checkpoint = path + history_checkpoint

    envs = create_envs(problem, n_envs, False)
    model_cfg = 'model/ann-actor-critic.cfg' # rlalgorithms_tf2.agents['ppo']['model']['ann'][0]
    optimizer = Adam(learning_rate=lr)
    model = ModelReader(
        model_cfg,
        seed=seed,
        output_units=[envs[0].action_space.n, 1],
        input_shape=envs[0].observation_space.shape,
        optimizer=optimizer,
    ).build_model()
    agent = PPO(envs, model, seed=seed, n_steps=n_steps, entropy_coef=entropy_coef, mini_batches=mini_batches, clip_norm=clip_norm, checkpoints=[checkpoints], history_checkpoint=history_checkpoint, log_frequency=4)
    agent.fit(target_reward=target_reward, max_steps=max_steps)
