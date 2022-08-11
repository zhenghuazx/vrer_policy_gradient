from tensorflow.keras.optimizers import Adam
import os
import vrer_policy_gradient
from vrer_policy_gradient import TRPO
from vrer_policy_gradient.utils.common import ModelReader, create_envs

for i in range(5):
    seed = i + 2021
    n_envs = 4
    n_steps = 128
    clip_norm = 0.2
    entropy_coef = 0.0
    mini_batches = 32
    lr = 0.0003
    max_steps = 400000
    target_reward = -70
    problem = 'Acrobot-v1'
    checkpoints = ['trpo-actor-acrobot-v1-seed-{}.tf'.format(i), 'trpo-critic-acrobot-v1-seed-{}.tf'.format(i)]
    history_checkpoint = 'trpo-acrobot-v1-seed-{}.parquet'.format(i)
    path = "TRPO/{}/approx-2nd-seed-{}-id-{}/".format(problem, seed, i)
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
    checkpoints = [path + i for i in checkpoints]
    history_checkpoint = path + history_checkpoint

    envs = create_envs(problem, n_envs, False)
    actor_model_cfg = 'TRPO/models/ann-actor.cfg'  # rlalgorithms_tf2.agents['trpo']['actor_model']['ann'][0]
    critic_model_cfg = 'TRPO/models/ann-critic.cfg'  # rlalgorithms_tf2.agents['trpo']['critic_model']['ann'][0]
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
    agent = TRPO(envs, actor_model, critic_model, seed=seed, n_steps=n_steps, entropy_coef=entropy_coef, mini_batches=mini_batches,
                 clip_norm=clip_norm, checkpoints=checkpoints, history_checkpoint=history_checkpoint, log_frequency=4)
    agent.fit(target_reward=target_reward, max_steps=max_steps)
