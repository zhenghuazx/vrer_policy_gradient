VRER: A Sample-Efficient Variance Reduction based Experience Replay Method for Policy Optimization Algorithms in tf2
===========

[![zheng](https://img.shields.io/badge/Author-Zheng.H-yellow)](https://zhenghuazx.github.io/hua.zheng/)
[![GitHub issues](https://img.shields.io/github/issues/zhenghuazx/vrer_policy_gradient)](https://github.com/zhenghuazx/vrer_policy_gradient/issues)
[![GitHub license](https://img.shields.io/github/license/zhenghuazx/vrer_policy_gradient)](https://github.com/zhenghuazx/vrer_policy_gradient/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/zhenghuazx/vrer_policy_gradient)](https://github.com/zhenghuazx/vrer_policy_gradient/stargazers)


**vrer-pg**, short for variance reduction based policy gradient, is a library of variance reduction based experience replay method proposed in the paper "Zheng, H., et al., 2022. Variance Reduction based Experience Replay for Reinforcement Learning Policy Optimization."

The research was conducted by Hua Zheng and supervised by Professor Wei Xie, Professor Ben Feng. We would appreciate a citation if you use the code or results! 

**Acknowledge**: This library is built on top of rlalgorithms-tf2 which no longer exists on Github. But I would like to acknowledge its author **unsignedrant**.

* [Introduction](#1-introduction)
  * [Description](#11-description)
  * [Installation](#12-installation)
  * [Results](#13-results)
  * [Reproduction](#-14-reproduction)
* [Features](#2-features)
  * [Command line options](#21-command-line-options)
  * [Intuitive hyperparameter tuning from cli](#22-intuitive-hyperparameter-tuning-from-cli)
  * [Early stopping / reduce on plateau](#23-early-stopping--reduce-on-plateau)
  * [Models are loaded from .cfg files](#24-models-are-loaded-from-cfg-files)
  * [Training history checkpoints](#25-training-history-checkpoints)
  * [Reproducible results](#26-reproducible-results)
  * [Gameplay output to .jpg frames or .mp4 vid](#27-gameplay-output-to-jpg-frames)
  * [Resumable training and history](#28-resume-training--history)
* [Usage](#3-usage)
* [Algorithms](#4-algorithms)
  * [TRPO and TRPO-VRER](#41-trpo-vrer)
  * [PPO and PPO-VRER](#42-ppovrer-and-ppo)
* [Contact](#5-contact)
* [Cite Us](#6-cite-us)


<!-- INTRODUCTION -->
## **1. Introduction**
___


Experience replay allows agents to remember and reuse historical transitions. However, the uniform reuse strategy regardless of their
significance is implicitly biased toward out-of-date observations. To overcome this limitation, we propose a general variance reduction based experience reply (VRER) approach, which allows policy optimization algorithms to selectively reuse the most relevant samples and improve policy gradient estimation. It tends to put more weight on historical observations that are more likely sampled from the target distribution. Different from other ER methods VRER is a theoretically justified and simple-to-use approach. Our theoretical and empirical studies demonstrate that the proposed VRER can accelerate the learning of optimal policy and enhance the performance of state-of-the-art policy optimization approaches.

### **1.1. Description**
___
**vrer-pg** is a tensorflow based AI library which facilitates experimentation with
existing reinforcement learning algorithms with variance reduction based policy optimization. 
It provides well tested components that can be easily modified or extended. The available
selection of algorithms can be used directly or through command line.

### **1.2. Installation**
___
 
    pip install git+https://github.com/zhenghuazx/vrer_policy_gradient.git

*Verify installation**

```sh
vrer-pg
```

**OUT:**

	vrer-pg 1.0.1

	Usage:
		vrer-pg <command> <agent> [options] [args]

	Available commands:
		train      Train given an agent and environment
		play       Play a game given a trained agent and environment
		tune       Tune hyperparameters given an agent, hyperparameter specs, and environment

	Use vrer-pg <command> to see more info about a command
	Use vrer-pg <command> <agent> to see more info about command + agent
	

### **1.3. Results**
___

The performance improvement of state-of-the-art PO algorithms after using VRER.
Results are described by the mean performance curves and 95% confidence intervals of PPO(-VRER), TRPO(-VRER) and VPG(-VRER).

![convergence-VPG](assets/Performance.png)

### **1.4. Reproduction**

All results of PPO(-VRER) and TRPO(-VRER) can be reproduced by scripts in ``reproduction`` and results of VPG(-VRER) 
can be reproduced by scripts in ``vrer_policy_gradient/vpgvrer-script`` and ``vrer_policy_gradient/vpg-script``.


<!-- FEATURES -->
## **2. Features**
___

### **2.1. Command line options**

All features are available through the command line. For more command line info,
check [command line options](#5-command-line-options)

![installation](/assets/vrer.gif)
### **2.2. Intuitive hyperparameter tuning from cli**

Command line tuning interface based on [optuna](https://optuna.org), which provides 
many hyperparameter features and types. 3 types are currently used by vrer-pg:

* **Categorical**:
  
      vrer-pg tune <agent> --env <env> --interesting-param <val1> <val2> <val3> # ...

* **Int / log uniform**:

      vrer-pg tune <agent> --env <env> --interesting-param <min-val> <max-val>

And in both examples if `--interesting-param` is not specified, it will have the default value, 
or a fixed value, if only 1 value is specified. 

### **2.3. Early stopping / reduce on plateau.**

Early train stopping usually when plateau is reached for a pre-specified
n number of times without any improvement. Learning rate is
reduced by some pre-determined factor. To activate these features: 

    --divergence-monitoring-steps <train-steps-at-which-should-monitor>
    
### **2.4. Models are loaded from .cfg files**

To facilitate experimentation, and eliminate redundancy, all agents support
loading models by passing either `--model <model.cfg>` or `--actor-model <actor.cfg>` and 
`--critic-model <critic.cfg>`. If no models were passed, the default ones will be loaded.
A typical `model.cfg` file would look like:

    [convolutional-0]
    filters=32
    size=8
    stride=4
    activation=relu
    initializer=orthogonal
    gain=1.4142135
    
    [convolutional-1]
    filters=64
    size=4
    stride=2
    activation=relu
    initializer=orthogonal
    gain=1.4142135
    
    [convolutional-2]
    filters=64
    size=3
    stride=1
    activation=relu
    initializer=orthogonal
    gain=1.4142135
    
    [flatten-0]
    
    [dense-0]
    units=512
    activation=relu
    initializer=orthogonal
    gain=1.4142135
    common=1
    
    [dense-1]
    initializer=orthogonal
    gain=0.01
    output=1
    
    [dense-2]
    initializer=orthogonal
    gain=1.0
    output=1

Which should generate a keras model similar to this one with output units 6, and 1 respectively:

    Model: "model"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            [(None, 84, 84, 1)]  0                                            
    __________________________________________________________________________________________________
    conv2d (Conv2D)                 (None, 20, 20, 32)   2080        input_1[0][0]                    
    __________________________________________________________________________________________________
    conv2d_1 (Conv2D)               (None, 9, 9, 64)     32832       conv2d[0][0]                     
    __________________________________________________________________________________________________
    conv2d_2 (Conv2D)               (None, 7, 7, 64)     36928       conv2d_1[0][0]                   
    __________________________________________________________________________________________________
    flatten (Flatten)               (None, 3136)         0           conv2d_2[0][0]                   
    __________________________________________________________________________________________________
    dense (Dense)                   (None, 512)          1606144     flatten[0][0]                    
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 6)            3078        dense[0][0]                      
    __________________________________________________________________________________________________
    dense_2 (Dense)                 (None, 1)            513         dense[0][0]                      
    ==================================================================================================
    Total params: 1,681,575
    Trainable params: 1,681,575
    Non-trainable params: 0
    __________________________________________________________________________________________________

**Notes**

* You don't have to worry about this if you're going to use the default models,
  which are loaded automatically.
* `common=1` marks a layer to be reused by the following layers, which means
`dense-1` and `dense-2` are called on the output of `dense-0`.

### **2.5. Training history checkpoints**

Saving training history is available for further benchmarking / visualizing results.
This is achieved by specifying `--history-checkpoint <history.parquet>` which will result
in a `.parquet` that will be updated at each episode end. A sample data point will have these 
columns:

* `mean_reward` most recent mean of agent episode rewards.
* `best_reward` most recent best of agent episode rewards.
* `episode_reward` most recent episode reward.
* `step` most recent agent step.
* `time` training elapsed time.

### **2.6. Reproducible results**

All operation results are reproducible by passing `--seed <some-seed>` or `seed=some_seed` 
to agent constructor.

### **2.7. Gameplay output to .jpg frames**

Gameplay visual output can be saved to `.jpg` frames by passing `--frame-dir <some-dir>` to `play` command.

### **2.8. Resume training / history**

Weights are saved to `.tf` by specifying `--checkpoints <ckpt1.tf> <ckpt2.tf>`. To resume training,
`--weights <ckpt1.tf> <ckpt2.tf>` should load the weights saved earlier. If `--history-checkpoint <ckpt.parquet>`
is specified, the file is looked for and if found, further training history will be saved
to the same history `ckpt.parquet` and the agent metrics will be updated with the most
recent ones contained in the history file.

<!-- USAGE -->
## **3. Usage**
___
All agents / commands are available through the command line.

    vrer-pg <command> <agent> [options] [args]

**Note:** Unless called from command line with `--weights` passed,
all models passed to agents in code, should be loaded with weights 
beforehand, if called for resuming training or playing.

**Command line**
```sh
vrer-pg train trpovrer
```
**Out**
	
	vrer-pg 1.0.1

	Usage:
		vrer-pg <command> <agent> [options] [args]

	Available commands:
		train      Train given an agent and environment
		play       Play a game given a trained agent and environment
		tune       Tune hyperparameters given an agent, hyperparameter specs, and environment

	Use vrer-pg <command> to see more info about a command
	Use vrer-pg <command> <agent> to see more info about command + agent
	
<!-- ALGORITHMS -->
## **4. Algorithms**
___
**General notes**

* All the default hyperparameters don't work for all environments.
  Which means you either need to tune them according to the given environment,
  or pass previously tuned ones, in order to get good results.
* `--model <model.cfg>` or `--actor-model <actor.cfg>` and `--critic-model <critic.cfg>` are optional 
  which means, if not specified, the default model(s) will be loaded, so you don't have to worry about it.
* You can also use external models by passing them to agent constructor. If you do, you will have to ensure
  your models outputs match what the implementation expects, or modify it accordingly.
* For atari environments / the ones that return an image by default, use the `--preprocess` flag for image preprocessing.
* For checkpoints to be saved, `--checkpoints <checkpoint1.tf> <checkpoint2.tf>` should
be specified for the model(s) to be saved. The number of passed checkpoints should match the number
  of models the agent accepts.
* For loading weights either for resuming training or for playing a game `--weights <weights1.tf> <weights2.tf>`
and same goes for the weights, they should match the number of agent models.
* For using a random seed, a `seed=some_seed` should be passed to agent constructor and ModelReader constructor if
specified from code. If from the command line, all you need is to pass `--seed <some-seed>`
* To save training history, `history_checkpoint=some_history.parquet` should be specified
to agent constructor or alternatively using `--history-checkpoint <some-history.parquet>`. 
  If the history checkpoint exists, training metrics will automatically start from where it left.
  
### *4.1. TRPO-VRER*
* *Number of models:* 1
* *Action spaces:* discrete, continuous

| flags                         | help                                                                               | default   | hp_type     |
|:------------------------------|:-----------------------------------------------------------------------------------|:----------|:------------|
| --actor-iterations            | Actor optimization iterations per train step                                       | 10        | int         |
| --actor-model                 | Path to actor model .cfg file                                                      | -         | -           |
| --advantage-epsilon           | Value added to estimated advantage                                                 | 1e-08     | log_uniform |
| --beta1                       | Beta1 passed to a tensorflow.keras.optimizers.Optimizer                            | 0.9       | log_uniform |
| --beta2                       | Beta2 passed to a tensorflow.keras.optimizers.Optimizer                            | 0.999     | log_uniform |
| --buffer_size                 | Maximum capacity of replay buffer                                                  | 100       | categorical |
| --cg-damping                  | Gradient conjugation damping parameter                                             | 0.001     | log_uniform |
| --cg-iterations               | Gradient conjugation iterations per train step                                     | 10        | -           |
| --cg-residual-tolerance       | Gradient conjugation residual tolerance parameter                                  | 1e-10     | log_uniform |
| --checkpoints                 | Path(s) to new model(s) to which checkpoint(s) will be saved during training       | -         | -           |
| --clip-norm                   | Clipping value passed to tf.clip_by_value()                                        | 0.1       | log_uniform |
| --critic-iterations           | Critic optimization iterations per train step                                      | 3         | int         |
| --critic-model                | Path to critic model .cfg file                                                     | -         | -           |
| --display-precision           | Number of decimals to be displayed                                                 | 2         | -           |
| --divergence-monitoring-steps | Steps after which, plateau and early stopping are active                           | -         | -           |
| --early-stop-patience         | Minimum plateau reduces to stop training                                           | 3         | -           |
| --entropy-coef                | Entropy coefficient for loss calculation                                           | 0         | log_uniform |
| --env                         | gym environment id                                                                 | -         | -           |
| --fvp-n-steps                 | Value used to skip every n-frames used to calculate FVP                            | 5         | int         |
| --gamma                       | Discount factor                                                                    | 0.99      | log_uniform |
| --grad-norm                   | Gradient clipping value passed to tf.clip_by_value()                               | 0.5       | log_uniform |
| --history-checkpoint          | Path to .parquet file to save training history                                     | -         | -           |
| --lam                         | GAE-Lambda for advantage estimation                                                | 1.0       | log_uniform |
| --log-frequency               | Log progress every n games                                                         | -         | -           |
| --lr                          | Learning rate passed to a tensorflow.keras.optimizers.Optimizer                    | 0.0007    | log_uniform |
| --max-frame                   | If specified, max & skip will be applied during preprocessing                      | -         | categorical |
| --max-kl                      | Maximum KL divergence used for calculating Lagrange multiplier                     | 0.001     | log_uniform |
| --max-steps                   | Maximum number of environment steps, when reached, training is stopped             | -         | -           |
| --mini-batches                | Number of mini-batches to use per update                                           | 4         | categorical |
| --monitor-session             | Wandb session name                                                                 | -         | -           |
| --n-envs                      | Number of environments to create                                                   | 1         | categorical |
| --n-steps                     | Transition steps                                                                   | 512       | categorical |
| --num_reuse_each_iter         | Number of randomly sampled transition from each behavioral policy in the reuse set | 3         | categorical |
| --opt-epsilon                 | Epsilon passed to a tensorflow.keras.optimizers.Optimizer                          | 1e-07     | log_uniform |
| --plateau-reduce-factor       | Factor multiplied by current learning rate when there is a plateau                 | 0.9       | -           |
| --plateau-reduce-patience     | Minimum non-improvements to reduce lr                                              | 10        | -           |
| --ppo-epochs                  | Gradient updates per training step                                                 | 4         | categorical |
| --preprocess                  | If specified, states will be treated as atari frames                               | -         | -           |
|                               | and preprocessed accordingly                                                       |           |             |
| --quiet                       | If specified, no messages by the agent will be displayed                           | -         | -           |
|                               | to the console                                                                     |           |             |
| --reward-buffer-size          | Size of the total reward buffer, used for calculating                              | 100       | -           |
|                               | mean reward value to be displayed.                                                 |           |             |
| --seed                        | Random seed                                                                        | -         | -           |
| --target-reward               | Target reward when reached, training is stopped                                    | -         | -           |
| --value-loss-coef             | Value loss coefficient for value loss calculation                                  | 0.5       | log_uniform |
| --weights                     | Path(s) to model(s) weight(s) to be loaded by agent output_models                  | -         | -           |

**Run TRPO with VRER with command line**
	
	vrer-pg train trpovrer --n-env 4 --target-reward 195 --env CartPole-v0
   	
```sh
number of reuse:  0
time: 0:00:15.744019, steps: 2048, games: 101, speed: 130 steps/s, mean reward: 20.1, best reward: -inf
number of reuse:  1
Best reward updated: -inf -> 20.1
time: 0:00:17.821029, steps: 4096, games: 183, speed: 986 steps/s, mean reward: 24.3, best reward: 20.1
number of reuse:  1
Best reward updated: 20.1 -> 24.3
time: 0:00:20.266888, steps: 6144, games: 266, speed: 837 steps/s, mean reward: 24.25, best reward: 24.3
number of reuse:  3
time: 0:00:22.747528, steps: 8192, games: 345, speed: 826 steps/s, mean reward: 25.9, best reward: 24.3
number of reuse:  0
Best reward updated: 24.3 -> 25.9
time: 0:00:24.894729, steps: 10240, games: 410, speed: 954 steps/s, mean reward: 29.28, best reward: 25.9
number of reuse:  5
Best reward updated: 25.9 -> 29.28
time: 0:00:26.956820, steps: 12288, games: 481, speed: 993 steps/s, mean reward: 30.03, best reward: 29.28
number of reuse:  4
Best reward updated: 29.28 -> 30.03
time: 0:00:29.242799, steps: 14336, games: 548, speed: 896 steps/s, mean reward: 31.22, best reward: 30.03
number of reuse:  0
Best reward updated: 30.03 -> 31.22
time: 0:00:31.301686, steps: 16384, games: 600, speed: 995 steps/s, mean reward: 34.61, best reward: 31.22
number of reuse:  5
Best reward updated: 31.22 -> 34.61
time: 0:00:33.378286, steps: 18432, games: 648, speed: 986 steps/s, mean reward: 41.5, best reward: 34.61
number of reuse:  0
Best reward updated: 34.61 -> 41.5
time: 0:00:35.461156, steps: 20480, games: 691, speed: 983 steps/s, mean reward: 45.26, best reward: 41.5
number of reuse:  0
Best reward updated: 41.5 -> 45.26
time: 0:00:37.535053, steps: 22528, games: 729, speed: 988 steps/s, mean reward: 50.42, best reward: 45.26
number of reuse:  11
Best reward updated: 45.26 -> 50.42
time: 0:00:39.798340, steps: 24576, games: 768, speed: 905 steps/s, mean reward: 50.69, best reward: 50.42
number of reuse:  12
Best reward updated: 50.42 -> 50.69
time: 0:00:42.106897, steps: 26624, games: 801, speed: 887 steps/s, mean reward: 56.27, best reward: 50.69
number of reuse:  13
Best reward updated: 50.69 -> 56.27
time: 0:00:44.179972, steps: 28672, games: 833, speed: 988 steps/s, mean reward: 59.49, best reward: 56.27
```

#### **4.1.1 TRPO Non-command line**
```python
''' TRPO '''
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
```

#### **4.1.2 TRPO-VRER Non-command line**
    
```python
''' TRPO-VRER '''
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
    entropy_coef = 0.0
    mini_batches = 32
    lr = 0.0003
    max_steps = 400000
    target_reward = -70
    buffer_size = 100
    problem = 'Acrobot-v1'
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
```

### *4.2. PPOVRER and PPO*

* *Number of models:* 1
* *Action spaces:* discrete, continuous

| flags                         | help                                                                               | default   | hp_type     |
|:------------------------------|:-----------------------------------------------------------------------------------|:----------|:------------|
| --advantage-epsilon           | Value added to estimated advantage                                                 | 1e-08     | log_uniform |
| --beta1                       | Beta1 passed to a tensorflow.keras.optimizers.Optimizer                            | 0.9       | log_uniform |
| --beta2                       | Beta2 passed to a tensorflow.keras.optimizers.Optimizer                            | 0.999     | log_uniform |
| --buffer_size                 | Maximum capacity of replay buffer                                                  | 100       | categorical |
| --checkpoints                 | Path(s) to new model(s) to which checkpoint(s) will be saved during training       | -         | -           |
| --clip-norm                   | Clipping value passed to tf.clip_by_value()                                        | 0.1       | log_uniform |
| --display-precision           | Number of decimals to be displayed                                                 | 2         | -           |
| --divergence-monitoring-steps | Steps after which, plateau and early stopping are active                           | -         | -           |
| --early-stop-patience         | Minimum plateau reduces to stop training                                           | 3         | -           |
| --entropy-coef                | Entropy coefficient for loss calculation                                           | 0.01      | log_uniform |
| --env                         | gym environment id                                                                 | -         | -           |
| --gamma                       | Discount factor                                                                    | 0.99      | log_uniform |
| --grad-norm                   | Gradient clipping value passed to tf.clip_by_value()                               | 0.5       | log_uniform |
| --history-checkpoint          | Path to .parquet file to save training history                                     | -         | -           |
| --lam                         | GAE-Lambda for advantage estimation                                                | 0.95      | log_uniform |
| --log-frequency               | Log progress every n games                                                         | -         | -           |
| --lr                          | Learning rate passed to a tensorflow.keras.optimizers.Optimizer                    | 0.0007    | log_uniform |
| --max-frame                   | If specified, max & skip will be applied during preprocessing                      | -         | categorical |
| --max-steps                   | Maximum number of environment steps, when reached, training is stopped             | -         | -           |
| --mini-batches                | Number of mini-batches to use per update                                           | 4         | categorical |
| --model                       | Path to model .cfg file                                                            | -         | -           |
| --monitor-session             | Wandb session name                                                                 | -         | -           |
| --n-envs                      | Number of environments to create                                                   | 1         | categorical |
| --n-steps                     | Transition steps                                                                   | 128       | categorical |
| --num_reuse_each_iter         | Number of randomly sampled transition from each behavioral policy in the reuse set | 3         | categorical |
| --opt-epsilon                 | Epsilon passed to a tensorflow.keras.optimizers.Optimizer                          | 1e-07     | log_uniform |
| --plateau-reduce-factor       | Factor multiplied by current learning rate when there is a plateau                 | 0.9       | -           |
| --plateau-reduce-patience     | Minimum non-improvements to reduce lr                                              | 10        | -           |
| --ppo-epochs                  | Gradient updates per training step                                                 | 4         | categorical |
| --preprocess                  | If specified, states will be treated as atari frames                               | -         | -           |
|                               | and preprocessed accordingly                                                       |           |             |
| --quiet                       | If specified, no messages by the agent will be displayed                           | -         | -           |
|                               | to the console                                                                     |           |             |
| --reward-buffer-size          | Size of the total reward buffer, used for calculating                              | 100       | -           |
|                               | mean reward value to be displayed.                                                 |           |             |
| --seed                        | Random seed                                                                        | -         | -           |
| --target-reward               | Target reward when reached, training is stopped                                    | -         | -           |
| --value-loss-coef             | Value loss coefficient for value loss calculation                                  | 0.5       | log_uniform |
| --weights                     | Path(s) to model(s) weight(s) to be loaded by agent output_models                  | -         | -           |

**Command line**

    vrer-pg train ppo --env PongNoFrameskip-v4 --target-reward 19 --n-envs 16 --preprocess --checkpoints ppo-pong.tf

or

    vrer-pg train ppovrer --env BipedalWalker-v3 --target-reward 200 --n-envs 16 --checkpoints ppo-bipedal-walker.tf

#### **4.2.1 PPO Non-command line**
    
```python
''' PPO '''
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
```

    
#### **4.2.2 PPO-VRER Non-command line**
```python
''' PPO-VRER '''
from tensorflow.keras.optimizers import Adam
import os
from vrer_policy_gradient import PPO, PPOVRER
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
    buffer_size = 400
    target_reward = 201
    problem = 'CartPole-v0'
    checkpoints = 'ppo-problem-{}-buffer_size-{}-seed-{}.tf'.format(problem, buffer_size, i)
    history_checkpoint = 'ppo-problem-{}-buffer_size-{}-seed-{}.parquet'.format(problem, buffer_size, i)
    path = "PPO/{}/approx-2nd-buffer_size-{}-seed-{}-id-{}/".format(problem, buffer_size, seed, i)
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
    checkpoints = path + checkpoints
    history_checkpoint = path + history_checkpoint

    envs = create_envs(problem, n_envs, False)

    model_cfg = vrer_policy_gradient.agents['ppovrer']['model']['ann'][0]
    optimizer = Adam(learning_rate=lr)
    optimizer = SGD(learning_rate=lr)
    model = ModelReader(
        model_cfg,
        seed=seed,
        output_units=[envs[0].action_space.n, 1],
        input_shape=envs[0].observation_space.shape,
        optimizer=optimizer,
    ).build_model()
    agent = PPOVRER(envs, model, seed=seed, n_steps=n_steps, entropy_coef=entropy_coef, mini_batches=mini_batches, clip_norm=clip_norm, checkpoints=[checkpoints], history_checkpoint=history_checkpoint, log_frequency=4, buffer_size=buffer_size)
    agent.fit(target_reward=target_reward, max_steps=max_steps)
```

## **5. Contact**
___

Website: https://zhenghuazx.github.io/hua.zheng/

Email: hua.zheng0908@gmail.com

Project link: https://github.com/zhenghuazx/vrer_policy_gradient


## **6. Cite Us**
___
The paper is under review. Please check the arXiv page: https://arxiv.org/abs/2110.08902.
