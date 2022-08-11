VRER: A Sample-Efficient Variance Reduction based Experience Replay Method for Policy Optimization Algorithms in tf2
===========

[![zheng](https://img.shields.io/badge/Author-Zheng.H-yellow)](https://zhenghuazx.github.io/hua.zheng/)


vrer_policy_gradient is a library of variance reduction based experience replay method proposed in the paper "Zheng, H., et al., 2022. Variance Reduction based Experience Replay for Reinforcement Learning Policy Optimization."

The research was conducted by Hua Zheng and supervised by Professor Wei Xie, Professor Ben Feng. We would appreciate a citation if you use the code or results! 

Acknowledge: This library is built on top of rlalgorithms-tf2 which no longer exists on Github. But I would like to acknowledge its author **unsignedrant**.

### **1. Installation**
___


Experience replay allows agents to remember and reuse historical transitions. However, the uniform reuse strategy regardless of their
significance is implicitly biased toward out-of-date observations. To overcome this limitation, we propose a general variance reduction based experience reply (VRER) approach, which allows policy optimization algorithms to selectively reuse the most relevant samples and improve policy gradient estimation. It tends to put more weight on historical observations that are more likely sampled from the target distribution. Different from other ER methods VRER is a theoretically justified and simple-to-use approach. Our theoretical and empirical studies demonstrate that the proposed VRER can accelerate the learning of optimal policy and enhance the performance of state-of-the-art policy optimization approaches.

### **1. Installation**
___

![installation](/assets/installation.gif)
    
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

<!-- DESCRIPTION -->


## **2. Description**
___
**vrer-pg** is a tensorflow based AI library which facilitates experimentation with
existing reinforcement learning algorithms with variance reduction based policy optimization. 
The current implementation is based on **rlalgorithms-tf2** library (which unforunately is discontiued in Github).
It provides well tested components that can be easily modified or extended. The available
selection of algorithms can be used directly or through command line.

<!-- FEATURES -->
## **3. Features**

### **3.1. Command line options**

All features are available through the command line. For more command line info,
check [command line options](#5-command-line-options)

### **3.2. Intuitive hyperparameter tuning from cli**

Command line tuning interface based on [optuna](https://optuna.org), which provides 
many hyperparameter features and types. 3 types are currently used by rlalgorithms-tf2:

* **Categorical**:
  
      rlalgorithms-tf2 tune <agent> --env <env> --interesting-param <val1> <val2> <val3> # ...

* **Int / log uniform**:

      rlalgorithms-tf2 tune <agent> --env <env> --interesting-param <min-val> <max-val>

And in both examples if `--interesting-param` is not specified, it will have the default value, 
or a fixed value, if only 1 value is specified. 

### **3.3. Early stopping / reduce on plateau.**

Early train stopping usually when plateau is reached for a pre-specified
n number of times without any improvement. Learning rate is
reduced by some pre-determined factor. To activate these features: 

    --divergence-monitoring-steps <train-steps-at-which-should-monitor>
    
### **3.4. Models are loaded from .cfg files**

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

### **3.5. Training history checkpoints**

Saving training history is available for further benchmarking / visualizing results.
This is achieved by specifying `--history-checkpoint <history.parquet>` which will result
in a `.parquet` that will be updated at each episode end. A sample data point will have these 
columns:

* `mean_reward` most recent mean of agent episode rewards.
* `best_reward` most recent best of agent episode rewards.
* `episode_reward` most recent episode reward.
* `step` most recent agent step.
* `time` training elapsed time.

### **3.6. Reproducible results**

All operation results are reproducible by passing `--seed <some-seed>` or `seed=some_seed` 
to agent constructor.

### **3.7. Gameplay output to .jpg frames**

Gameplay visual output can be saved to `.jpg` frames by passing `--frame-dir <some-dir>` to `play` command.

### **3.8. Resume training / history**

Weights are saved to `.tf` by specifying `--checkpoints <ckpt1.tf> <ckpt2.tf>`. To resume training,
`--weights <ckpt1.tf> <ckpt2.tf>` should load the weights saved earlier. If `--history-checkpoint <ckpt.parquet>`
is specified, the file is looked for and if found, further training history will be saved
to the same history `ckpt.parquet` and the agent metrics will be updated with the most
recent ones contained in the history file.

## **4. Usage**
___
All agents / commands are available through the command line.

    rlalgorithms-tf2 <command> <agent> [options] [args]

**Note:** Unless called from command line with `--weights` passed,
all models passed to agents in code, should be loaded with weights 
beforehand, if called for resuming training or playing.

<!-- ALGORITHMS -->
## **5. Algorithms**
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
  
### *6.1. TRPO-VRER*

**Command line**
	vrer-pg train trpovrer
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

train trpovrer

| flags                         | help                                                                         | default   | hp_type     |
|:------------------------------|:-----------------------------------------------------------------------------|:----------|:------------|
| --actor-iterations            | Actor optimization iterations per train step                                 | 10        | int         |
| --actor-model                 | Path to actor model .cfg file                                                | -         | -           |
| --advantage-epsilon           | Value added to estimated advantage                                           | 1e-08     | log_uniform |
| --beta1                       | Beta1 passed to a tensorflow.keras.optimizers.Optimizer                      | 0.9       | log_uniform |
| --beta2                       | Beta2 passed to a tensorflow.keras.optimizers.Optimizer                      | 0.999     | log_uniform |
| --cg-damping                  | Gradient conjugation damping parameter                                       | 0.001     | log_uniform |
| --cg-iterations               | Gradient conjugation iterations per train step                               | 10        | -           |
| --cg-residual-tolerance       | Gradient conjugation residual tolerance parameter                            | 1e-10     | log_uniform |
| --checkpoints                 | Path(s) to new model(s) to which checkpoint(s) will be saved during training | -         | -           |
| --clip-norm                   | Clipping value passed to tf.clip_by_value()                                  | 0.1       | log_uniform |
| --critic-iterations           | Critic optimization iterations per train step                                | 3         | int         |
| --critic-model                | Path to critic model .cfg file                                               | -         | -           |
| --display-precision           | Number of decimals to be displayed                                           | 2         | -           |
| --divergence-monitoring-steps | Steps after which, plateau and early stopping are active                     | -         | -           |
| --early-stop-patience         | Minimum plateau reduces to stop training                                     | 3         | -           |
| --entropy-coef                | Entropy coefficient for loss calculation                                     | 0         | log_uniform |
| --env                         | gym environment id                                                           | -         | -           |
| --fvp-n-steps                 | Value used to skip every n-frames used to calculate FVP                      | 5         | int         |
| --gamma                       | Discount factor                                                              | 0.99      | log_uniform |
| --grad-norm                   | Gradient clipping value passed to tf.clip_by_value()                         | 0.5       | log_uniform |
| --history-checkpoint          | Path to .parquet file to save training history                               | -         | -           |
| --lam                         | GAE-Lambda for advantage estimation                                          | 1.0       | log_uniform |
| --log-frequency               | Log progress every n games                                                   | -         | -           |
| --lr                          | Learning rate passed to a tensorflow.keras.optimizers.Optimizer              | 0.0007    | log_uniform |
| --max-frame                   | If specified, max & skip will be applied during preprocessing                | -         | categorical |
| --max-kl                      | Maximum KL divergence used for calculating Lagrange multiplier               | 0.001     | log_uniform |
| --max-steps                   | Maximum number of environment steps, when reached, training is stopped       | -         | -           |
| --mini-batches                | Number of mini-batches to use per update                                     | 4         | categorical |
| --monitor-session             | Wandb session name                                                           | -         | -           |
| --n-envs                      | Number of environments to create                                             | 1         | categorical |
| --n-steps                     | Transition steps                                                             | 512       | categorical |
| --opt-epsilon                 | Epsilon passed to a tensorflow.keras.optimizers.Optimizer                    | 1e-07     | log_uniform |
| --plateau-reduce-factor       | Factor multiplied by current learning rate when there is a plateau           | 0.9       | -           |
| --plateau-reduce-patience     | Minimum non-improvements to reduce lr                                        | 10        | -           |
| --ppo-epochs                  | Gradient updates per training step                                           | 4         | categorical |
| --preprocess                  | If specified, states will be treated as atari frames                         | -         | -           |
|                               | and preprocessed accordingly                                                 |           |             |
| --quiet                       | If specified, no messages by the agent will be displayed                     | -         | -           |
|                               | to the console                                                               |           |             |
| --reward-buffer-size          | Size of the total reward buffer, used for calculating                        | 100       | -           |
|                               | mean reward value to be displayed.                                           |           |             |
| --seed                        | Random seed                                                                  | -         | -           |
| --target-reward               | Target reward when reached, training is stopped                              | -         | -           |
| --value-loss-coef             | Value loss coefficient for value loss calculation                            | 0.5       | log_uniform |
| --weights                     | Path(s) to model(s) weight(s) to be loaded by agent output_models            | -         | -           |
