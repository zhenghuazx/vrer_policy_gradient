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
or a fixed value, if only 1 value is specified. Also, some nice visualization options using 
[optuna.visualization.matplotlib](https://optuna.readthedocs.io/en/latest/reference/visualization/matplotlib.html):

![param-importances](/assets/param-importances.png)

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
