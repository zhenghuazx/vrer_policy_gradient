VRER: A Sample-Efficient Variance Reduction based Experience Replay Method for Policy Optimization Algorithms in tf2
===========

[![zheng](https://img.shields.io/badge/Author-Zheng.H-yellow)](https://zhenghuazx.github.io/hua.zheng/)


vrer_policy_gradient is a library of variance reduction based experience replay method proposed in the paper "Zheng, H., et al., 2022. Variance Reduction based Experience Replay for Reinforcement Learning Policy Optimization."

The research was conducted by Hua Zheng and supervised by Professor Wei Xie, Professor Ben Feng. We would appreciate a citation if you use the code or results!

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
