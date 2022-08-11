from vrer_policy_gradient import a2c

a2c_args = a2c.cli_args
ppovrer_args = {
    'model': {'help': 'Path to model .cfg file'},
    'lam': {
        'help': 'GAE-Lambda for advantage estimation',
        'type': float,
        'default': 0.95,
        'hp_type': 'log_uniform',
    },
    'ppovrer-epochs': {
        'help': 'Gradient updates per training step',
        'type': int,
        'default': 4,
        'hp_type': 'categorical',
    },
    'mini-batches': {
        'help': 'Number of mini-batches to use per update',
        'type': int,
        'default': 4,
        'hp_type': 'categorical',
    },
    'advantage-epsilon': {
        'help': 'Value added to estimated advantage',
        'type': float,
        'default': 1e-8,
        'hp_type': 'log_uniform',
    },
    'clip-norm': {
        'help': 'Clipping value passed to tf.clip_by_value()',
        'type': float,
        'default': 0.1,
        'hp_type': 'log_uniform',
    },
    'n-steps': {
        'help': 'Transition steps',
        'type': int,
        'default': 128,
        'hp_type': 'categorical',
    },
    'num_reuse_each_iter': {
        'help': 'Number of randomly sampled transition from each behavioral policy in the reuse set',
        'type': int,
        'default': 3,
        'hp_type': 'categorical',
    },
    'buffer_size': {
        'help': 'Maximum capacity of replay buffer ',
        'type': int,
        'default': 100,
        'hp_type': 'categorical',
    },      
}
cli_args = a2c_args.copy()
cli_args.update(ppovrer_args)
