from vrer_policy_gradient import a2c, ppo, trpo, a2cvrer, ppovrer, trpovrer, acer
from vrer_policy_gradient.a2c.agent import A2C
from vrer_policy_gradient.a2cvrer.agent import A2CVRER
from vrer_policy_gradient.base import OffPolicy
from vrer_policy_gradient.ppo.agent import PPO
from vrer_policy_gradient.trpo.agent import TRPO
from vrer_policy_gradient.utils.cli import play_args, train_args, tune_args
from vrer_policy_gradient.utils.common import register_models
from vrer_policy_gradient.ppovrer.agent import PPOVRER
from vrer_policy_gradient.trpovrer.agent import TRPOVRER
from vrer_policy_gradient.acer.agent import ACER

__author__ = 'Hua Zheng and unsignedrant'
__email__ = 'hua.zheng0908@gmail.com'
__license__ = 'MIT'
__version__ = '1.1.1'

agents = {
    'a2c': {'module': a2c, 'agent': A2C},
    'ppo': {'module': ppo, 'agent': PPO},
    'trpo': {'module': trpo, 'agent': TRPO},
    'a2cvrer': {'module': a2cvrer, 'agent': A2CVRER},
    'ppovrer': {'module': ppovrer, 'agent': PPOVRER},
    'trpovrer': {'module': trpovrer, 'agent': TRPOVRER},
    'acer': {'module': acer, 'agent': ACER},
}
register_models(agents)
commands = {
    'train': (train_args, 'fit', 'Train given an agent and environment'),
    'play': (
        play_args,
        'play',
        'Play a game given a trained agent and environment',
    ),
    'tune': (
        tune_args,
        '',
        'Tune hyperparameters given an agent, hyperparameter specs, and environment',
    ),
}
