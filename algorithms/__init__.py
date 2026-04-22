from .es import ES, ESConfig
from .random_search import RandomPolicy
from .dqn import DQN, DQNConfig
from .a3c import A3C, A3CConfig
from .ppo import PPO, PPOConfig

REGISTRY = {
    "es":     ES,
    "random": RandomPolicy,
    "dqn":    DQN,
    "a3c":    A3C,
    "ppo":    PPO,
}
