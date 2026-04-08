from .es import ES, ESConfig
from .random_search import RandomPolicy
from .dqn import DQN, DQNConfig
from .a3c import A3C, A3CConfig

REGISTRY = {
    "es":     ES,
    "random": RandomPolicy,
    "dqn":    DQN,
    "a3c":    A3C,
}
