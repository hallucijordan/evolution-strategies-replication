from .es import ES, ESConfig
from .random_search import RandomPolicy

REGISTRY = {
    "es": ES,
    "random": RandomPolicy,
}
