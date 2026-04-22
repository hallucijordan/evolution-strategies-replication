"""
Training entry point.

Usage examples:
    python train.py --algo es --env PongNoFrameskip-v4
    python train.py --algo es --env PongNoFrameskip-v4 --config configs/es_pong.yaml
    python train.py --algo random --env BreakoutNoFrameskip-v4 --total_steps 500000
    python train.py --algo es --env PongNoFrameskip-v4 --n_workers 8
"""

import argparse
from pathlib import Path

import yaml

from algorithms import REGISTRY
from algorithms.es import ES, ESConfig
from algorithms.dqn import DQN, DQNConfig
from algorithms.a3c import A3C, A3CConfig
from algorithms.ppo import PPO, PPOConfig
from envs import make_atari_env, EnvFactory
from models import AtariCNN, MLP, ActorCriticCNN, ActorCriticMLP, count_params
from utils.logger import Logger

ATARI_ENVS = {"NoFrameskip", "ALE/"}   # keywords that identify Atari envs


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train an algorithm on Atari")
    p.add_argument("--algo",        choices=list(REGISTRY.keys()), default="es")
    p.add_argument("--env",         default="PongNoFrameskip-v4")
    p.add_argument("--config",      default=None,        help="Path to YAML config file")
    p.add_argument("--total_steps", type=int, default=1_000_000)
    p.add_argument("--n_workers",   type=int, default=1, help="Parallel rollout workers (ES only)")
    p.add_argument("--run_name",    default=None,        help="Override the auto-generated run name")
    p.add_argument("--results_dir", default="results")
    return p.parse_args()


# ---------------------------------------------------------------------------

def is_atari(env_id: str) -> bool:
    return any(k in env_id for k in ATARI_ENVS)


def build_dqn(env_id: str, config_path: str | None) -> DQN:
    overrides: dict = {}
    if config_path:
        with open(config_path) as f:
            overrides = yaml.safe_load(f).get("algorithm", {})

    cfg = DQNConfig(**overrides)

    if is_atari(env_id):
        tmp_env = make_atari_env(env_id)
        policy = AtariCNN(tmp_env.action_space.n, use_vbn=False)
        tmp_env.close()
        print(f"Policy: AtariCNN (no VBN)  |  parameters: {count_params(policy):,}")
    else:
        import gymnasium as gym
        tmp_env = gym.make(env_id)
        obs_dim = tmp_env.observation_space.shape[0]
        policy = MLP(obs_dim, tmp_env.action_space.n)
        tmp_env.close()
        print(f"Policy: MLP  |  parameters: {count_params(policy):,}")

    return DQN(policy, cfg)


def build_a3c(env_id: str, config_path: str | None, n_workers: int) -> A3C:
    overrides: dict = {}
    if config_path:
        with open(config_path) as f:
            overrides = yaml.safe_load(f).get("algorithm", {})
    if n_workers > 1:
        overrides["n_workers"] = n_workers

    cfg = A3CConfig(**overrides)

    if is_atari(env_id):
        tmp_env = make_atari_env(env_id)
        policy = ActorCriticCNN(tmp_env.action_space.n)
        tmp_env.close()
        print(f"Policy: ActorCriticCNN  |  parameters: {count_params(policy):,}")
    else:
        import gymnasium as gym
        tmp_env = gym.make(env_id)
        obs_dim = tmp_env.observation_space.shape[0]
        policy = ActorCriticMLP(obs_dim, tmp_env.action_space.n)
        tmp_env.close()
        print(f"Policy: ActorCriticMLP  |  parameters: {count_params(policy):,}")

    return A3C(policy, cfg)


def build_ppo(env_id: str, config_path: str | None) -> PPO:
    overrides: dict = {}
    if config_path:
        with open(config_path) as f:
            overrides = yaml.safe_load(f).get("algorithm", {})

    cfg = PPOConfig(**overrides)

    if is_atari(env_id):
        tmp_env = make_atari_env(env_id)
        policy = ActorCriticCNN(tmp_env.action_space.n)
        tmp_env.close()
        print(f"Policy: ActorCriticCNN  |  parameters: {count_params(policy):,}")
    else:
        import gymnasium as gym
        tmp_env = gym.make(env_id)
        obs_dim = tmp_env.observation_space.shape[0]
        policy = ActorCriticMLP(obs_dim, tmp_env.action_space.n)
        tmp_env.close()
        print(f"Policy: ActorCriticMLP  |  parameters: {count_params(policy):,}")

    return PPO(policy, cfg)


def build_es(env_id: str, config_path: str | None, n_workers: int) -> ES:
    overrides: dict = {}
    if config_path:
        with open(config_path) as f:
            overrides = yaml.safe_load(f).get("algorithm", {})
    if n_workers > 1:
        overrides["n_workers"] = n_workers

    cfg = ESConfig(**overrides)

    if is_atari(env_id):
        tmp_env = make_atari_env(env_id)
        policy = AtariCNN(tmp_env.action_space.n)
        tmp_env.close()
        print(f"Policy: AtariCNN  |  parameters: {count_params(policy):,}")
    else:
        import gymnasium as gym
        tmp_env = gym.make(env_id)
        obs_dim = tmp_env.observation_space.shape[0]
        policy = MLP(obs_dim, tmp_env.action_space.n)
        tmp_env.close()
        print(f"Policy: MLP  |  parameters: {count_params(policy):,}")

    return ES(policy, cfg)


# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    run_name = args.run_name or f"{args.algo}_{args.env}"
    logger   = Logger(args.results_dir, run_name)

    if is_atari(args.env):
        env_fn = EnvFactory(args.env)
    else:
        import gymnasium as gym
        env_id = args.env
        env_fn = lambda: gym.make(env_id)

    # total_steps: yaml training.total_steps > CLI --total_steps
    total_steps = args.total_steps
    if args.config:
        with open(args.config) as f:
            total_steps = yaml.safe_load(f).get("training", {}).get(
                "total_steps", total_steps
            )

    # Build algorithm
    if args.algo == "es":
        agent = build_es(args.env, args.config, args.n_workers)
    elif args.algo == "dqn":
        agent = build_dqn(args.env, args.config)
    elif args.algo == "a3c":
        agent = build_a3c(args.env, args.config, args.n_workers)
    elif args.algo == "ppo":
        agent = build_ppo(args.env, args.config)
    else:
        AlgoCls = REGISTRY[args.algo]
        agent = AlgoCls()

    print(f"\nAlgorithm : {args.algo}")
    print(f"Env       : {args.env}")
    print(f"Budget    : {total_steps:,} steps")
    print(f"Run dir   : {Path(args.results_dir) / run_name}\n")

    agent.train(env_fn, total_steps, logger)

    # Save checkpoint
    ckpt_path = Path(args.results_dir) / run_name / "checkpoint"
    agent.save(str(ckpt_path))
    print(f"\nCheckpoint saved → {ckpt_path}.npy")


if __name__ == "__main__":
    main()
