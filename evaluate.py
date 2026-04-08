"""
Evaluation entry point — load a saved checkpoint and measure performance.

Usage:
    python evaluate.py --env PongNoFrameskip-v4 --checkpoint results/es_PongNoFrameskip-v4/checkpoint.npy
    python evaluate.py --env PongNoFrameskip-v4 --checkpoint results/es_PongNoFrameskip-v4/checkpoint.npy --render
    python evaluate.py --env PongNoFrameskip-v4 --checkpoint results/es_PongNoFrameskip-v4/checkpoint.npy --n_episodes 20
"""

import argparse

import numpy as np

from algorithms.es import ES, ESConfig
from algorithms.dqn import DQN, DQNConfig
from algorithms.a3c import A3C, A3CConfig
from envs import make_atari_env
from models import AtariCNN, ActorCriticCNN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env",        required=True)
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint (.npy for ES, .pt for DQN/A3C)")
    p.add_argument("--algo",       choices=["es", "dqn", "a3c"], default="es")
    p.add_argument("--n_episodes", type=int, default=10)
    p.add_argument("--render",     action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    render_mode = "human" if args.render else None
    env = make_atari_env(
        args.env,
        clip_reward=False,      # don't clip during evaluation
        episodic_life=False,    # use real episode boundaries
        render_mode=render_mode,
    )

    if args.algo == "es":
        policy = AtariCNN(env.action_space.n)
        agent  = ES(policy, ESConfig())
    elif args.algo == "dqn":
        policy = AtariCNN(env.action_space.n, use_vbn=False)
        agent  = DQN(policy, DQNConfig())
    elif args.algo == "a3c":
        policy = ActorCriticCNN(env.action_space.n)
        agent  = A3C(policy, A3CConfig())

    agent.load(args.checkpoint)

    print(f"Evaluating checkpoint: {args.checkpoint}")
    print(f"Environment          : {args.env}")
    print(f"Episodes             : {args.n_episodes}\n")

    returns = []
    for ep in range(1, args.n_episodes + 1):
        obs, _ = env.reset()
        ep_return = 0.0
        done = False
        while not done:
            action = agent.policy.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            done = terminated or truncated
        returns.append(ep_return)
        print(f"  Episode {ep:3d}: {ep_return:8.1f}")

    print(f"\nMean ± Std : {np.mean(returns):.1f} ± {np.std(returns):.1f}")
    print(f"Min  / Max : {np.min(returns):.1f} / {np.max(returns):.1f}")
    env.close()


if __name__ == "__main__":
    main()
