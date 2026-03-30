"""
Visualize Atari preprocessing pipeline.

Shows 8 consecutive frames across two 4-frame stacks:
  Row 1: Raw RGB frames
  Row 2: After grayscale + 84x84 resize (WarpFrame output)
  Row 3: Final stacked input to CNN — stack 1 (frames 1-4)
  Row 4: Final stacked input to CNN — stack 2 (frames 5-8)

Usage:
    python visualize_preprocessing.py
    python visualize_preprocessing.py --env BreakoutNoFrameskip-v4
    python visualize_preprocessing.py --env PongNoFrameskip-v4 --steps 80 --out preprocessing.png
"""

import argparse

import ale_py
import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from envs.atari_wrappers import (
    NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv,
    WarpFrame, ClipRewardEnv, ChannelFirstFrameStack,
)

gym.register_envs(ale_py)


# ---------------------------------------------------------------------------
# Build two partial pipelines so we can intercept at each stage
# ---------------------------------------------------------------------------

def build_raw_env(env_id: str):
    """RGB env with only frame-skip — gives us raw frames at action frequency."""
    env = gym.make(env_id, render_mode="rgb_array")
    env = NoopResetEnv(env, noop_max=5)
    env = MaxAndSkipEnv(env, skip=4)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    return env


def build_warped_env(env_id: str):
    """Up to WarpFrame — grayscale 84x84, no stacking yet."""
    env = gym.make(env_id, render_mode="rgb_array")
    env = NoopResetEnv(env, noop_max=5)
    env = MaxAndSkipEnv(env, skip=4)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    return env


def build_full_env(env_id: str):
    """Full pipeline including frame stack — what the CNN actually sees."""
    from envs.atari_wrappers import make_atari_env
    return make_atari_env(env_id, clip_reward=False, episodic_life=False)


# ---------------------------------------------------------------------------

def warp_frame(rgb: np.ndarray) -> np.ndarray:
    """Manually apply the same grayscale + resize as WarpFrame. (210,160,3) -> (84,84)"""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)


def collect_frames(env_id: str, n_steps: int = 60):
    """
    Run ONE full-pipeline env. At each step:
      - save raw RGB via render()         → Row 1
      - manually warp that RGB frame      → Row 2  (perfectly in sync with Row 1)
      - save the stacked CNN obs          → Row 3/4

    Using a single env eliminates sync issues and ensures the ball
    is always in the same position across all three representations.
    """
    env = build_full_env(env_id)
    # underlying raw-RGB env for render()
    raw_env = gym.make(env_id, render_mode="rgb_array")

    env.reset(seed=42)
    raw_env.reset(seed=42)

    # warm up — alternate FIRE and move to keep ball in play
    for i in range(n_steps):
        action = 1 if i % 20 == 0 else 2   # press FIRE every 20 steps to respawn ball
        env.step(action)
        raw_env.step(action)

    raw_frames    = []
    warped_frames = []
    stacked_obs   = []

    for i in range(8):
        action = 2
        full_obs, _, terminated, truncated, _ = env.step(action)
        raw_rgb = raw_env.render()
        raw_env.step(action)

        # if episode ended mid-collection, reset both and press FIRE
        if terminated or truncated:
            env.reset()
            raw_env.reset()
            for _ in range(3):
                env.step(1)
                raw_env.step(1)

        raw_frames.append(raw_rgb)
        warped_frames.append(warp_frame(raw_rgb))          # derived from same frame

        if i in (3, 7):
            stacked_obs.append(np.array(full_obs))         # (4, 84, 84)

    env.close()
    raw_env.close()

    return raw_frames, warped_frames, stacked_obs


# ---------------------------------------------------------------------------

def plot(raw_frames, warped_frames, stacked_obs, env_id: str, out: str):
    n = 8
    fig, axes = plt.subplots(4, n, figsize=(n * 2.2, 4 * 2.2))
    fig.suptitle(
        f"Atari Preprocessing Pipeline — {env_id}",
        fontsize=14, fontweight="bold", y=1.01
    )

    row_labels = [
        "Raw RGB\n(original)",
        "Grayscale + 84×84\n(WarpFrame)",
        "CNN input — stack 1\n(frames 1–4)",
        "CNN input — stack 2\n(frames 5–8)",
    ]

    # Row 0: raw RGB
    for i in range(n):
        ax = axes[0, i]
        ax.imshow(raw_frames[i])
        ax.set_title(f"frame {i+1}", fontsize=8)
        ax.axis("off")

    # Row 1: warped grayscale
    for i in range(n):
        ax = axes[1, i]
        ax.imshow(warped_frames[i], cmap="gray", vmin=0, vmax=255)
        ax.axis("off")

    # Rows 2–3: stacked CNN inputs (4 frames each)
    for stack_idx, stack in enumerate(stacked_obs):   # stack: (4, 84, 84)
        for ch in range(4):
            ax = axes[2 + stack_idx, ch * 2]
            ax.imshow(stack[ch], cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"ch {ch+1}", fontsize=8)
            ax.axis("off")
            # leave the odd columns blank for visual spacing
            axes[2 + stack_idx, ch * 2 + 1].axis("off")

    # Row labels on the left
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=9, labelpad=6)
        axes[row, 0].yaxis.set_label_coords(-0.15, 0.5)

    # Highlight the two stacks with a colored border
    colors = ["#2196F3", "#4CAF50"]
    for stack_idx, color in enumerate(colors):
        for ch in range(4):
            ax = axes[2 + stack_idx, ch * 2]
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.5)
                spine.set_visible(True)

    patches = [
        mpatches.Patch(color=colors[0], label="Stack 1 — frames 1–4 (CNN input at t=4)"),
        mpatches.Patch(color=colors[1], label="Stack 2 — frames 5–8 (CNN input at t=8)"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.show()


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env",   default="PongNoFrameskip-v4")
    p.add_argument("--steps", type=int, default=60,
                   help="Warm-up steps before capturing (more = later in episode)")
    p.add_argument("--out",   default="results/preprocessing_visualization.png")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Collecting frames from {args.env} ...")
    raw, warped, stacked = collect_frames(args.env, args.steps)
    print(f"  raw frames    : {len(raw)} × {raw[0].shape}")
    print(f"  warped frames : {len(warped)} × {warped[0].shape}")
    print(f"  stacked obs   : {len(stacked)} × {stacked[0].shape}")
    plot(raw, warped, stacked, args.env, args.out)


if __name__ == "__main__":
    main()
