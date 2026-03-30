"""
Human play mode — control the Pong paddle with keyboard.

Controls:
    ↑  /  W   →  move UP
    ↓  /  S   →  move DOWN
    R         →  restart episode
    Q / ESC   →  quit

Usage:
    python play.py                          # Pong (default)
    python play.py --env BreakoutNoFrameskip-v4
    python play.py --env PongNoFrameskip-v4 --fps 30
"""

import argparse
import sys

import ale_py
import gymnasium as gym
import numpy as np
import pygame

gym.register_envs(ale_py)


# ---------------------------------------------------------------------------
# Keyboard → action mapping (Pong)
#
# Pong action meanings: NOOP(0) FIRE(1) RIGHT(2) LEFT(3) RIGHTFIRE(4) LEFTFIRE(5)
# In Pong from the agent's perspective: RIGHT = up, LEFT = down
# ---------------------------------------------------------------------------

PONG_KEYMAP = {
    pygame.K_UP:    2,   # RIGHT = paddle up
    pygame.K_w:     2,
    pygame.K_DOWN:  3,   # LEFT  = paddle down
    pygame.K_s:     3,
}

BREAKOUT_KEYMAP = {
    pygame.K_RIGHT: 2,
    pygame.K_d:     2,
    pygame.K_LEFT:  3,
    pygame.K_a:     3,
    pygame.K_SPACE: 1,   # FIRE to launch ball
}

ENV_KEYMAPS = {
    "Pong":     PONG_KEYMAP,
    "Breakout": BREAKOUT_KEYMAP,
}


def get_keymap(env_id: str) -> dict:
    for name, km in ENV_KEYMAPS.items():
        if name in env_id:
            return km
    return {}   # NOOP for unknown envs


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="PongNoFrameskip-v4")
    p.add_argument("--fps", type=int, default=24,
                   help="Rendering speed (default 24 — roughly real-time for frame-skip=4)")
    p.add_argument("--frame_skip", type=int, default=4,
                   help="Repeat each action for this many frames")
    return p.parse_args()


# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    keymap = get_keymap(args.env)

    env = gym.make(args.env, render_mode="rgb_array")
    action_meanings = env.unwrapped.get_action_meanings()
    n_actions = env.action_space.n

    print(f"\nEnvironment  : {args.env}")
    print(f"Actions      : {action_meanings}")
    print(f"Controls     : ↑/W = UP   ↓/S = DOWN   R = restart   Q/ESC = quit\n")

    pygame.init()
    obs_raw, _ = env.reset()
    frame = env.render()
    h, w = frame.shape[:2]
    scale = max(1, 600 // h)   # scale up small Atari frames
    screen = pygame.display.set_mode((w * scale, h * scale))
    pygame.display.set_caption(f"Playing {args.env}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 18)

    total_reward = 0.0
    episode = 1
    running = True

    def render_frame(frame_rgb, reward, ep):
        surf = pygame.surfarray.make_surface(frame_rgb.transpose(1, 0, 2))
        surf = pygame.transform.scale(surf, (w * scale, h * scale))
        screen.blit(surf, (0, 0))
        label = font.render(
            f"Episode {ep}   Score: {reward:+.0f}   Q=quit  R=restart",
            True, (255, 255, 0)
        )
        screen.blit(label, (8, 8))
        pygame.display.flip()

    while running:
        # --- Collect keyboard input ---
        action = 0   # NOOP by default
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                if event.key == pygame.K_r:
                    obs_raw, _ = env.reset()
                    total_reward = 0.0
                    episode += 1

        keys = pygame.key.get_pressed()
        for key, act in keymap.items():
            if keys[key]:
                action = act
                break

        # --- Step environment (repeat action for frame_skip frames) ---
        ep_done = False
        step_reward = 0.0
        for _ in range(args.frame_skip):
            obs_raw, reward, terminated, truncated, _ = env.step(action)
            step_reward += reward
            if terminated or truncated:
                ep_done = True
                break

        total_reward += step_reward

        # --- Render ---
        frame = env.render()
        render_frame(frame, total_reward, episode)

        if ep_done:
            print(f"Episode {episode:3d} finished — total score: {total_reward:+.0f}")
            obs_raw, _ = env.reset()
            total_reward = 0.0
            episode += 1

        clock.tick(args.fps)

    env.close()
    pygame.quit()
    print("Bye!")


if __name__ == "__main__":
    main()
