"""
Standard Atari preprocessing wrappers.
Follows the setup from Mnih et al. [2015/2016] and Salimans et al. [2017]:
  - Random no-ops on reset
  - Frame-skip=4 with max-pooling over last 2 frames
  - Episodic life (life loss = episode end during training)
  - FIRE on reset for games that require it
  - Resize to 84x84 grayscale
  - Stack 4 frames → (4, 84, 84) uint8
  - Clip rewards to {-1, 0, +1} during training
"""

import numpy as np
import gymnasium as gym
import ale_py
from gymnasium import spaces

# gymnasium 1.x requires manual registration of ALE environments
gym.register_envs(ale_py)


# ---------------------------------------------------------------------------
# Individual wrappers
# ---------------------------------------------------------------------------

class NoopResetEnv(gym.Wrapper):
    """Take 1..noop_max random no-ops at the start of each episode."""

    def __init__(self, env: gym.Env, noop_max: int = 30):
        super().__init__(env)
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        n_noops = self.np_random.integers(1, self.noop_max + 1)
        for _ in range(n_noops):
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """Repeat action for `skip` frames; return max over the last 2 frames."""

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = np.zeros(
            (2,) + env.observation_space.shape, dtype=np.uint8
        )

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info


class EpisodicLifeEnv(gym.Wrapper):
    """
    Treat life loss as episode termination (training only).
    The real episode ends only when all lives are gone.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.lives = 0
        self._real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self._real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # Step with no-op to advance past the life-loss screen
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class FireResetEnv(gym.Wrapper):
    """Press FIRE at the start of each episode (required by some games)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        meanings = env.unwrapped.get_action_meanings()
        assert meanings[1] == "FIRE", f"Expected FIRE at index 1, got {meanings[1]}"
        assert len(meanings) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)  # FIRE
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
            return obs, info
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class WarpFrame(gym.ObservationWrapper):
    """Resize RGB frame to (84, 84, 1) grayscale uint8."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        import cv2
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized[:, :, np.newaxis]


class ClipRewardEnv(gym.RewardWrapper):
    """Clip rewards to {-1, 0, +1} (used during training, not evaluation)."""

    def reward(self, reward: float) -> float:
        return float(np.sign(reward))


class ChannelFirstFrameStack(gym.ObservationWrapper):
    """
    Wraps gymnasium.wrappers.FrameStack output (N, H, W, 1) → (N, H, W) uint8.
    Must be applied after FrameStack.
    """

    def __init__(self, env: gym.Env, n_stack: int):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(n_stack, 84, 84), dtype=np.uint8
        )

    def observation(self, obs) -> np.ndarray:
        # LazyFrames or ndarray: (N, 84, 84, 1) → squeeze → (N, 84, 84)
        arr = np.array(obs, dtype=np.uint8)
        if arr.ndim == 4:          # (N, H, W, C) with C=1
            arr = arr.squeeze(-1)
        return arr


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_atari_env(
    env_id: str,
    frame_stack: int = 4,
    clip_reward: bool = True,
    episodic_life: bool = True,
    render_mode: str = None,
) -> gym.Env:
    """
    Build a fully preprocessed Atari environment.

    Output observations: (frame_stack, 84, 84) uint8 numpy arrays.
    """
    env = gym.make(env_id, render_mode=render_mode)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if episodic_life:
        env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if clip_reward:
        env = ClipRewardEnv(env)
    env = gym.wrappers.FrameStackObservation(env, frame_stack)
    env = ChannelFirstFrameStack(env, frame_stack)
    return env


class EnvFactory:
    """
    Picklable env factory — safe to pass to multiprocessing workers.
    A plain local function (closure) cannot be pickled; a class instance can.
    """
    def __init__(self, env_id: str, **kwargs):
        self.env_id = env_id
        self.kwargs = kwargs

    def __call__(self):
        return make_atari_env(self.env_id, **self.kwargs)
