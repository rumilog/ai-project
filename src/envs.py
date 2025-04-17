"""Environment wrappers for OpenAI Gym environments."""

import gymnasium as gym
from typing import Optional, Dict, Any, Tuple
import numpy as np


def make_env(env_id: str, seed: Optional[int] = None, **kwargs) -> gym.Env:
    """Create and configure a gym environment.
    
    Args:
        env_id: The ID of the environment to create
        seed: Random seed for reproducibility
        **kwargs: Additional arguments to pass to the environment
        
    Returns:
        The initialized gym environment
    """
    env = gym.make(env_id, **kwargs)
    
    if seed is not None:
        env.reset(seed=seed)
        
    return env


class FrameStackWrapper(gym.Wrapper):
    """Stack multiple frames as observation for frame-based environments."""
    
    def __init__(self, env: gym.Env, n_frames: int = 4):
        """Initialize the wrapper.
        
        Args:
            env: The environment to wrap
            n_frames: Number of frames to stack
        """
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = []
        
        # Update observation space
        low = np.repeat(self.observation_space.low[np.newaxis, ...], n_frames, axis=0)
        high = np.repeat(self.observation_space.high[np.newaxis, ...], n_frames, axis=0)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=self.observation_space.dtype)
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset the environment and stack initial frames.
        
        Args:
            **kwargs: Additional arguments to pass to the environment
            
        Returns:
            Tuple of (stacked observation, info)
        """
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs] * self.n_frames
        return self._get_observation(), info
    
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment and update the frame stack.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (stacked observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.pop(0)
        self.frames.append(obs)
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Stack the frames into a single observation.
        
        Returns:
            Stacked frames as a single observation
        """
        return np.array(self.frames)


def get_wrapped_env(env_id: str, frame_stack: Optional[int] = None, seed: Optional[int] = None, **kwargs) -> gym.Env:
    """Create an environment with optional wrappers.
    
    Args:
        env_id: The ID of the environment to create
        frame_stack: If provided, number of frames to stack
        seed: Random seed for reproducibility
        **kwargs: Additional arguments to pass to the environment
        
    Returns:
        The wrapped environment
    """
    env = make_env(env_id, seed, **kwargs)
    
    if frame_stack is not None and frame_stack > 1:
        env = FrameStackWrapper(env, n_frames=frame_stack)
        
    return env 