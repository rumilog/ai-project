"""Utility functions for logging, plotting, and configuration."""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union
import torch
from stable_baselines3.common.logger import configure
from datetime import datetime


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logger(log_dir: str, exp_name: Optional[str] = None) -> None:
    """Configure the Stable Baselines logger.
    
    Args:
        log_dir: Directory to save logs
        exp_name: Optional experiment name
    """
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    log_path = os.path.join(log_dir, exp_name)
    os.makedirs(log_path, exist_ok=True)
    
    return configure(log_path, ["stdout", "csv", "tensorboard"])


def ensure_dir(path: str) -> None:
    """Ensure that a directory exists.
    
    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)


def plot_rewards(rewards: List[float], smoothing: int = 10, title: str = "Episode Rewards", 
                 save_path: Optional[str] = None) -> None:
    """Plot episode rewards with optional smoothing.
    
    Args:
        rewards: List of episode rewards
        smoothing: Window size for smoothing (moving average)
        title: Plot title
        save_path: If provided, save the plot to this path
    """
    plt.figure(figsize=(10, 6))
    
    # Raw rewards
    plt.plot(rewards, alpha=0.3, label="Raw rewards")
    
    # Smoothed rewards
    if smoothing > 1 and len(rewards) > smoothing:
        kernel = np.ones(smoothing) / smoothing
        smooth_rewards = np.convolve(rewards, kernel, mode='valid')
        plt.plot(range(smoothing-1, len(rewards)), smooth_rewards, label=f"Smoothed (window={smoothing})")
    
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()


def plot_training_progress(csv_path: str, save_path: Optional[str] = None) -> None:
    """Plot training progress from Stable Baselines CSV logs.
    
    Args:
        csv_path: Path to the CSV file containing training logs
        save_path: If provided, save the plot to this path
    """
    data = np.genfromtxt(csv_path, delimiter=',', names=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot rewards
    if 'reward' in data.dtype.names:
        ax1.plot(data['timesteps'], data['reward'], label='reward')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
    
    # Plot losses
    loss_cols = [name for name in data.dtype.names if 'loss' in name.lower()]
    for col in loss_cols:
        ax2.plot(data['timesteps'], data[col], label=col)
    
    ax2.set_xlabel('Timesteps')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 