"""Evaluate a trained RL model."""

import os
import argparse
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import gym
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from src.envs import get_wrapped_env
from src.utils import load_config, plot_rewards, ensure_dir


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained RL model")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", 
                        help="Path to config file")
    parser.add_argument("--n-episodes", type=int, default=None, 
                        help="Number of episodes to evaluate (default: from config)")
    parser.add_argument("--deterministic", action="store_true", 
                        help="Use deterministic actions for evaluation")
    parser.add_argument("--render", action="store_true", help="Render environment")
    parser.add_argument("--save-video", action="store_true", help="Save video of evaluation")
    parser.add_argument("--output-dir", type=str, default="./results", 
                        help="Directory to save results")
    return parser.parse_args()


def load_model(model_path: str) -> Tuple[BaseAlgorithm, str]:
    """Load a trained model from path.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Tuple of (loaded model, algorithm name)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Try loading as different algorithms
    algorithms = {
        "DQN": DQN,
        "PPO": PPO,
    }
    
    # Detect algorithm from filename if possible
    algorithm_name = None
    for name in algorithms.keys():
        if name.lower() in model_path.lower():
            algorithm_name = name
            break
    
    # Try the detected algorithm first, then others
    errors = []
    if algorithm_name:
        try:
            return algorithms[algorithm_name].load(model_path), algorithm_name
        except Exception as e:
            errors.append(f"{algorithm_name}: {str(e)}")
    
    # Try all algorithms
    for name, algo_class in algorithms.items():
        if name != algorithm_name:  # Skip if already tried
            try:
                return algo_class.load(model_path), name
            except Exception as e:
                errors.append(f"{name}: {str(e)}")
    
    # If we get here, no algorithm worked
    raise ValueError(f"Failed to load model as any known algorithm: {model_path}\nErrors: {errors}")


def evaluate_model(model: BaseAlgorithm, env: gym.Env, n_episodes: int, deterministic: bool = True) -> List[float]:
    """Evaluate a model on an environment for multiple episodes.
    
    Args:
        model: Trained model to evaluate
        env: Environment to evaluate on
        n_episodes: Number of episodes to run
        deterministic: Whether to use deterministic actions
        
    Returns:
        List of episode rewards
    """
    episode_rewards = []
    
    for i in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            
        episode_rewards.append(episode_reward)
        print(f"Episode {i+1}/{n_episodes}: Reward = {episode_reward}")
    
    return episode_rewards


def visualize_episode(model: BaseAlgorithm, env: gym.Env, deterministic: bool = True, 
                      save_path: Optional[str] = None) -> Tuple[List[np.ndarray], float]:
    """Run a single episode and record the frames.
    
    Args:
        model: Trained model to visualize
        env: Environment to run on
        deterministic: Whether to use deterministic actions
        save_path: If provided, save the rendered frames as a video
        
    Returns:
        Tuple of (frames, episode reward)
    """
    try:
        env = gym.wrappers.RecordVideo(env, save_path or ".", name_prefix="rl-episode")
    except:
        print("Warning: Could not wrap environment with RecordVideo")
    
    obs, _ = env.reset()
    done = False
    truncated = False
    episode_reward = 0
    frames = []
    
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, truncated, _ = env.step(action)
        episode_reward += reward
        
        if hasattr(env, "render"):
            try:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            except:
                pass
    
    return frames, episode_reward


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    ensure_dir(args.output_dir)
    
    # Create environment
    env_id = config["env"]["id"]
    env = get_wrapped_env(env_id)
    
    # Load model
    model, algo_name = load_model(args.model)
    print(f"Loaded {algo_name} model from {args.model}")
    
    # Get number of evaluation episodes
    n_episodes = args.n_episodes or config["eval"]["n_episodes"]
    deterministic = args.deterministic or config["eval"]["deterministic"]
    
    # Evaluate model
    rewards = evaluate_model(model, env, n_episodes, deterministic)
    
    # Print statistics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)
    
    print(f"\nEvaluation results over {n_episodes} episodes:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Min reward: {min_reward:.2f}")
    print(f"Max reward: {max_reward:.2f}")
    
    # Plot rewards
    plot_path = os.path.join(args.output_dir, "evaluation_rewards.png")
    plot_rewards(rewards, title=f"{algo_name} Evaluation Rewards", save_path=plot_path)
    
    # Visualize an episode if requested
    if args.render or args.save_video:
        video_path = args.output_dir if args.save_video else None
        frames, reward = visualize_episode(model, env, deterministic, video_path)
        print(f"Visualized episode reward: {reward:.2f}")


if __name__ == "__main__":
    main() 