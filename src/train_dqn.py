"""Train a DQN agent on a gym environment."""

import os
import argparse
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

from src.envs import get_wrapped_env
from src.utils import load_config, setup_logger, ensure_dir, set_random_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a DQN agent on a gym environment")
    parser.add_argument("--config", type=str, default="configs/default.yaml", 
                        help="Path to config file")
    parser.add_argument("--model-name", type=str, default="dqn_cartpole",
                        help="Name for saving the model")
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create directories
    ensure_dir(config["train"]["log_dir"])
    ensure_dir(config["train"]["save_dir"])
    
    # Set random seed for reproducibility
    if "seed" in config["train"]:
        set_random_seed(config["train"]["seed"])
    
    # Setup logger
    logger = setup_logger(config["train"]["log_dir"], f"dqn_{config['env']['id']}")
    
    # Create environment
    env_id = config["env"]["id"]
    env = get_wrapped_env(env_id, seed=config["train"]["seed"])
    
    # Create DQN agent
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=config["dqn"]["learning_rate"],
        buffer_size=config["dqn"]["buffer_size"],
        learning_starts=config["dqn"]["learning_starts"],
        batch_size=config["dqn"]["batch_size"],
        gamma=config["dqn"]["gamma"],
        tau=config["dqn"]["tau"],
        target_update_interval=config["dqn"]["target_update_interval"],
        exploration_fraction=config["dqn"]["exploration_fraction"],
        exploration_initial_eps=config["dqn"]["exploration_initial_eps"],
        exploration_final_eps=config["dqn"]["exploration_final_eps"],
        tensorboard_log=config["train"]["log_dir"],
        verbose=1,
    )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=2000,
        save_path=config["train"]["save_dir"],
        name_prefix=args.model_name,
        verbose=1,
    )
    
    # Train the agent
    total_timesteps = config["train"]["total_timesteps"]
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        tb_log_name=args.model_name,
        log_interval=4,
    )
    
    # Save the final model
    model_path = os.path.join(config["train"]["save_dir"], f"{args.model_name}.zip")
    model.save(model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main() 