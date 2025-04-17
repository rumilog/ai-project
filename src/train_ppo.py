"""Train a PPO agent on a gym environment."""

import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from src.envs import get_wrapped_env
from src.utils import load_config, setup_logger, ensure_dir, set_random_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a PPO agent on a gym environment")
    parser.add_argument("--config", type=str, default="configs/default.yaml", 
                        help="Path to config file")
    parser.add_argument("--model-name", type=str, default="ppo_cartpole",
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
    logger = setup_logger(config["train"]["log_dir"], f"ppo_{config['env']['id']}")
    
    # Create environment
    env_id = config["env"]["id"]
    env = get_wrapped_env(env_id, seed=config["train"]["seed"])
    
    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config["ppo"]["learning_rate"],
        n_steps=config["ppo"]["n_steps"],
        batch_size=config["ppo"]["batch_size"],
        n_epochs=config["ppo"]["n_epochs"],
        gamma=config["ppo"]["gamma"],
        gae_lambda=config["ppo"]["gae_lambda"],
        clip_range=config["ppo"]["clip_range"],
        ent_coef=config["ppo"]["ent_coef"],
        vf_coef=config["ppo"]["vf_coef"],
        max_grad_norm=config["ppo"]["max_grad_norm"],
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
    )
    
    # Save the final model
    model_path = os.path.join(config["train"]["save_dir"], f"{args.model_name}.zip")
    model.save(model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main() 