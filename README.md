# RL for Gaming MVP

A reinforcement learning project for training agents in gaming environments.

## Project Overview

This project provides a framework for training and evaluating reinforcement learning agents in gaming environments. Currently, it includes:

- DQN (Deep Q-Network) implementation using Stable Baselines3
- PPO (Proximal Policy Optimization) placeholder
- Environment wrappers for OpenAI Gym
- Utilities for logging, plotting, and configuration
- Evaluation scripts for trained models

## Project Structure

```
rl-gaming-mvp/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── src/
│   ├── __init__.py
│   ├── envs.py
│   ├── train_dqn.py
│   ├── train_ppo.py
│   ├── eval.py
│   └── utils.py
├── notebooks/
│   └── exploration.ipynb
├── models/
└── logs/
```

## Setup Instructions

1. Clone the repository

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Examples

### Training a DQN agent on CartPole

```bash
python src/train_dqn.py --config configs/default.yaml
```

### Evaluating a trained model

```bash
python src/eval.py --model models/dqn_cartpole.zip
```

### Training a PPO agent

```bash
python src/train_ppo.py --config configs/default.yaml
```

## Configuration

The `configs/default.yaml` file contains hyperparameters for training. You can modify this file or create new configuration files.

## Extension Points

- Add new environments in `src/envs.py`
- Implement additional algorithms beyond DQN and PPO
- Create custom neural network architectures
- Add support for multi-agent environments

## License

This project is licensed under the MIT License - see the LICENSE file for details. 