# Self-Landing Rocket Reinforcement Learning

A custom reinforcement learning environment for training autonomous rocket landing agents. Built with Gymnasium, PyGame, and Ray RLlib, this project demonstrates how deep RL algorithms can learn stable control policies in a physics-based landing task.

---

## Table of Contents

- [Self-Landing Rocket Reinforcement Learning](#self-landing-rocket-reinforcement-learning)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Reinforcement Learning Concepts](#reinforcement-learning-concepts)
    - [What is Reinforcement Learning?](#what-is-reinforcement-learning)
    - [Key RL Components](#key-rl-components)
    - [Core Setup](#core-setup)
    - [Algorithms Used](#algorithms-used)
        - [**DQN (Deep Q-Network)**](#dqn-deep-q-network)
        - [**PPO (Proximal Policy Optimization)**](#ppo-proximal-policy-optimization)
  - [Environment Details](#environment-details)
    - [State Space (7 Continuous Variables)](#state-space-7-continuous-variables)
    - [Action Space (Discrete)](#action-space-discrete)
    - [Physics Model](#physics-model)
  - [Reward Function](#reward-function)
    - [Per-Step Components](#per-step-components)
    - [Terminal Rewards](#terminal-rewards)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
    - [Requirements](#requirements)
    - [Setup](#setup)
  - [Training Agents](#training-agents)
    - [Train DQN](#train-dqn)
    - [Train PPO](#train-ppo)
  - [Algorithm Configuration Summaryc](#algorithm-configuration-summaryc)
  - [Customization](#customization)
  - [Results \& Visualization](#results--visualization)
  - [Acknowledgments](#acknowledgments)

---

## Overview

This project formulates rocket landing as a sequential decision-making problem. The agent must learn to:

- Stabilize orientation during descent  
- Control vertical and horizontal velocity  
- Remain centered over a landing pad  
- Execute a soft landing within strict tolerances  
- Manage limited fuel  

The environment uses continuous 2D rigid-body dynamics and a discrete thruster-based control scheme. Agents are trained using Ray RLlib implementations of DQN and PPO with distributed environment sampling.

---

## Reinforcement Learning Concepts

### What is Reinforcement Learning?

**Reinforcement Learning (RL)** is a machine learning paradigm where an agent learns to make decisions by interacting with an environment. The agent:

1. **Observes** the current state of the environment
2. **Takes actions** based on its policy (decision-making strategy)
3. **Receives rewards** (positive or negative) based on action quality
4. **Updates its policy** to maximize cumulative long-term reward

Unlike supervised learning, there are no labeled "correct" answers—the agent must discover effective strategies through trial and error.

### Key RL Components

- **State (s)**: The current situation of the rocket (position, velocity, angle, etc.)
- **Action (a)**: Control decisions (thrust power, rotation direction)
- **Reward (r)**: Feedback signal indicating action quality
- **Policy (π)**: The agent's strategy mapping states to actions
- **Value Function (V/Q)**: Expected future reward from a state or state-action pair

### Core Setup

At each timestep the agent:

1. Observes the current rocket state  
2. Selects one of four discrete actions  
3. Receives a shaped reward  
4. Updates its policy to maximize long-term return  

There are no demonstrations or scripted heuristics. The policy is learned entirely from reward feedback.

### Algorithms Used

##### **DQN (Deep Q-Network)**

- **Type**: Value-based, off-policy
- **Best for**: Discrete action spaces, sample-efficient learning
- **How it works**: Learns a Q-function that estimates expected future rewards for each action, then selects actions with highest Q-values
- **Advantages**: 
  - Experience replay improves sample efficiency
  - Stable convergence with target networks
  - Well-suited for environments with clear optimal actions
- **Use when**: You have discrete actions and want reliable, sample-efficient training
- **Model Setup**:
  - Experience replay (capacity = 50,000)  
  - Target network updates every 500 steps  
  - Double Q-learning enabled  
  - Dueling architecture enabled  
  - Learning rate = 1e-3  
  - Discount factor = 0.99  
  - Train batch size = 3000  
  - Parallel sampling = 6 env runners × 4 envs each  

##### **PPO (Proximal Policy Optimization)**

- **Type**: Policy-based, on-policy
- **Best for**: Continuous/discrete actions, stable training
- **How it works**: Directly optimizes the policy while constraining updates to prevent destructive large changes
- **Advantages**:
  - More stable than older policy gradient methods
  - Works well with continuous action spaces
  - Less sensitive to hyperparameters
- **Use when**: You need robust, stable training or plan to extend to continuous control
- **Model Setup**:
  - Learning rate = 1e-4  
  - Gamma = 0.99  
  - Train batch size = 6000  
  - Minibatch size = 128  
  - Num epochs = 10  
  - Clip parameter = 0.2  
  - Entropy coefficient = 0.01  
  - Value loss coefficient = 0.5  
  - Parallel sampling = 8 env runners × 4 envs each  

Both algorithms use Ray’s distributed execution model for scalable rollout collection.

---

## Environment Details

This section defines the learning problem itself: what the agent observes, what actions it can take, how the physics evolves over time, and when an episode ends. Together, these components specify the Markov Decision Process that the reinforcement learning algorithms optimize against.


### State Space (7 Continuous Variables)

The state space describes what the agent can observe at each timestep. It is represented as a 7-dimensional continuous vector:


```
[x, y, vx, vy, angle, angular_velocity, fuel]

```

| Variable | Description | Range |
|-----------|------------|--------|
| x | Horizontal position | [-1.5, 1.5] |
| y | Vertical position | [-1.95, 1.95] |
| vx | Horizontal velocity | [-2.0, 2.0] |
| vy | Vertical velocity | [-2.0, 2.0] |
| angle | Orientation in radians | [-π, π] |
| angular_velocity | Rotational velocity | [-2.0, 2.0] |
| fuel | Remaining fuel fraction | [0.0, 1.0] |

Initial state:
- x sampled uniformly from [-0.5, 0.5]  
- y sampled from [0.8, 1.2]  
- angle sampled from [-0.3, 0.3]  
- vx, vy, angular velocity initialized to 0  
- fuel initialized to 1.0  

These randomized initial conditions promote generalization and prevent the agent from overfitting to a single descent trajectory.

---

### Action Space (Discrete)

The action space defines the set of control inputs available to the agent. At each timestep, the agent selects one of four discrete actions that correspond to simplified thruster commands.

| Action | Description |
|--------|------------|
| 0 | No-op |
| 1 | Rotate left |
| 2 | Fire main engine |
| 3 | Rotate right |

Main engine thrust is applied along the rocket axis and consumes 0.01 fuel per activation.

This discrete structure simplifies the control problem while still requiring coordinated orientation and thrust decisions for a successful landing.


---

### Physics Model

The physics model governs how the rocket transitions from one state to the next after an action is applied. It defines the environment’s dynamics and determines how thrust, gravity, and rotation influence motion over time.

- Gravity = -0.05 per timestep  
- Main thrust acceleration = 0.20  
- Angular thrust = ±0.05  
- Angular damping multiplier = 0.99  
- Integration method = Euler  
- Time step dt = 0.05  

Episode termination conditions:
- Rocket hits the ground (y ≤ 0)  
- Rocket leaves horizontal bounds (|x| > 1.5)  

These termination rules define success and failure boundaries and ensure each episode eventually concludes.

---

## Reward Function

The reward function specifies the objective the agent optimizes during training. Instead of rewarding only final success, the environment uses dense reward shaping to guide learning throughout the descent, encouraging progressively more stable and controlled behavior.

### Per-Step Components

- Angle penalty = -0.15 × angle²  
- Angular velocity penalty = -0.05 × |angular_velocity|  
- Horizontal drift penalty = -0.3 × |x|  
- Velocity penalty = -0.2 × (vx² + vy²)  
- Descent encouragement = +0.1 × (1.0 - y)  
- Survival penalty = -0.05 per step  

Near-ground shaping (if y < 0.3):
- Bonus for slow vertical velocity  
- Bonus for low horizontal drift  
- Bonus for near-upright orientation  

These components help stabilize training by rewarding incremental improvements rather than only perfect landings.

### Terminal Rewards

Terminal rewards are applied when an episode ends and provide strong outcome-based signals. They distinguish successful landings from partial success or failure, reinforcing high-quality control strategies.

Soft landing (all conditions satisfied):
- |x| < 0.1  
- |vx| < 0.1  
- |vy| < 0.2  
- |angle| < 0.1  
Reward = +200  

Near landing:
- |x| < 0.3  
- |vx| < 0.3  
- |vy| < 0.4  
- |angle| < 0.3  
Reward = +75  

Crash:
Reward = -100  

Leaving bounds of the screens:
Reward = -50  

---

## Project Structure

```
Self-Landing-Rocket/
│
├── rocket_landing/
│   ├── environment/
│   │   └── rocket_env.py
│   │
│   └── training/
│       ├── train_rllib_dqn.py
│       ├── train_rllib_ppo.py
│       └── training_plots/
│
├── policies/
│   ├── rllib_dqn_best/
│   └── rllib_ppo_best/
│
├── gifs/
├── ray_results/
│
├── pyproject.toml
└── README.md
```

---

## Installation

This section walks through setting up the project locally. The environment is managed with Poetry to ensure consistent dependencies and reproducible training runs.

### Requirements

Before installing, make sure the following are available on your system:

- Python 3.11  
- Poetry  

Python 3.11 is recommended to ensure compatibility with Gymnasium and Ray RLlib. Poetry is used for dependency management and virtual environment isolation.

### Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/dcorc7/Self-Landing-Rocket.git
cd Self-Landing-Rocket

poetry install
poetry shell
```

- **git clone** downloads the project source code.
- **poetry install** installs all required dependencies listed in pyproject.toml.
- **poetry shell** activates the project’s virtual environment so all commands run in an isolated environment.

Verify installation:

```bash
poetry run python -c "import gymnasium; import ray; print(\"Setup successful\")"
```

This command confirms that Gymnasium and Ray are properly installed and importable.

---

## Training Agents

This project supports training with both DQN and PPO using Ray RLlib. Each training script initializes the environment, configures the algorithm, and runs distributed rollouts until a stopping condition is met.

Training logs, checkpoints, and plots are automatically generated.

### Train DQN

```bash
poetry run python -m rocket_landing.training.train_rllib_dqn
```

This script:

- Registers the custom Gymnasium environment
- Configures RLlib’s DQN algorithm
- Enables experience replay and target networks
- Launches parallel environment workers
- Periodically logs performance metrics

Training stops when:

- 150 iterations are reached, or
- Mean episode return ≥ 150

These stopping criteria prevent unnecessary overtraining once stable landing behavior has emerged.

Best checkpoint saved to:

```
policies/rllib_dqn_best/
```

This directory contains the highest-performing model weights discovered during training. These can be reloaded later for evaluation or visualization.

Training curves saved to:

```
rocket_landing/training/training_plots/dqn_training_curves.png
```

The generated plots typically include:

- Mean episode return over time
- Episode length trends

These curves help diagnose learning stability and convergence.

---

### Train PPO

```bash
poetry run python -m rocket_landing.training.train_rllib_ppo
```

This script configures RLlib’s PPO implementation, which differs from DQN in several key ways:

- Uses on-policy rollouts
- Optimizes a clipped surrogate objective
- Performs multiple epochs over collected batches

Training stops when:

- 200 iterations are reached, or
- Mean episode return ≥ 175

PPO generally requires larger batch sizes and more iterations due to its on-policy nature.

Best checkpoint saved to:

```
policies/rllib_ppo_best/
```

Training curves saved to:

```
rocket_landing/training/training_plots/ppo_training_curves.png
```

These plots provide a visual comparison to DQN and help evaluate stability and sample efficiency.

---

## Algorithm Configuration Summaryc

This table highlights structural differences between the two algorithms:

| Feature           | DQN              | PPO               |
| ----------------- | ---------------- | ----------------- |
| Type              | Off-policy       | On-policy         |
| Replay Buffer     | Yes              | No                |
| Parallel Sampling | 24 envs          | 32 envs           |
| Train Batch Size  | 3000             | 6000              |
| Stop Condition    | Return ≥ 150     | Return ≥ 175      |
| Architecture      | Double + Dueling | Clipped objective |

Key distinctions:

- DQN reuses past experience through replay, improving sample efficiency.
- PPO collects fresh trajectories each iteration, improving stability at the cost of more environment interaction.
- PPO typically produces smoother learning curves, while DQN can converge faster but may be more sensitive to hyperparameters.

---

## Customization

The project is modular and designed for experimentation. 

Modify physics parameters in:

```
rocket_landing/environment/rocket_env.py
```

Key adjustable parameters:

- gravity
- main_thrust
- angular_thrust
- fuel consumption rate
- landing tolerances
- reward weights

Changing these values alters the difficulty and structure of the control problem. For example:

- Increasing gravity makes descent harder.
- Increasing fuel consumption forces more efficient thrust usage.
- Adjusting reward weights changes what behaviors are emphasized during learning.

Modify training hyperparameters in:

```
rocket_landing/training/train_rllib_dqn.py
rocket_landing/training/train_rllib_ppo.py
```

Here you can adjust:

- Learning rate
- Batch sizes
- Discount factor
- Entropy regularization
- vNetwork architecture

This enables systematic experimentation and hyperparameter tuning.

---

## Results & Visualization

Training produces:

- vMean episode return curves
- Mean episode length curves
- Best-performing checkpoints

These outputs allow you to:

- Measure convergence speed
- Compare DQN vs PPO performance
- Reload trained policies for simulation

TensorBoard logs are stored in:

```
ray_results/
```

These logs include:

- Training loss
- Policy entropy
- Value loss
- Episode reward metrics

Launch TensorBoard:

```bash
poetry run tensorboard --logdir ray_results/
```

Open in browser:

```
http://localhost:6006
```

TensorBoard provides interactive visualizations that make it easier to monitor training dynamics and diagnose instability or divergence.

---

## Acknowledgments

* Gymnasium for the RL environment API
* Ray RLlib for scalable reinforcement learning
* PyGame for rendering and visualization
