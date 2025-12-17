# Self-Landing Rocket Reinforcement Learning

A reinforcement learning environment for training autonomous rocket landing agents. Built with Gymnasium, PyGame, and Ray RLlib, this project demonstrates how deep RL algorithms can master complex physics-based control tasks similar to SpaceX's autonomous landing systems.

---

## Table of Contents

- [Overview](#-overview)
- [Reinforcement Learning Concepts](#-reinforcement-learning-concepts)
- [Environment Details](#-environment-details)
- [Project Structure](#-project-structure)
- [Installation](#️-installation)
- [Training Agents](#-training-agents)
- [Algorithm Comparison](#-algorithm-comparison)
- [Customization](#-customization)
- [Results & Visualization](#-results--visualization)

---

## Overview

This project tackles the challenge of autonomous rocket landing using reinforcement learning. The agent must learn to:

- Control thrust to manage velocity and altitude
- Adjust orientation to remain upright
- Execute soft landings within a designated target zone
- Conserve fuel while maintaining stability

The environment provides realistic physics simulation with continuous state dynamics and discrete control actions, making it an ideal testbed for modern deep RL algorithms.

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

### Training Approaches in This Project

#### 1. **RLlib-Based Training**

**Ray RLlib** is an industry-standard, production-ready RL library that provides:

- **Scalable distributed training** across multiple CPUs/GPUs
- **Battle-tested implementations** of state-of-the-art algorithms
- **Automatic hyperparameter tuning** with Ray Tune integration
- **Robust checkpointing** and experiment tracking
- **Vectorized environments** for faster training

**Why use RLlib?**
- Pre-optimized algorithms with years of community refinement
- Handles complex training infrastructure automatically
- Production deployment capabilities
- Active maintenance and bug fixes
- Extensive documentation and community support

**Supported RLlib Algorithms:**

##### **DQN (Deep Q-Network)**
- **Type**: Value-based, off-policy
- **Best for**: Discrete action spaces, sample-efficient learning
- **How it works**: Learns a Q-function that estimates expected future rewards for each action, then selects actions with highest Q-values
- **Advantages**: 
  - Experience replay improves sample efficiency
  - Stable convergence with target networks
  - Well-suited for environments with clear optimal actions
- **Use when**: You have discrete actions and want reliable, sample-efficient training

##### **PPO (Proximal Policy Optimization)**
- **Type**: Policy-based, on-policy
- **Best for**: Continuous/discrete actions, stable training
- **How it works**: Directly optimizes the policy while constraining updates to prevent destructive large changes
- **Advantages**:
  - More stable than older policy gradient methods
  - Works well with continuous action spaces
  - Less sensitive to hyperparameters
- **Use when**: You need robust, stable training or plan to extend to continuous control

#### 2. **Custom RL Implementations**

The project also includes custom-built RL agents for learning purposes:

- **`dqn_agent.py`**: Educational DQN implementation from scratch
- **`ppo_agent.py`**: Custom PPO implementation

**When to use custom implementations:**
- Learning RL fundamentals and algorithm internals
- Experimenting with novel algorithm modifications
- Research projects requiring non-standard techniques
- Educational purposes and debugging

---

## Environment Details

### State Space (Continuous)

The rocket's state is represented by 6 continuous variables:

| Variable | Description | Range |
|----------|-------------|-------|
| **x** | Horizontal position | [-1.0, 1.0] (normalized) |
| **y** | Vertical position | [0.0, 1.2] (normalized) |
| **vx** | Horizontal velocity | [-2.0, 2.0] |
| **vy** | Vertical velocity | [-2.0, 2.0] |
| **θ** | Angle from vertical | [-π, π] radians |
| **ω** | Angular velocity | [-2.0, 2.0] rad/s |

### Action Space (Discrete)

The agent controls the rocket through 4 discrete actions:

| Action ID | Description | Effect |
|-----------|-------------|--------|
| 0 | No-op | Coast/freefall (no thrust, no rotation) |
| 1 | Rotate left | Apply angular thrust counterclockwise |
| 2 | Main engine | Fire main thruster along rocket axis |
| 3 | Rotate right | Apply angular thrust clockwise |

### Reward Function

The reward function shapes learning through continuous feedback and terminal bonuses:

```python
# Step rewards (during flight)
reward = 0.0

# Angle penalty - encourages staying upright
reward -= 0.1 * abs(angle)

# Velocity penalty - encourages slow, controlled descent
reward -= 0.3 * (abs(vx) + abs(vy))

# Position penalty - encourages staying centered over landing pad
reward -= 0.2 * abs(x)

# Time penalty - encourages efficiency
reward -= 0.05

# Terminal rewards (upon ground contact)
if y <= 0.0:
    # Check landing conditions
    soft_landing = (
        abs(x) < 0.1          # Centered on landing pad
        and abs(vx) < 0.1     # Low horizontal velocity
        and abs(vy) < 0.2     # Low vertical velocity  
        and abs(angle) < 0.1  # Nearly upright
    )
    
    if soft_landing:
        reward += 100.0       # Success bonus
    else:
        reward -= 100.0       # Crash penalty
```

**Reward Shaping Strategy:**
- **Continuous penalties** guide the agent toward desired behavior during flight
- **Angle stability**: Staying upright is constantly rewarded
- **Velocity control**: Slower descent receives better rewards
- **Centering**: Being over the landing pad is incentivized
- **Terminal bonus**: Large rewards/penalties at episode end drive learning

### Physics Simulation

The environment uses simplified 2D rigid body dynamics:

**Forces & Motion:**
- **Gravity**: Constant downward acceleration (`-0.05` per timestep)
- **Main thrust**: Force applied along rocket's longitudinal axis (`0.20` acceleration)
- **Angular thrust**: Torque for rotation control (`±0.05` angular acceleration)
- **Velocity integration**: Direct position updates from velocity

**Physical Constraints:**
- **Bounded arena**: Horizontal position clipped to `[-1.0, 1.0]`, vertical to `[0.0, 1.2]`
- **Ground collision**: Episode terminates when `y ≤ 0`
- **No drag**: Simplified physics without aerodynamic forces
- **No fuel limit**: Unlimited thrust available (can be modified for added difficulty)

---

## Project Structure

```
Self-Landing-Rocket/
│
├── rocket_landing/
│   ├── environment/
│   │   └── rocket_env.py              # Gymnasium environment implementation
│   │
│   ├── agents/
│   │   ├── dqn_agent.py               # Custom DQN implementation
│   │   └── ppo_agent.py               # Custom PPO implementation 
│   │
│   └── training/
│       ├── train_rllib_dqn.py         # RLlib DQN training 
│       ├── train_rllib_ppo.py         # RLlib PPO training 
│       ├── train_custom_dqn.py        # Custom DQN training 
│       ├── train_custom_ppo.py        # Custom PPO training 
│       └── training_plots/            # Generated learning curves
│
├── policies/
│   ├── rllib_dqn_best/                # Best DQN checkpoint
│   └── rllib_ppo_best/                # Best PPO checkpoint
│
├── gifs/                              # Rollout visualizations
├── ray_results/                       # RLlib experiment logs
│
├── pyproject.toml                     # Poetry dependencies
├── README.md
└── .venv/                             # Virtual environment
```

---

## Installation

### Prerequisites

- **Python 3.11** (required — Ray RLlib doesn't support Python 3.13 yet)
- **Poetry** package manager

#### Verify Python Installation

```bash
# Check available Python versions
py --list

# Should show Python 3.11.x
```

#### Install Poetry (if not already installed)

```bash
# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -

# Linux/macOS
curl -sSL https://install.python-poetry.org | python3 -
```

### Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Self-Landing-Rocket.git
cd Self-Landing-Rocket

# 2. Install dependencies with Poetry
poetry install

# 3. Activate virtual environment
poetry shell

# 4. Verify installation
poetry run python -c "import gymnasium; import ray; print('Setup successful!')"
```

---

## Training Agents

### Option 1: RLlib Training

#### Train with DQN

```bash
poetry run python -m rocket_landing.training.train_rllib_dqn
```

**Training process:**
- Initializes Ray cluster for distributed execution
- Creates vectorized environments for parallel sampling
- Trains DQN with experience replay and target networks
- Saves checkpoints every N iterations
- Generates training curves automatically

**Outputs:**
- Checkpoints: `./ray_results/DQN_RocketLanding_[timestamp]/`
- Best policy: `./policies/rllib_dqn_best/`
- Training plots: `./training_plots/dqn_training_curves.png`

#### Train with PPO

```bash
poetry run python -m rocket_landing.training.train_rllib_ppo
```

**Training process:**
- Uses multiple parallel workers for on-policy sampling
- Implements clipped surrogate objective for stable updates
- Employs GAE (Generalized Advantage Estimation)
- Automatically tunes learning rate during training

**Outputs:**
- Checkpoints: `./ray_results/PPO_RocketLanding_[timestamp]/`
- Best policy: `./policies/rllib_ppo_best/`
- Training plots: `./training_plots/ppo_training_curves.png`

### Option 2: Custom Implementations

#### Train Custom DQN

```bash
poetry run python -m rocket_landing.training.train_custom_dqn
```

**What you'll learn:**
- Q-learning update rules
- Experience replay buffer implementation
- Epsilon-greedy exploration strategies
- Target network synchronization

#### Train Custom PPO

```bash
poetry run python -m rocket_landing.training.train_custom_ppo
```

**What you'll learn:**
- Policy gradient theorem
- Advantage estimation techniques
- Clipping mechanisms for stability
- On-policy vs off-policy learning

---

## Algorithm Comparison

| Feature | RLlib DQN | RLlib PPO | Custom DQN | Custom PPO |
|---------|-----------|-----------|------------|------------|
| **Training Speed** | Fast | Fast | Slow | Slow |
| **Sample Efficiency** | High | Medium | High  Medium |
| **Stability** | Excellent |  Excellent  Good  Good |
| **Scalability** | Multi-core | Multi-core | Single-core | Single-core |
| **Best For** | Production | Production | Learning | Learning |
| **Action Space** | Discrete only | Both | Discrete only | Both |
| **Hyperparameter Tuning** | Auto-tuning | Auto-tuning | Manual | Manual |

### When to Choose Each Algorithm

**Choose RLlib DQN if:**
- You have discrete actions
- Sample efficiency is critical (limited environment interactions)
- You want reliable, production-ready training
- You need fast convergence

**Choose RLlib PPO if:**
- You might extend to continuous actions later
- You want maximum stability
- You're new to RL and want forgiving hyperparameters
- You prefer on-policy learning

**Choose Custom Implementations if:**
- You're learning RL fundamentals
- You need to modify algorithm internals
- You're conducting research on novel techniques
- Educational purposes

---

## Customization

### Modify Environment Parameters

Edit `rocket_landing/environment/rocket_env.py`:

```python
class RocketEnv(gym.Env):
    def __init__(self):
        # Physics constants
        self.gravity = -0.05          # Downward acceleration per step
        self.main_thrust = 0.20       # Main engine acceleration
        self.angular_thrust = 0.05    # Rotational acceleration
        
        # Boundary limits
        self.max_x = 1.0              # Horizontal bounds
        self.max_y = 1.2              # Maximum altitude
        
        # Initial conditions (in reset())
        # x: random in [-0.1, 0.1]
        # y: starts at 1.0
        # angle: random in [-0.05, 0.05]
```

**Difficulty Adjustments:**
- Increase `gravity` for harder landings
- Decrease `main_thrust` to limit control authority
- Increase initial altitude by changing `y = 1.0` to `y = 1.5`
- Add initial velocities in `reset()` for dynamic starts
- Tighten landing tolerances in reward function

### Tune Hyperparameters

Edit `rocket_landing/training/train_rllib_dqn.py`:

```python
config = (
    DQNConfig()
    .environment(
        env="RocketLanding-v0",
        env_config={"render_mode": None},
    )
    .framework("torch")
    .training(
        gamma=0.99,                       # Discount factor
        lr=1e-3,                          # Learning rate
        train_batch_size=4000,            # Samples per training step
        replay_buffer_config={
            "capacity": 200000            # Experience replay size
        },
        target_network_update_freq=1000,  # Target net sync frequency
        dueling=True,                     # Use dueling DQN architecture
        double_q=True,                    # Use double Q-learning
    )
)

# Training configuration
stop = {"training_iteration": 200}        # Number of training iterations
checkpoint_frequency = 10                 # Save checkpoint every N iterations
```

**Key Hyperparameters Explained:**
- `gamma`: How much future rewards matter (0.99 = far-sighted)
- `lr`: Learning rate for neural network updates
- `train_batch_size`: Larger = more stable but slower
- `replay_buffer_config.capacity`: More memory = better sample diversity
- `target_network_update_freq`: Lower = more stable but slower learning

### Custom Reward Shaping

Modify the reward function in `rocket_env.py`:

```python
def step(self, action):
    # ... [action and physics code] ...
    
    # Custom reward logic
    reward = 0.0
    
    # Heavier penalty for being off-center
    reward -= 0.5 * abs(x)  # Increased from 0.2
    
    # Reward for descending (negative vy)
    if vy < 0:
        reward += 0.1  # Bonus for controlled descent
    
    # Exponential angle penalty
    reward -= 0.2 * (angle ** 2)  # Quadratic instead of linear
    
    # Terminal rewards
    if y <= 0.0:
        soft_landing = (
            abs(x) < 0.05       # Tighter centering requirement
            and abs(vx) < 0.05  # Slower horizontal speed
            and abs(vy) < 0.15  # Slower vertical speed
            and abs(angle) < 0.05  # More upright requirement
        )
        
        if soft_landing:
            reward += 100.0
        else:
            reward -= 100.0
    
    return self.state, reward, terminated, truncated, {}
```

---

## Results & Visualization

### Training Curves

After training, view learning progress:

```
training_plots/
├── dqn_training_curves.png         # DQN learning curves
└── ppo_training_curves.png         # PPO learning curves
```

**Metrics plotted:**
- **Episode Reward Mean**: Average cumulative reward per episode
- **Episode Length Mean**: Average number of steps before termination
- **Success Rate**: Percentage of successful landings (optional)

### Generate Rollout Videos

Visualize trained policies in action:

```bash
# Rollout with DQN policy
poetry run python -m rocket_landing.utils.rollout --policy rllib_dqn_best --episodes 10

# Rollout with PPO policy
poetry run python -m rocket_landing.utils.rollout --policy rllib_ppo_best --episodes 10
```

Saves animated GIFs to `./gifs/`

### Real-Time Training Monitoring with TensorBoard

RLlib automatically logs training metrics to TensorBoard format. You can monitor training progress in real-time or review completed experiments.

**Launch TensorBoard**

```bash 
# From project root directory
poetry run tensorboard --logdir ray_results/
```

Then open your browser and navigate to:

```
http://localhost:6006
```

**What You'll See**

TensorBoard provides rich visualizations of your training run:

**Scalars Tab** (most important):

- `episode_reward_mean:` Average reward per episode (main success metric)
- `episode_len_mean:` Average episode length in timesteps
- `num_env_steps_sampled_lifetime:` Total environment interactions
- `num_episodes_lifetime:` Total episodes completed
- `Loss metrics:` DQN loss, TD error, Q-values

**Distributions Tab:**

- Q-value distributions over time
- Action selection frequencies
- Reward distributions

---

## Acknowledgments

- **OpenAI Gymnasium**: Standard RL environment API
- **Ray RLlib**: Scalable RL framework
- **PyGame**: Visualization and rendering

---
