import os
from pathlib import Path
import shutil

import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym
import ray
from ray import tune
from ray.tune import RunConfig
from ray.tune import CheckpointConfig
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn import DQNConfig

from rocket_landing.environment.rocket_env import RocketEnv


def env_creator(env_config):
    render_mode = env_config.get("render_mode", None)
    return RocketEnv(render_mode = render_mode)


storage_path = Path("./ray_results").resolve()
storage_uri = f"file:///{storage_path.as_posix()}"

def main():
    # Register environment
    register_env("RocketLanding-v0", env_creator)


    # ---------------------------------
    # ----- CONFIGURE ENVIRONEMNT -----
    # ---------------------------------

    config = (
        DQNConfig()
        .environment(
            env = "RocketLanding-v0",
            env_config = {"render_mode": None},
        )
        .framework("torch")
        .env_runners(num_env_runners = 0)
        .training(
            gamma = 0.99,
            lr = 1e-3,
            train_batch_size = 4000,
            replay_buffer_config = {"capacity": 200000},
            target_network_update_freq = 1000,
            dueling = True,
            double_q = True,
        )
    )

    ray.init(
        local_mode = True,
        ignore_reinit_error = True,
        include_dashboard = False,
        logging_level = "ERROR"
    )

    # -------------------------------------
    # ----- TRAIN & WRITE CHECKPOINTS -----
    # -------------------------------------

    tuner = tune.Tuner(
        "DQN",
        run_config = RunConfig(
            stop = {"training_iteration": 20},
            checkpoint_config = CheckpointConfig(
                checkpoint_frequency = 10,
                checkpoint_at_end = True
            ),
            storage_path = storage_uri,
            name = "dqn_rocket"
        ),
        param_space = config.to_dict(),
    )

    results = tuner.fit()

    for result in results:
        print(
            result.metrics["training_iteration"],
            result.metrics["episode_reward_mean"]
        )

    # ---------------------------
    # ----- EXTRACT METRICS -----
    # ---------------------------

    metrics = []

    for result in results:
        metrics.append({
            "iteration": result.metrics["training_iteration"],
            "episode_reward_mean": result.metrics["episode_reward_mean"],
            "episode_len_mean": result.metrics["episode_len_mean"],
        })

    df = pd.DataFrame(metrics)

    # Ensure output directory exists
    os.makedirs("./training_plots", exist_ok = True)

    # ------------------------
    # ----- PLOT METRICS -----
    # ------------------------

    fig, axes = plt.subplots(1, 2, figsize = (14, 5))

    # Reward plot
    axes[0].plot(df["iteration"], df["episode_reward_mean"])
    axes[0].set_title("Mean Episode Reward")
    axes[0].set_xlabel("Training Iteration")
    axes[0].set_ylabel("Reward")

    # Episode length plot
    axes[1].plot(df["iteration"], df["episode_len_mean"])
    axes[1].set_title("Mean Episode Length")
    axes[1].set_xlabel("Training Iteration")
    axes[1].set_ylabel("Timesteps")

    plt.tight_layout()
    plt.savefig("./training_plots/dqn_training_curves.png", dpi = 150)
    plt.close()

    # Print the best checkpoint path
    best_result = results.get_best_result(
        metric = "env_runners/episode_reward_mean",
        mode = "max"
    )

    best_checkpoint = best_result.checkpoint.path
    print(f"Best checkpoint: {best_checkpoint}")

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    policy_dir = PROJECT_ROOT / "policies" / "rllib_dqn_best"
    policy_dir.mkdir(parents = True, exist_ok = True)

    shutil.copytree(
        best_checkpoint,
        policy_dir,
        dirs_exist_ok = True
    )


if __name__ == "__main__":
    main()


"""
HOW TO RUN:

python -m rocket_landing.training.train_rllib_dqn

"""