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
from ray.rllib.algorithms.ppo import PPOConfig

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
        PPOConfig()
        .environment(
            env = "RocketLanding-v0",
            env_config = {"render_mode": None},
        )
        .framework("torch")
        .debugging(log_level = "INFO")
        .env_runners(
            num_env_runners = 8,
            num_envs_per_env_runner = 4
        )
        .training(
            gamma = 0.99,
            lr = 1e-4,
            train_batch_size = 6000,
            minibatch_size = 128,
            num_epochs = 10,
            clip_param = 0.2,
            entropy_coeff = 0.01,
            vf_loss_coeff = 0.5,
        )
    )

    ray.init(
        ignore_reinit_error = True,
        include_dashboard = False,
        logging_level = "ERROR"
    )

    # -------------------------------------
    # ----- TRAIN & WRITE CHECKPOINTS -----
    # -------------------------------------

    tuner = tune.Tuner(
        "PPO",
        run_config = RunConfig(
            stop = {
                "training_iteration": 200,
                "env_runners/episode_return_mean": 175
            },
            checkpoint_config = CheckpointConfig(
                checkpoint_frequency = 10,
                checkpoint_at_end = True
            ),
            storage_path = storage_uri,
            name = "ppo_rocket"
        ),
        param_space = config.to_dict(), 
    )

    results = tuner.fit()



    # ---------------------------
    # ----- EXTRACT METRICS -----
    # ---------------------------

    dfs = []
    for result in results:
        df = result.metrics_dataframe
        if df is not None:
            dfs.append(df[["training_iteration", "env_runners/episode_return_mean", "env_runners/episode_len_mean"]])


    df = pd.concat(dfs).dropna()
    df = df.rename(columns={
        "training_iteration": "iteration",
        "env_runners/episode_return_mean": "episode_return_mean",
        "env_runners/episode_len_mean": "episode_len_mean"
    })

    # ------------------------
    # ----- PLOT METRICS -----
    # ------------------------

    fig, axes = plt.subplots(1, 2, figsize = (14, 5))

    # Reward plot
    axes[0].plot(df["iteration"], df["episode_return_mean"])
    axes[0].set_title("Mean Episode Return")
    axes[0].set_xlabel("Training Iteration")
    axes[0].set_ylabel("Reward")

    # Episode length plot
    axes[1].plot(df["iteration"], df["episode_len_mean"])
    axes[1].set_title("Mean Episode Length")
    axes[1].set_xlabel("Training Iteration")
    axes[1].set_ylabel("Timesteps")

    PLOTS_DIR = Path(__file__).resolve().parent / "training_plots"
    PLOTS_DIR.mkdir(parents = True, exist_ok = True)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ppo_training_curves.png", dpi = 150)
    plt.close()

    # Print the best checkpoint path
    best_result = results.get_best_result(
        metric = "env_runners/episode_return_mean",
        mode = "max"
    )

    best_checkpoint = best_result.checkpoint.path
    print(f"Best checkpoint: {best_checkpoint}")

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    policy_dir = PROJECT_ROOT / "policies" / "rllib_ppo_best"
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

python -m rocket_landing.training.train_rllib_ppo

"""