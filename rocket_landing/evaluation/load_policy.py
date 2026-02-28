from pathlib import Path
import torch
import numpy as np

from ray import tune
from ray.rllib.algorithms.algorithm import Algorithm

from rocket_landing.environment.rocket_env import RocketEnv


def load_policy(policy_path):
    policy_path = Path(policy_path).resolve()

    # ---------------------------------
    # ----- REGISTER ENV FOR RLLIB -----
    # ---------------------------------

    def env_creator(env_config):
        render_mode = env_config.get("render_mode", None)
        return RocketEnv(render_mode = render_mode)

    tune.register_env(
        "RocketLanding-v0",
        lambda cfg: env_creator(cfg)
    )

    # ---------------------------------
    # ----- RLlib CHECKPOINT ----------
    # ---------------------------------

    if policy_path.is_dir():
        algo = Algorithm.from_checkpoint(str(policy_path))
        print("Loaded RLlib policy")

        module = algo.get_module()

        def policy_fn(observation):
            obs_batch = np.expand_dims(observation, axis = 0)
        
            outputs = module.forward_inference(
                {"obs": torch.tensor(obs_batch, dtype=torch.float32)}
            )

            # PPO returns logits, DQN returns Q-values — different keys
            if "action_dist_inputs" in outputs:
                # PPO — argmax over logits
                logits = outputs["action_dist_inputs"]
                action = torch.argmax(logits, dim=-1).cpu().numpy()
                if action.ndim == 0:
                    return int(action.item())
                else:
                    return int(action[0])
            else:
                # DQN — action is already selected
                action = outputs["actions"]
                if action.ndim == 0:
                    return int(action.item())
                else:
                    return int(action[0])

        return policy_fn

    # ---------------------------------
    # ----- TORCH MODEL (.pth) --------
    # ---------------------------------

    model = torch.load(policy_path, map_location = "cpu")
    model.eval()
    print("Loaded Torch policy")

    def policy_fn(observation):
        with torch.no_grad():
            obs = torch.tensor(
                observation,
                dtype = torch.float32
            ).unsqueeze(0)
            return model(obs).argmax(dim = 1).item()

    return policy_fn
