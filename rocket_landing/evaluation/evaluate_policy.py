import imageio
from pathlib import Path

from rocket_landing.evaluation.load_policy import load_policy
from rocket_landing.environment.rocket_env import RocketEnv



# ---------------------------------
# ----- CONFIG --------------------
# ---------------------------------

USE_RANDOM_POLICY = False
policy = "policies/rllib_dqn_best"


# ---------------------------------
# ----- ENV SETUP -----------------
# ---------------------------------

# rgb_array so taht the frames can save as a gif
env = RocketEnv(render_mode = "rgb_array")

# ---------------------------------
# ----- POLICY SETUP --------------
# ---------------------------------

if USE_RANDOM_POLICY:
    print("Using RANDOM policy")

    policy = "random"

    def chosen_policy(observation):
        return env.action_space.sample()

else:
    print("Using TRAINED policy")
    chosen_policy = load_policy(policy)


# ---------------------------------
# ----- INITIAL STATE -------------
# ---------------------------------

# Set the initial state
observation, info = env.reset(seed = 42)

# initlialize an emtpy list to hold frames
frames = []

# Set max step and step counter variables to safeguard against infinite loops
max_steps = 2000
steps = 0

# Set initial reward to 0
G = 0

# Get the first frame from the environment
frames.append(env.render())



# ---------------------------------
# ----- RUN SINGLE EPISODE --------
# ---------------------------------

# Loop through episodes
while True:
    # Take an action
    action = chosen_policy(observation)

    # Obtain the observation, reward, termiantion and truncation status, and info
    observation, reward, terminated, truncated, info = env.step(action)

    # Append the new frame to the frames list
    frames.append(env.render())

    # Upadte total reward for the episode
    G += reward

    # Increment step counter
    steps += 1

    # Update termination status if the episode is over
    if terminated or truncated or steps >= max_steps:
        break

# Clsoe the environment
env.close()

# Print episodes rewards
print("Episode return: {G}")

# ---------------------------------
# ----- SAVE GIF ------------------
# ---------------------------------

# Set project root for saving
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GIF_DIR = PROJECT_ROOT / "gifs"
GIF_DIR.mkdir(parents = True, exist_ok = True)

name = Path(policy).stem
gif_path = GIF_DIR / f"{name}.gif"

# Save the frames as a gif
imageio.mimsave(
    gif_path,
    frames,
    fps = 60
)


"""
HOW TO RUN:

poetry run python -m rocket_landing.evaluation.evaluate_policy

"""