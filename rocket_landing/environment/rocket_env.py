import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math


class RocketEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode = None):
        super().__init__()

        # -----------------------------
        # ----- PHYSICS CONSTANTS -----
        # -----------------------------
        self.gravity = -0.05
        self.main_thrust = 0.20
        self.side_thrust = 0.05
        self.angular_thrust = 0.05

        self.max_x = 1.0
        self.max_y = 1.2

        # ------------------------
        # ----- ACTION SPACE -----
        # ------------------------
        # 0: no-op
        # 1: rotate left
        # 2: main engine
        # 3: rotate right

        self.action_space = spaces.Discrete(4)

        # -----------------------------
        # ----- OBSERVATION SPACE -----
        # -----------------------------
        # [x, y, vx, vy, angle, angular_velocity]


        high = np.array(
            [1.0, 1.2, 2.0, 2.0, math.pi, 2.0],
            dtype = np.float32
        )

        self.observation_space = spaces.Box(
            low = -high,
            high = high,
            dtype = np.float32
        )

        # Rendering
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.screen_width = 600
        self.screen_height = 400
        self.scale = 200

        self.reset()

    # -----------------
    # ----- RESET -----
    # -----------------
    def reset(self, seed = None, options = None):
        super().reset(seed = seed)

        # Return all varaibles back to a common rest state
        self.state = np.array(
            [
                np.random.uniform(-0.1, 0.1),       # x
                1.0,                                # y
                0.0,                                # vx
                0.0,                                # vy
                np.random.uniform(-0.05, 0.05),     # angle
                0.0                                 # angular_velocity
            ],
            dtype = np.float32
        )

        self.state = np.clip(
            self.state,
            self.observation_space.low,
            self.observation_space.high
        ).astype(np.float32)

        return self.state, {}

    # -----------------
    # Step
    # -----------------
    def step(self, action):
        # Initialize the current state
        x, y, vx, vy, angle, ang_vel = self.state

        # -------------------
        # ----- Actions -----
        # -------------------

        # Action is right side thrust 
        if action == 1:
            ang_vel += self.angular_thrust

        # Action is main thrust
        elif action == 2:
            vx += -math.sin(angle) * self.main_thrust
            vy +=  math.cos(angle) * self.main_thrust

        # Action is left side thrust
        elif action == 3:
            ang_vel -= self.angular_thrust

        # -------------------
        # ----- PHYSICS -----
        # -------------------

        # Add gravity
        vy += self.gravity

        # Update x velocity
        x += vx

        # Update y velocity
        y += vy

        # Update angualr velocity
        angle += ang_vel

        # --------------------------
        # ----- REWARD SHAPING -----
        # --------------------------

        reward = 0.0

        # Positive reward for being upright/Negative reward for being sideways
        reward -= 0.1 * abs(angle)

        # Positive reward for going slow/Negative reward for going fast
        reward -= 0.3 * (abs(vx) + abs(vy))

        # Positive reward for being centered/Negative reward for being off center
        reward -= 0.2 * abs(x)

        # Negative reward for time elapsed
        reward -= 0.05

        # --------------------------
        # ----- GROUND CONTACT -----
        # --------------------------

        # Ensure terminated is set to false
        terminated = False

        # Check if y value is less than or equal to 0
        if y <= 0.0:
            # Temrinate if y value is less than or equal to 0
            terminated = True

            # Set y to 0
            y = 0.0

            # Set conditions for a soft landing
            soft_landing = (
                abs(x) < 0.1            # Rocket is centered
                and abs(vx) < 0.1       # x velocity is low
                and abs(vy) < 0.2       # x velocity is low
                and abs(angle) < 0.1    # Rocket is upright
            )

            # Check if the landing was soft
            if soft_landing:
                # Large positive reward for soft and successful landing
                reward += 100.0

            # Check for a crash landing
            else:
                # Large negative reward for a crash landing
                reward -= 100.0

        # Clip bounds
        x = np.clip(x, -self.max_x, self.max_x)
        y = np.clip(y, 0.0, self.max_y)

        # Update state after action is taken and reward is gained/lost
        self.state = np.array([x, y, vx, vy, angle, ang_vel], dtype = np.float32)

        self.state = np.clip(
            self.state,
            self.observation_space.low,
            self.observation_space.high
        )

        # Render the frame if render_mode is "human"
        if self.render_mode == "human":
            self.render()

        truncated = False

        return self.state, reward, terminated, truncated, {}

    # ----------------------
    # ----- RENDER ENV -----
    # ----------------------

    def render(self):
        # Initialize screen
        if self.screen is None:
            # Initialize pygame
            pygame.init()

            # Set pygame screen to set width and height
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )

            # Start clock
            self.clock = pygame.time.Clock()

        # Fill screen with dark gray
        self.screen.fill((30, 30, 30))

        # Dray the ground as bright green
        pygame.draw.line(
            self.screen,
            (100, 255, 100),
            (0, self.screen_height - 10),
            (self.screen_width, self.screen_height - 10),
            3
        )

        # Initialize x, y, and rocket agnle variable from the state
        x, y, _, _, angle, _ = self.state

        # Set
        px = int(self.screen_width / 2 + x * self.scale)

        # Set 
        py = int(self.screen_height - (y * self.scale) - 10)

        # Draw the rocket
        rocket = pygame.Surface((10, 30))

        # Color the rocket purple 
        rocket.fill((200, 200, 255))

        # Angle the rocket the correct amount 
        rotated = pygame.transform.rotate(rocket, -math.degrees(angle))

        # Get the location of the rocket
        rect = rotated.get_rect(center = (px, py))

        # Draw the rocket
        self.screen.blit(rotated, rect)

        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)),
                axes = (1, 0, 2)
            )

    def close(self):
        if self.screen:
            pygame.quit()
