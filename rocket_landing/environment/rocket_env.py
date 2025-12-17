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

        self.x_limit = 1

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
            [1.0, 1.2, 2.0, 2.0, math.pi, 2.0, 1.0],
            dtype = np.float32
        )

        low = -high
        low[6] = 0.0

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

        self.last_action = 0

        # Return all varaibles back to a common rest state
        self.state = np.array(
            [
                np.random.uniform(-0.1, 0.1),       # x
                1.0,                                # y
                0.0,                                # vx
                0.0,                                # vy
                np.random.uniform(-0.05, 0.05),     # angle
                0.0,                                # angular_velocity
                1.0                                 # fuel
            ],
            dtype = np.float32
        )

        self.state = np.clip(
            self.state,
            self.observation_space.low,
            self.observation_space.high
        ).astype(np.float32)

        return self.state, {}

    # ----------------
    # ----- STEP -----
    # ----------------

    def step(self, action):
        x, y, vx, vy, angle, ang_vel, fuel = self.state
        dt = 0.05
        self.last_action = action

        if action == 1:
            ang_vel += self.angular_thrust
        elif action == 3:
            ang_vel -= self.angular_thrust

        if action == 2 and fuel > 0.0:
            vx += math.sin(angle) * self.main_thrust
            vy +=  math.cos(angle) * self.main_thrust
            fuel -= 0.01

        fuel = max(0.0, fuel)

        vy += self.gravity
        ang_vel *= 0.99

        x += vx * dt
        y += vy * dt

        angle += ang_vel * dt
        angle = (angle + math.pi) % (2 * math.pi) - math.pi

        reward = 0.0
        reward -= 0.1 * abs(angle)
        reward -= 0.3 * (abs(vx) + abs(vy))
        reward -= 0.2 * abs(x)
        reward -= 0.05 * abs(ang_vel)
        reward -= 0.1 * (1.0 - fuel)

        terminated = False

        if abs(x) > self.x_limit:
            terminated = True
            reward -= 50.0

        if y <= 0.0:
            terminated = True
            y = 0.0

            soft_landing = (
                abs(x) < 0.1
                and abs(vx) < 0.1
                and abs(vy) < 0.2
                and abs(angle) < 0.1
            )

            reward += 100.0 if soft_landing else -100.0

        if not terminated:
            reward -= 0.05

        self.state = np.array([x, y, vx, vy, angle, ang_vel, fuel], dtype = np.float32)

        self.state = np.clip(
            self.state,
            self.observation_space.low,
            self.observation_space.high
        )

        return self.state, reward, terminated, False, {}



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
        x, y, _, _, angle, _, fuel = self.state

        # Set
        px = int(self.screen_width / 2 + x * self.scale)

        # Set 
        py = int(self.screen_height - (y * self.scale) - 10)

        # Draw the rocket
        rocket = pygame.Surface((20, 50), pygame.SRCALPHA)

        # Color the rocket purple 
        rocket.fill((200, 200, 255))

        # Angle the rocket the correct amount 
        rotated = pygame.transform.rotate(rocket, -math.degrees(angle))

        # Get the location of the rocket
        rect = rotated.get_rect(center = (px, py))

        # -----------------------------
        # ----- VISUALIZE THRUST ------
        # -----------------------------

        def rotate_point(x, y, angle):
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            return (
                x * cos_a - y * sin_a,
                x * sin_a + y * cos_a
            )

        # MAIN THRUST

        if self.last_action == 2:
            flame_local = [
                (-6, 25),
                (6, 25),
                (0, 45)
            ]

            flame_world = []
            for lx, ly in flame_local:
                rx, ry = rotate_point(lx, ly, angle)
                flame_world.append((px + rx, py + ry))

            pygame.draw.polygon(
                self.screen,
                (255, 180, 0),
                flame_world
            )

        # LEFT ROTATION THRUST

        if self.last_action == 1:
            side_flame_local = [
                (10, -4),
                (18, 0),
                (10, 4)
            ]

            flame_world = []
            for lx, ly in side_flame_local:
                rx, ry = rotate_point(lx, ly, angle)
                flame_world.append((px + rx, py + ry))

            pygame.draw.polygon(
                self.screen,
                (255, 100, 0),
                flame_world
            )

        # RIGHT ROTATION THRUST

        if self.last_action == 3:
            side_flame_local = [
                (-10, -4),
                (-18, 0),
                (-10, 4)
            ]

            flame_world = []
            for lx, ly in side_flame_local:
                rx, ry = rotate_point(lx, ly, angle)
                flame_world.append((px + rx, py + ry))

            pygame.draw.polygon(
                self.screen,
                (255, 100, 0),
                flame_world
            )

        # -----------------------------
        # ----- FUEL BAR --------------
        # -----------------------------

        bar_width = 12
        bar_height = 120
        bar_margin = 15

        bar_x = self.screen_width - bar_width - bar_margin
        bar_y = bar_margin

        # Clamp fuel to [0, 1]
        fuel_frac = np.clip(fuel, 0.0, 1.0)

        # Color based on remaining fuel
        if fuel_frac > 0.3:
            fuel_color = (50, 220, 50)
        elif fuel_frac > 0.1:
            fuel_color = (255, 180, 0)
        else:
            fuel_color = (255, 80, 80)

        # Background (empty tank)
        pygame.draw.rect(
            self.screen,
            (80, 80, 80),
            (bar_x, bar_y, bar_width, bar_height),
            border_radius = 3
        )

        # Filled portion
        fill_height = int(bar_height * fuel_frac)
        fill_y = bar_y + (bar_height - fill_height)

        pygame.draw.rect(
            self.screen,
            fuel_color,
            (bar_x, fill_y, bar_width, fill_height),
            border_radius = 3
        )

        # Outline
        pygame.draw.rect(
            self.screen,
            (200, 200, 200),
            (bar_x, bar_y, bar_width, bar_height),
            1,
            border_radius = 3
        )


        pygame.draw.rect(
            self.screen,
            (50, 220, 50),
            (bar_x, fill_y, bar_width, fill_height),
            border_radius = 3
        )

        # Optional outline
        pygame.draw.rect(
            self.screen,
            (200, 200, 200),
            (bar_x, bar_y, bar_width, bar_height),
            1,
            border_radius = 3
        )


        # Draw the rocket
        self.screen.blit(rotated, rect)

        # Draw a landing pad
        pygame.draw.rect(
            self.screen,
            (200, 200, 200),
            (self.screen_width // 2 - 40, self.screen_height - 15, 80, 5)
        )

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
