import gym
from gym import spaces
import pygame
import numpy as np

class MyWorld(gym.Env):
    def __init__(self, training=True, render_mode="human", size=25, quantity_obstacle=5, render_fps=4):
        self.size = size
        self.quantity_obstacle = quantity_obstacle
        self.window_size = 500
        self.observation_space = spaces.Discrete(size)
        self.action_space = spaces.Discrete(4)
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.training = training
        self.window = None
        self.clock = None
        self.target_positions = [3, 10, 22]
        self.target_index = 0

    def get_obs(self):
        return self.agent_location

    def reset(self):
        super().reset(seed=None)
        self.obstacles = [2, 12, 9, 21]
        self.target_location = self.target_positions[self.target_index]
        self.target_index = (self.target_index + 1) % len(self.target_positions)
        self.agent_location = 0

        observation = self.get_obs()
        if self.render_mode == "human" and self.training:
            self.render_frame()

        return observation

    def step(self, action):
        old_agent_location = np.copy(self.agent_location)

        checker = int(np.sqrt(self.size))
        if action == 0:  # Move right
            if (self.agent_location + 1) % checker != 0 or 0 <= self.agent_location < 4:
                self.agent_location = self.agent_location + 1
            else:
                reward = -1
        elif action == 1:  # Move left
            if (self.agent_location - 1) % checker != 4 or 0 < self.agent_location <= 4:
                self.agent_location = self.agent_location - 1
            else:
                reward = -1
        elif action == 2:  # Move down
            if 0 < self.agent_location + checker < self.size:
                self.agent_location = self.agent_location + checker
            else:
                reward = -1
        elif action == 3:  # Move up
            if 0 <= self.agent_location - checker < self.size:
                self.agent_location = self.agent_location - checker
            else:
                reward = -1

        terminated = False
        if self.agent_location in self.obstacles:
            self.agent_location = old_agent_location
            reward = -1
        else:
            if self.agent_location == self.target_location:
                reward = 10
                terminated = True
            else:
                reward = 0

        observation = self.get_obs()
        if self.render_mode == "human" and self.training:
            self.render_frame()

        return observation, reward, terminated, False

    def render(self):
        if self.render_mode == "rgb_array":
            return self.render_frame()

    def render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        pix_square_size = self.window_size / np.sqrt(self.size)

        # Цель
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * (self.target_location % int(np.sqrt(self.size))),
                pix_square_size * (self.target_location // int(np.sqrt(self.size))),
                pix_square_size,
                pix_square_size
            )
        )
        # Агент
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (
                (self.agent_location % int(np.sqrt(self.size)) + 0.5) * pix_square_size,
                (self.agent_location // int(np.sqrt(self.size)) + 0.5) * pix_square_size
            ),
            pix_square_size / 3
        )

        for obstacle in self.obstacles:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    pix_square_size * (obstacle % int(np.sqrt(self.size))),
                    pix_square_size * (obstacle // int(np.sqrt(self.size))),
                    pix_square_size,
                    pix_square_size
                )
            )

        for x in range(int(np.sqrt(self.size)) + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.render_fps)
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
