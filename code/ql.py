import random
import numpy as np
import pickle
import gym
from gym import spaces
import pygame
import numpy as np

class MyWorld(gym.Env):
    def __init__(self, training=True, render_mode="human", size=100, quantity_obstacle=5, render_fps=4):
        self.size = size
        self.quantity_obstacle = quantity_obstacle
        self.window_size = 500
        self.observation_space = spaces.Discrete(size)
        self.action_space = spaces.Discrete(4)
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.training = not training
        self.window = None
        self.clock = None

    def get_obs(self):
        return self.agent_location

    def reset(self):
        super().reset(seed=None)
        self.obstacles = [4, 22, 27, 28, 40, 62, 66, 72, 35, 89, 93]
        # self.agent_location = 0
        self.target_location = 99
        self.agent_location = np.random.randint(0, self.size)
        while self.agent_location in self.obstacles:
            self.agent_location = np.random.randint(0, self.size)
        self.agent_position = self.agent_location
        # self.target_location = self.agent_location
        # while self.target_location == self.agent_location or self.target_location in self.obstacles:
        #     self.target_location = np.random.randint(0, self.size)

        # ob = [1, 2, 3, 4, 10, 11, 12, 13, 20, 21, 22, 30, 31, 40, 59, 68, 69, 77, 78, 79, 86, 87, 88, 89, 95, 96, 97, 98]
        # self.obstacles = []
        # for _ in range(self.quantity_obstacle):
        #     obstacle = np.random.randint(0, self.size)
        #     while (
        #             obstacle in self.obstacles
        #             or obstacle in ob
        #             or obstacle == self.agent_location
        #             or obstacle == self.target_location
        #     ):
        #         obstacle = np.random.randint(0, self.size)
        #     self.obstacles.append(obstacle)

        observation = self.get_obs()
        if self.render_mode == "human" and self.training:
            self.render_frame()

        return observation

    def step(self, action):
        old_agent_location = np.copy(self.agent_location)

        checker = int(np.sqrt(self.size))
        if action == 0:  # Move right
            if (self.agent_location + 1) % checker != 0 or 0 <= self.agent_location < 9:
                self.agent_location = self.agent_location + 1
            else:
                reward = -1
        elif action == 1:  # Move left
            if (self.agent_location - 1) % checker != 9 or 0 < self.agent_location <= 9:
                self.agent_location = self.agent_location - 1
            else:
                reward = -1
        elif action == 2:  # Move down
            if 0 < self.agent_location + checker < 100:
                self.agent_location = self.agent_location + int(np.sqrt(self.size))
            else:
                reward = -1
        elif action == 3:  # Move up
            if 0 <= self.agent_location - checker < 100:
                self.agent_location = self.agent_location - int(np.sqrt(self.size))
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

        pygame.draw.rect(
            canvas,
            (57, 255, 20),
            pygame.Rect(
                pix_square_size * (self.agent_position % int(np.sqrt(self.size))),
                pix_square_size * (self.agent_position // int(np.sqrt(self.size))),
                pix_square_size,
                pix_square_size
            )
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
def learning(episodes, fps, training):
    env = MyWorld(training=training, render_fps=fps)
    if (training):
        q_table = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        f = open('table.pkl', 'rb')
        q_table = pickle.load(f)
        f.close()
    alpha = 0.9
    gamma = 0.6
    epsilon = 0.1
    for i in range(episodes):
        state = env.reset()
        epochs, penalties, reward, = 0, 0, 0
        terminated = False
        while not terminated:
            if training and random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, terminated, truncated = env.step(action)

            if training:
                old_value = q_table[state, action]
                next_max = np.max(q_table[next_state])

                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                q_table[state, action] = new_value

                if reward == -1:
                    penalties += 1
            state = next_state
            epochs += 1
        print(f"Episode: {i}")
    if training:
        print("Training finished.\n")
        f = open("table.pkl", "wb")
        pickle.dump(q_table, f)
        f.close()

if __name__ == "__main__":
    learning(1000, 10, training=False)