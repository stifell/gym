from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from collections import deque
import time
import gym
from gym import spaces
import random
import numpy as np
import pygame
import warnings
from keras.src.saving.saving_api import load_model
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class MyWorld(gym.Env):
    def __init__(self, training=True, render_mode="human", size=100, quantity_obstacle=4, render_fps=4):
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
        self.target_positions = [99, 90, 9, 33]
        self.target_index = 0
        self.path_index = 2
        self.path = 'trained_model10x10_1.h5'

    def get_obs(self):
        return self.agent_location

    def reset(self):
        super().reset(seed=None)
        self.obstacles = [4, 22, 27, 28, 40, 62, 66, 72, 35, 89, 93]
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
                self.path = 'trained_model10x10_' + str(self.path_index) + '.h5'
                self.target_location = self.target_positions[self.target_index]
                self.target_index = (self.target_index + 1) % len(self.target_positions)
                terminated = True
                self.path_index = 1 if self.path_index == 5 else self.path_index
                self.path_index += 1
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

class Agent:
    def __init__(self, state_size, action_size):
        self.memory = deque(maxlen=2500)
        self.learning_rate = 0.001
        self.epsilon = 0.9
        self.max_eps = 0.9
        self.min_eps = 0.01
        self.eps_decay = 0.0047
        self.gamma = 0.9
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon_lst = []
        self.model = self.buildmodel()

    def buildmodel(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def add_memory(self, new_state, reward, done, state, action):
        self.memory.append((new_state, reward, done, state, action))

    def action(self, state):
        if np.random.rand() > self.epsilon:
            return np.random.randint(0, 4)
        return np.argmax(self.model.predict(state, verbose=0))

    def pred(self, state):
        return np.argmax(self.model.predict(state, verbose=0))

    def replay(self, batch_size, episode):
        minibatch = random.sample(self.memory, batch_size)
        for new_state, reward, done, state, action in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(new_state, verbose=0))
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.min_eps:
            self.epsilon = (self.max_eps - self.min_eps) * np.exp(-self.eps_decay * episode) + self.min_eps

def learning(train_episodes, max_steps, batch_size, fps, training):
    env = MyWorld(training=training, render_fps=fps)
    state_size = env.observation_space.n
    action_size = env.action_space.n
    if training:
        agent = Agent(state_size, action_size)
    else:
        agent = Agent(state_size, action_size)
        agent.model = load_model('trained_model10x10_1.h5')
    if training:
        count = 0
        print("Training started.\n")
        for episode in range(train_episodes):
            state = env.reset()
            state_arr = np.zeros(state_size)
            state_arr[state] = 1
            state = np.reshape(state_arr, [1, state_size])
            reward = 0
            done = False
            for t in range(max_steps):
                action = agent.action(state)
                new_state, reward, done, _ = env.step(action)
                new_state_arr = np.zeros(state_size)
                new_state_arr[new_state] = 1
                new_state = np.reshape(new_state_arr, [1, state_size])
                agent.add_memory(new_state, reward, done, state, action)
                state = new_state
                if done:
                    print(
                        f'Episode: {episode:4}/{train_episodes} and step: {t:4}. Eps: {float(agent.epsilon):.2}, reward {reward}')
                    count += 1
                    break
            print(f"Episode: {episode}, count: {count}")
            if len(agent.memory) > batch_size:
                agent.replay(batch_size, episode)
            if episode % 100 == 0:
                agent.model.save('trained_model10x10.h5')
                print("trained_model10x10.h2 save.\n")
        print("Training finished.\n")
        agent.model.save('trained_model10x10.h5')
    else:
        state = env.reset()
        for episode in range(train_episodes):
            state_arr = np.zeros(state_size)
            state_arr[state] = 1
            state = np.reshape(state_arr, [1, state_size])
            done = False
            reward = 0
            state_lst = []
            state_lst.append(state)
            print('******* EPISODE ', episode, ' *******')

            for step in range(max_steps):
                action = agent.pred(state)
                new_state, reward, done, info = env.step(action)
                new_state_arr = np.zeros(state_size)
                new_state_arr[new_state] = 1
                new_state = np.reshape(new_state_arr, [1, state_size])
                state = new_state
                state_lst.append(state)
                if done:
                    agent.model = load_model(env.path)


if __name__ == "__main__":
    learning(500, 1000, 32, 10, training=False)