import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import shutil
from ppo import PPO
from collections import deque
import gym
from gym import spaces
import pygame
import numpy as np

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

    def get_obs(self):
        return self.agent_location

    def reset(self):
        super().reset(seed=None)
        self.obstacles = [4, 22, 27, 28, 40, 62, 66, 72, 35, 89]
        self.target_location = 99
        # self.target_location = self.target_positions[self.target_index]
        # self.target_index = (self.target_index + 1) % len(self.target_positions)
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
                terminated = True
            else:
                reward = -0.1

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

dir = "/home/dv/gym-examples/code/mnt/"

def save_results(max_ep_len):
    print("============================================================================================")

    K_epochs = 80
    eps_clip = 0.2
    gamma = 0.99

    lr_actor = 0.0003
    lr_critic = 0.001
    env_name = 'MyWorld'
    env = MyWorld()

    state_dim = 1
    action_dim = env.action_space.n
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)

    reward_path = dir + "PPO_results/"
    if not os.path.exists(reward_path):
        os.mkdir(reward_path)
    reward_path1 = reward_path + f"test1_{test_num}_episode{total_test_episodes}" + ".pth"
    reward_path = reward_path + f"test{test_num}_episode{total_test_episodes}" + ".pth"

    res_r = []

    test_running_reward = 0
    last_avg_reward = 0
    last_reward = 0

    for ep in range(1, total_test_episodes + 1):
        ep_reward = 0
        state = env.reset()
        flag = 0
        for t in range(1, max_ep_len + 1):
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward

            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            if ep % log_freq == 0:
                avg_reward = test_running_reward / ep
                res_r.append({'episode': ep, 'time_step': t, 'mean_return': avg_reward})

            if done:
                break

        test_running_reward += ep_reward

        if ep % save_model_freq == 0:
            val_reward = ep_reward / t
            avg_reward2 = test_running_reward / ep
            if (val_reward > last_reward) or (
                    (val_reward >= last_reward) and (avg_reward2 >= last_avg_reward)):
                print("current_episode_mean_reward : ", val_reward)
                print("Test average reward: ", avg_reward2)
                print(
                    "--------------------------------------------------------------------------------------------")
                print("saving model at : " + reward_path)
                ppo_agent.save(reward_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print(
                    "--------------------------------------------------------------------------------------------")

                last_reward = val_reward
                last_avg_reward = avg_reward2

        if ep % update_freq == 0:
            ppo_agent.update()

        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0
        torch.cuda.empty_cache()

    env.close()

    print("============================================================================================")

    print("total number of frames / timesteps / images saved : ", t)

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")

    ppo_agent.save(reward_path1)
    # torch.save(ppo_agent.policy.state_dict(), reward_path)

def learning(max_ep_len, fps, training):
    K_epochs = 80  # политика обновления для K эпох
    eps_clip = 0.2  # параметр clip для PPO
    gamma = 0.99  # коэффициент дисконтирования

    lr_actor = 0.0003  # скорость обучения актера
    lr_critic = 0.001  # скорость обучения для критика

    env = MyWorld(training=training, render_fps=fps)

    # измерение пространства состояний
    state_dim = 1

    # измерение пространства действия
    action_dim = env.action_space.n

    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)

    # Загрузка обученной модели
    checkpoint_path = dir + "PPO_results/test5_episode10000.pth"
    ppo_agent.load(checkpoint_path)

    try_time = 0
    reach_goal_time = 0

    for _ in range(total_test_episodes):
        ep_reward = 0
        state = env.reset()

        for t in range(1, max_ep_len + 1):
            # Действие выбирается обученной моделью
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward

            if done:
                break

        print('Episode: {} \t\t Reward: {}'.format(_, round(ep_reward, 2)))


if __name__ == '__main__':
    has_continuous_action_space = False
    max_ep_len = 100
    total_test_episodes = int(1000)

    test_num = 6

    action_std = None

    update_freq = int(100)
    log_freq = int(200)
    save_model_freq = int(100)

    random_seed = 0
    run_num_pretrained = 0
    save_num_pretrained = 0
    start_time = datetime.now().replace(microsecond=0)
    pretrained = False

    learning(max_ep_len, 50, False)
    # save_results(max_ep_len)
