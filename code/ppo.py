import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gym
from gym import spaces
import pygame
import warnings
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal

warnings.filterwarnings("ignore")

# set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Класс для буфера опыта
class RolloutBuffer:
    def __init__(self):
        self.actions = [] # список для хранения действий
        self.states = [] # список для хранения состояний
        self.logprobs = [] # список для хранения журнала вероятностей действий
        self.rewards = [] # список для хранения вознаграждений
        self.state_values = [] # список для хранения оценок состояний критика
        self.is_terminals = [] # список, указывающий, является ли каждое состояние конечным состоянием

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

# Архитектура политики агента
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        # первый полностью связный слой с входным размером state_dim и выходным размером 64.
        self.fc1 = nn.Linear(state_dim, 64)
        # второй полносвязный слой с входным размером 64 и выходным размером 64.
        self.fc2 = nn.Linear(64, 64)
        # третий полносвязный слой с входным размером 64 и выходным размером action_dim
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x): # реализует логику прямого прохода нейронной сети
        # применяет первый полностью подключенный слой к входным данным x
        x = self.fc1(x)
        # Применяет функцию активации гиперболического тангенса к выходным данным первого слоя
        x = torch.tanh(x)
        # применяет второй полносвязный слой
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        # применяет функцию активации softmax к последнему измерению, преобразуя выходные данные сети в распределение вероятностей по действиям
        x = torch.softmax(x, dim=-1)
        return x


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()


        self.actor = Actor(state_dim, action_dim).to(device)

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        # Логарифм вероятности выбора действия и оценку состояния критика на основе входного состояния
        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        #  Оценивает логарифм вероятности выбора действия, оценку состояния критика и энтропию распределения вероятностей политики
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip):

        self.gamma = gamma
        self.eps_clip = eps_clip # максимальное отклонение для суррогатных потерь
        self.k_epochs = k_epochs # количество эпох оптимизации политики на одном и том же наборе данных

        self.buffer = RolloutBuffer() # прогноз для накопления опыта в процессе взаимодействия с окружением

        # представляет собой объединенную нейронную сеть для аппроксимации и обновления политики и значения состояния-ценности
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        # прогноз для обновления весов политики и критика
        self.optimizer = optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        #  копия существующей политики, которая используется при вычислении суррогатных потерь
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        # Загрузка весов современной политики из текущей политики
        self.policy_old.load_state_dict(self.policy.state_dict())

        # MseLoss- критерии среднеквадратичной ошибки, являющиеся суррогатными потерями для значения состояния-ценности
        self.mse_loss = nn.MSELoss()

    # метод для выбора действия агента на основе текущего состояния
    def select_action(self, state):
        # Контекстный блок для отключения вычисления градиентов во время выбора действия
        with torch.no_grad():
            # преобразование входного состояния в тензор и вызов метода act старой политики для выбора действия.
            state = torch.tensor(state, dtype=torch.float).to(device)
            state = state.unsqueeze(0)

            action, action_logprob, state_val = self.policy_old.act(state)

        # Добавление текущего состояния, выбранного действия,
        # логарифма вероятности выбора действия и значения состояния в буфер роллаута
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        # Возвращение выбранного действия в виде скаляра (item)
        return action.item()

    def update(self):
        # Оценка доходности по методу Монте-Карло
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Нормализация вознаграждений
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # преобразовать список в тензор
        old_states = torch.stack(self.buffer.states, dim=0).detach().to(device)
        old_actions = torch.stack(self.buffer.actions, dim=0).detach().to(device)
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach().to(device)
        old_state_values = torch.stack(self.buffer.state_values, dim=0).detach().to(device)

        advantages = rewards.detach() - old_state_values.detach()

        # Оптимизировать политику для K эпох
        for _ in range(self.k_epochs):
            # Оценка старых действий и ценностей
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # сопоставьте измерения тензора state_values с тензором вознаграждений
            state_values = torch.squeeze(state_values)

            # Нахождение соотношения (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Поиск суррогатной потери
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # окончательная потеря отсеченного целевого PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, rewards) - 0.01 * dist_entropy

            # сделайте шаг по градиенту
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Скопируйте новые веса в старую политику
        self.policy_old.load_state_dict(self.policy.state_dict())

        # очистить буфер
        self.buffer.clear()

    # Определение метода сохранения весов политики
    def save(self, checkpoint_path):
        # Сохранение весов старой политики в указанный путь с использованием функции torch.save
        torch.save(self.policy_old.state_dict(), checkpoint_path)
        # state_dict() используется для получения словаря состояния модели, который содержит веса и параметры

    # Определение метода загрузки весов политики из сохраненного файла
    def load(self, checkpoint_path):
        # Загрузка состояния старой политики из сохраненных весов с использованием функции torch.load
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        #  Загрузка состояния текущей политики из сохраненных весов с использованием той же процедуры
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))