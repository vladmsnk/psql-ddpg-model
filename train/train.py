import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from environment.environment import Environment
from model.ddpg import DDPG
from replay_memory.replay_memory import PrioritizedReplayMemory


# training DDPG model
class Trainer:
    def __init__(self, environment : Environment, replay_memory : PrioritizedReplayMemory, model : DDPG):
        self.environment = environment
        self.replay_memory = replay_memory
        self.model = model
    

    def train(self, num_episodes, batch_size, gamma, tau):

        for episode in range(num_episodes):
            self.environment.Initialize()
            current_state = self.environment.GetState()
            initial_latency, initial_tps = self.environment.GetReward()

            while True:
                action = self.model.select_action(current_state)
                self.environment.Step(action)
                next_state = self.environment.GetState()





    
        # for each episode
        # initialize the environment calling RecommendationsAPI.InitEnvironment
        # get the initial state calling RecommendationsAPI.GetStates
        # get the initial latency and tps calling RecommendationsAPI.GetRewardMetrics

        # for each step
        # select an action using model DDPG.select_action
        # generate knobs using the action
        # make environment step calling RecommendationsAPI.ApplyActions
        # calculate the reward calling RecommendationsAPI.GetRewardMetrics
        # store the transition in the replay memory
        # if the replay memory has enough samples then sample a batch of transitions else continue
        # get current state, action, reward, next state, done from the batch
        # get current Q values from the critic model
        # calculate the target Q values using the target critic model
        # calculate the expected Q values using the critic model
        # calculate the critic loss
        # calculate policy loss
        # update the target critic model
        # update the target actor model



