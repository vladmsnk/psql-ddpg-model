import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from environment.environment import Environment
from model.ddpg import DDPG
from replay_memory.replay_memory import PrioritizedReplayMemory

instance_name = ""

# training DDPG model
class Trainer:
    def __init__(self, environment : Environment, replay_memory : PrioritizedReplayMemory, model : DDPG):
        self.environment = environment
        self.replay_memory = replay_memory
        self.model = model


    def calculate_reward(self, initial_latency, initial_tps,previous_latency, previous_tps, current_latency, current_tps):
        pass
    
    def update_knob_values(knobs, actions):
        """
        Adjusts the knob settings based on a list of action values from an actor network.

        Parameters:
            knobs (dict): A dictionary where each key is the knob name and each value is another
                        dictionary containing the default value, min_value, and max_value of the knob.
            actions (list): A list of action values, each between [0, 1], corresponding to each knob.

        Returns:
            dict: A dictionary with the same structure as `knobs`, but with updated values.
        """

        updated_knobs = {}
        knob_keys = list(knobs.keys())

        for index, action in enumerate(actions):
            knob_key = knob_keys[index]
            knob_info = knobs[knob_key]
            min_val = knob_info['min_value']
            max_val = knob_info['max_value']
            # Linearly interpolate the new knob value based on the action
            new_value = min_val + (max_val - min_val) * action
            updated_knobs[knob_key] = {
                'value': new_value,
                'min_value': min_val,
                'max_value': max_val
            }

        return updated_knobs



    def train(self, num_episodes, batch_size, gamma, tau):

        for episode in range(num_episodes):
            self.environment.init_environment()

            current_state = self.environment.get_states()

            initial_latency, initial_tps = self.environment.get_reward_metrics()

            while True:
                # action = self.model.select_action(current_state)

                # knobs_to_set = self.update_knob_values(current_state, action)

                # self.environment.apply_actions(instance_name, knobs_to_set)

                reward = self.environment.get_reward_metrics(instance_name=instance_name)





    
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



# # Example knobs and actions
# knobs = {
#     "autovacuum_max_workers": {"value": 3, "min_value": 1, "max_value": 262143},
#     "checkpoint_completion_target": {"value": 0.9, "min_value": 0, "max_value": 1},
#     "checkpoint_timeout": {"value": 300, "min_value": 30, "max_value": 86400},
#     "effective_cache_size": {"value": 524288, "min_value": 1, "max_value": 2.14748365e+09},
#     "maintenance_work_mem": {"value": 65536, "min_value": 1024, "max_value": 2.14748365e+09},
#     "max_connections": {"value": 100, "min_value": 1, "max_value": 262143},
#     "shared_buffers": {"value": 16384, "min_value": 16, "max_value": 1.07374182e+09},
#     "wal_buffers": {"value": 512, "min_value": -1, "max_value": 262143},
#     "wal_writer_delay": {"value": 200, "min_value": 1, "max_value": 10000},
#     "work_mem": {"value": 4096, "min_value": 60, "max_value": 2.14748365e+09}
# }

# # Example actions from the actor network
# actions = [0.01, 0.09, 0.05, 0.075, 0.03, 0.06, 0.02, 0.04, 0.08, 0.0001]

# # Update knobs based on actions
# updated_knobs = update_knob_values(knobs, actions)

# # Print updated knobs
# for key, value in updated_knobs.items():
#     print(f"{key}: {value['value']}")