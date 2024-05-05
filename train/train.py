import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from environment.environment import Environment
from model.ddpg import DDPG
from replay_memory.replay_memory import PrioritizedReplayMemory

instance_name = "test"

# training DDPG model
class Trainer:
    def __init__(self, model : DDPG, environment : Environment, knobs):
        self.environment = environment
        self.model = model
        self.knobs = knobs

    
    @staticmethod
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



    def train(self, num_episodes, batch_size):
        fine_state_actions = []


        for episode in range(num_episodes):
            print(f"Episode {episode + 1}/{num_episodes}")

            self.environment.init_environment(instance_name)

            current_state = list(self.environment.get_states(instance_name).metrics)

            initial_latency, initial_tps = self.environment.get_reward_metrics(instance_name)
            previous_latency, previous_tps = initial_latency, initial_tps

            i = 0
            while i < 15:
                
                action = self.model.choose_action(current_state)

                knobs_to_set = self.update_knob_values(self.knobs, action)

                next_state, reward, ext_metrics = self.environment.step(instance_name=instance_name, knobs=knobs_to_set, initial_latency=initial_latency, initial_tps=initial_tps, previous_latency=previous_latency, previous_tps=previous_tps)

                self.model.replay_memory.add(reward, (current_state, action, reward, next_state, False))

                if reward > 5:
                    fine_state_actions.append((next_state, action))


                self.model.add_sample(current_state, action, reward, next_state, False)
    
                current_state = next_state
                previous_tps, previous_latency = ext_metrics

                if len(self.model.replay_memory) > batch_size:
                    self.model.update()

                if self.environment.perofrmance_increased:
                    print(f"Performance increased tps {ext_metrics.tps} latency {ext_metrics.latency}")
                    break



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