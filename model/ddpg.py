import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Actor, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.fc1 = nn.Linear(n_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, n_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.uniform_(-init_w, init_w)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    def __init__(self, n_states, n_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden1)
        self.fc2 = nn.Linear(hidden1+n_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x, u):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(torch.cat([x, u], 1)))
        x = self.fc3(x)
        return x
    

class DDPG(object):
    def __init__(self, n_states, n_actions, hidden1=400, hidden2=300, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=1e-3):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau

        self.actor = Actor(n_states, n_actions, hidden1, hidden2)
        self.actor_target = Actor(n_states, n_actions, hidden1, hidden2)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(n_states, n_actions, hidden1, hidden2)
        self.critic_target = Critic(n_states, n_actions, hidden1, hidden2)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def select_action(self, state):
        self.actor.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            action = self.actor(state).squeeze(0).numpy()
        self.actor.train()
        return action
    
    def update(self, batch):
        states, actions, rewards, next_states, dones = batch

        # Update critic
        self.critic_optimizer.zero_grad()
        next_actions = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, next_actions.detach())
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic(states, actions)
        critic_loss = nn.MSELoss()(Q_expected, Q_targets)
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        self.actor_optimizer.zero_grad()
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self.soft_update(self.critic_target, self.critic)
        self.soft_update(self.actor_target, self.actor)





     