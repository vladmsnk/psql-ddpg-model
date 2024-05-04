import numpy as np
import random
import pickle

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
    
    def _update_tree(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
    
    def _retrieve_leaf(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve_leaf(left, s)
        else:
            return self._retrieve_leaf(right, s - self.tree[left])
        
    def add(self, priority, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self._update_tree(tree_idx, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
    
    def update(self, idx, priority):
        tree_idx = idx + self.capacity - 1
        self._update_tree(tree_idx, priority)

    def get_leaf(self, s):
        idx = self._retrieve_leaf(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]
    
    def total_priority(self):
        return self.tree[0]
    
class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        self.tree = SumTree(capacity)
        self.alpha = alpha # Alpha parameter for prioritized experience replay
        self.beta = beta # Beta parameter for prioritized experience replay
        self.beta_increment_per_sampling = beta_increment_per_sampling # Beta increment per sampling
        self.capacity = capacity # Capacity of the replay memory
        self.size = 0 # Number of transitions stored in the replay memory

    def _get_priority(self, error):
        return (np.abs(error) + 1e-6) ** self.alpha

    def add(self, error, sample):
        priority = self._get_priority(error)
        self.tree.add(priority, sample)
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        batch_idx = np.empty(batch_size, dtype=np.int32)
        batch_memory = np.empty(batch_size, dtype=object)
        batch_weights = np.empty(batch_size, dtype=np.float32)
        segment = self.tree.total_priority() / batch_size
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get_leaf(s)
            batch_idx[i] = idx
            batch_memory[i] = data
            batch_weights[i] = (priority / self.tree.total_priority()) ** -self.beta
        return batch_idx, batch_memory, batch_weights
    
    def update(self, idx, error):
        priority = self._get_priority(error)
        self.tree.update(idx, priority)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.tree, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            self.tree = pickle.load(f)
            self.capacity = self.tree.capacity
            self.size = self.tree.data_pointer

