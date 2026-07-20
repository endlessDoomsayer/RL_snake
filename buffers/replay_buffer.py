import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, state_shape):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate NumPy arrays
        self.states = np.zeros((capacity, state_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_shape), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def push(self, s, a, r, ns, d):
        n = s.shape[0]
        indices = np.arange(self.ptr, self.ptr + n) % self.capacity
        
        self.states[indices] = s
        self.actions[indices] = a
        self.rewards[indices] = r
        self.next_states[indices] = ns
        self.dones[indices] = d
        
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (self.states[idx], self.actions[idx], self.rewards[idx], 
                self.next_states[idx], self.dones[idx])