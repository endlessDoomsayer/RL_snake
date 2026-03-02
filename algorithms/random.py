
import tensorflow as tf
import numpy as np


def create_logic(state_shape, action_dim, n_boards, optimizer, gamma=0.9):
    return RandomLogic(n_boards)

class RandomLogic:
    def __init__(self, n_boards):
        self.n_boards = n_boards

    def get_action(self, state, training=False):
        # Simply return random integers 0-3 for each board
        actions = np.random.randint(0, 4, size=(self.n_boards, 1))
        return tf.constant(actions, dtype=tf.int32), None

    def train_step(self, *args):
        return 0 # No learning