import numpy as np
import tensorflow as tf


def create_logic(state_shape, action_dim, n_boards, optimizer, gamma=0.9):
    return ZigZagLogic(n_boards, state_shape[0])


class ZigZagLogic:
    def __init__(self, n_boards, board_size):
        self.n_boards = n_boards
        self.size = board_size
        self.min_p = 1              # First playable index
        self.max_p = board_size - 2 # Last playable index

    def get_action(self, state, training=False):
        actions = []
        for b in range(self.n_boards):
            head_coords = np.argwhere(state[b][..., 3] == 1)
            if len(head_coords) == 0:
                actions.append([0]); continue
            
            r, c = head_coords[0]
            
            # 1. Highway: Left-most column goes DOWN to the start
            if c == self.min_p and r > self.min_p:
                move = 2 # DOWN
            
            # 2. Bottom Row: Connect Highway to the Zig-Zag area
            elif r == self.min_p and c < self.max_p:
                move = 1 # RIGHT
            
            # 3. Zig-Zag Area (Columns from max_p down to min_p + 1)
            else:
                # We are in the area where we zig-zag vertically
                # We use the column index to decide direction
                # For S=7, max_p=5. Cols 5, 3 are 'UP', Cols 4, 2 are 'DOWN'
                
                dist_from_right = self.max_p - c
                
                if dist_from_right % 2 == 0: # Even distance (Col 5, 3, 1...)
                    if r < self.max_p:
                        move = 0 # UP
                    else:
                        move = 3 # LEFT
                else: # Odd distance (Col 4, 2...)
                    if r > self.min_p + 1:
                        move = 2 # DOWN
                    else:
                        move = 3 # LEFT

            actions.append([move])
            
        return tf.constant(actions, dtype=tf.int32), None
    
    def train_step(self, *args):
        # Baseline doesn't learn
        return 0.0