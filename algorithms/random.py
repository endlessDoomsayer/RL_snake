
import tensorflow as tf
import numpy as np


def create_logic(state_shape, action_dim, n_boards, optimizer, gamma=0.9):
    return SafeRandomLogic(n_boards)

class SafeRandomLogic:
    def __init__(self, n_boards):
        self.n_boards = n_boards

    def get_action(self, state, training=False):
        """
        Picks a random action.
        Priority:
        1. Randomly from moves that avoid walls AND body.
        2. If trapped by body, randomly from moves that hit body but avoid walls.
        3. If trapped by walls, fallback to default.
        """
        # state shape: (n_boards, size, size, 4)
        # Channels: 0:Empty, 1:Fruit, 2:Body, 3:Head
        size = state.shape[1]
        final_actions = []

        for b in range(self.n_boards):
            board = state[b]
            
            # 1. Locate the head [row, col]
            head_coords = np.argwhere(board[..., 3] == 1)
            if len(head_coords) == 0:
                final_actions.append([0]) 
                continue
                
            r, c = head_coords[0]
            
            # UP=0 (r+1), RIGHT=1 (c+1), DOWN=2 (r-1), LEFT=3 (c-1)
            moves = {
                0: (1, 0),  # UP
                1: (0, 1),  # RIGHT
                2: (-1, 0), # DOWN
                3: (0, -1)  # LEFT
            }
            
            safe_moves = [] # No wall, no body
            body_moves = [] # No wall, but hits body
            
            # 2. Evaluate each direction
            for move_idx, (dr, dc) in moves.items():
                nr, nc = r + dr, c + dc
                
                # A. Check for Walls (Indices 0 and size-1 are walls)
                # If it hits a wall, we ignore this move entirely (Terminal)
                if nr <= 0 or nr >= size - 1 or nc <= 0 or nc >= size - 1:
                    continue
                
                # B. Check for Body (Channel 2)
                if board[nr, nc, 2] == 1:
                    body_moves.append(move_idx)
                else:
                    safe_moves.append(move_idx)
            
            # 3. Pick the action based on priority
            if safe_moves:
                # Best case: random safe move
                action = np.random.choice(safe_moves)
            elif body_moves:
                # Second best: eat himself to stay in the game
                action = np.random.choice(body_moves)
            else:
                # Worst case: trapped by walls, pick UP and die
                action = 0 
                
            final_actions.append([action])

        return tf.constant(final_actions, dtype=tf.int32), None

    def train_step(self, *args, **kwargs):
        # Baselines do not learn, return 0 loss
        return 0.0