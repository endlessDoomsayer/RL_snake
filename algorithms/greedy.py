import tensorflow as tf
import numpy as np


HEAD = 4
BODY = 3
FRUIT = 2
EMPTY = 1
WALL = 0


def create_logic(state_shape, action_dim, n_boards, optimizer=None, gamma=0.9):
    return GreedyLogic(n_boards)



class GreedyLogic:
    def __init__(self, n_boards):
        self.n_boards = n_boards

    def get_action(self, state, training=False):
        # state: (n_boards, size, size, 4)
        # Channels: 0:Empty, 1:Fruit, 2:Body, 3:Head
        # Wall is where all channels are 0 (due to categorical slice [..., 1:])
        
        actions = []
        size = state.shape[1]
        
        for b in range(self.n_boards):
            board = state[b]
            head_coords = np.argwhere(board[..., 3] == 1)
            fruit_coords = np.argwhere(board[..., 1] == 1)
            
            if len(head_coords) == 0 or len(fruit_coords) == 0:
                actions.append([0])
                continue
                
            hr, hc = head_coords[0]
            fr, fc = fruit_coords[0]
            
            # Possible moves and their coordinate offsets
            # UP=0 (r+1), RIGHT=1 (c+1), DOWN=2 (r-1), LEFT=3 (c-1)
            moves = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
            
            best_move = 0
            min_dist = float('inf')
            valid_move_found = False
            
            # Filter moves to prevent bumping into walls or body
            safe_moves = []
            for m_idx, (dr, dc) in moves.items():
                nr, nc = hr + dr, hc + dc
                
                # 1. Check Wall (Indices 0 and size-1)
                if nr <= 0 or nr >= size - 1 or nc <= 0 or nc >= size - 1:
                    continue
                
                # 2. Check Body (Channel 2)
                if board[nr, nc, 2] == 1:
                    continue
                
                safe_moves.append(m_idx)
            
            if not safe_moves:
                # If trapped, just go UP and die gracefully
                actions.append([0])
                continue

            # Among safe moves, pick the one closest to fruit
            for m_idx in safe_moves:
                dr, dc = moves[m_idx]
                nr, nc = hr + dr, hc + dc
                dist = abs(nr - fr) + abs(nc - fc)
                if dist < min_dist:
                    min_dist = dist
                    best_move = m_idx
            
            actions.append([best_move])
            
        return tf.constant(actions, dtype=tf.int32), None
    
    def train_step(self, *args):
        # Baseline doesn't learn
        return 0.0