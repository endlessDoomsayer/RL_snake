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
            
            # UP=0 (r+1), RIGHT=1 (c+1), DOWN=2 (r-1), LEFT=3 (c-1)
            moves = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
            
            safe_moves = []  # Avoids walls AND body
            body_moves = []  # Avoids walls, but hits body
            
            for m_idx, (dr, dc) in moves.items():
                nr, nc = hr + dr, hc + dc
                
                # 1. Strict Wall Check (Terminal)
                # Never move into index 0 or size-1
                if nr <= 0 or nr >= size - 1 or nc <= 0 or nc >= size - 1:
                    continue
                
                # 2. Body Check
                if board[nr, nc, 2] == 1:
                    body_moves.append(m_idx)
                else:
                    safe_moves.append(m_idx)

            # --- DECISION LOGIC ---
            
            selected_move = 0
            best_dist = float('inf')

            # Preference 1: Move to an empty cell or fruit
            if safe_moves:
                for m_idx in safe_moves:
                    dr, dc = moves[m_idx]
                    dist = abs((hr + dr) - fr) + abs((hc + dc) - fc)
                    if dist < best_dist:
                        best_dist = dist
                        selected_move = m_idx
            
            # Preference 2: If trapped by body, pick body cell closest to fruit
            elif body_moves:
                for m_idx in body_moves:
                    dr, dc = moves[m_idx]
                    dist = abs((hr + dr) - fr) + abs((hc + dc) - fc)
                    if dist < best_dist:
                        best_dist = dist
                        selected_move = m_idx
            
            # Preference 3: Truly trapped by walls (unlikely), go UP
            else:
                selected_move = 0
            
            actions.append([selected_move])
            
        return tf.constant(actions, dtype=tf.int32), None
    
    def train_step(self, *args):
        # Baseline doesn't learn
        return 0.0