import numpy as np
import tensorflow as tf
from collections import deque

def create_logic(grid_size, n_boards):
    return SafeHamiltonianLogic(n_boards, grid_size)


class SafeHamiltonianLogic:
    def __init__(self, n_boards, grid_size):
        self.n_boards = n_boards
        self.size = grid_size
        self.N = grid_size - 2
        
        # Phase Control
        self.on_track = [False for _ in range(n_boards)]
        self.start_node = (1, 1)

        # In a odd-sized grid, the top left corner is not connected to the rest of the cycle, so we need a special maneuver to get there safely
        # Odd cycles don't visit (N,1), even cycles do, so we can use that tile as a trigger for the maneuver
        self.odd_grid_maneuver_state = [0 for _ in range(n_boards)]
        
        # Generate static tile -> action mapping
        self.action_map = self._generate_cycle_map()

    def _generate_cycle_map(self):
        """
        Creates a dictionary mapping (row, col) -> action_idx.
        """
        amap = {}
        N = self.N
        
        if N % 2 == 0:
            # --- EVEN GRID LOGIC ---
            for r in range(1, N + 1):
                for c in range(1, N + 1):
                    if c == 1 and r < N:
                        amap[(r, c)] = 0
                    elif r == N and c < N:
                        amap[(r, c)] = 1
                    else:
                        if c % 2 == 0:
                            if r > 1:
                                amap[(r, c)] = 2
                            else:
                                amap[(r, c)] = 3
                        else:
                            if r < N - 1:
                                amap[(r, c)] = 0
                            else:
                                amap[(r, c)] = 3

            
        else:
            # --- ODD GRID LOGIC ---
            for r in range(1, N):
                if r % 2 == 1: amap[(r, 1)] = 0
                else: amap[(r, 1)] = 1
                if r % 2 == 0: amap[(r, 2)] = 0
                elif r > 1: amap[(r, 2)] = 3
            
            for c in range(1, N):
                amap[(N, c)] = 1
                
            for r in range(2, N + 1):
                amap[(r, N)] = 2
            
            for c in range(N - 1, 2, -1):
                dist_from_right = (N - 1) - c
                
                if dist_from_right % 2 == 0:
                    for r in range(1, N):
                        if r < N - 1: amap[(r, c)] = 0
                        else: amap[(r, c)] = 3
                else:
                    for r in range(1, N):
                        if r > 1: amap[(r, c)] = 2
                        else: amap[(r, c)] = 3

            amap[(1, N)] = 3
            amap[(1, 2)] = 3


        print("Generated Action Map:")
        for r in range(N, 0, -1):
            row_actions = [amap.get((r, c), -1) for c in range(1, N + 1)]
            print(f"Row {r}: {row_actions}")
        return amap

    def get_action(self, state, training=False):
        actions = []
        for b in range(self.n_boards):
            board = state[b]
            head = np.argwhere(board[..., 3] == 1)

            if len(head) == 0:
                actions.append([0])
                continue
            
            curr = tuple(head[0]) # (row, col)

            # Check for Phase Transition
            if not self.on_track[b]:
                if curr == self.start_node:
                    self.on_track[b] = True
                else:
                    # PHASE 1: BFS to find the starting corner safely
                    move = self._bfs_to_start(board, curr)
            
            # Special Maneuver for Odd Grids
            if self.N % 2 != 0 and curr == (self.N - 1, 1):
                if self.odd_grid_maneuver_state[b] == 0:
                    move = 1 # RIGHT to (N-1, 2)
                    self.odd_grid_maneuver_state[b] = 1
                elif self.odd_grid_maneuver_state[b] == 1:
                    move = 0 # UP to (N, 2)
                    self.odd_grid_maneuver_state[b] = 0
            else:
                # Even grid uses standard map
                move = self.action_map.get(curr, 0)
            
            
            actions.append([move])
            
        return tf.constant(actions, dtype=tf.int32), None

    def _bfs_to_start(self, board, start):
        """Finds path to (1, 1) avoiding Wall, Body, and Fruit."""
        queue = deque([(start, [])])
        visited = {start}
        moves = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}

        while queue:
            (r, c), path = queue.popleft()
            if (r, c) == self.start_node:
                return path[0] if path else 0

            for m_idx, (dr, dc) in moves.items():
                nr, nc = r + dr, c + dc
                
                # Boundary Check
                if not (1 <= nr <= self.N and 1 <= nc <= self.N):
                    continue
                # Treat Fruit (1) and Body (2) as Walls during search
                if board[nr, nc, 2] == 1 or board[nr, nc, 1] == 1:
                    continue
                
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [m_idx]))
        
        return 0

    def train_step(self, *args):
        return 0.0