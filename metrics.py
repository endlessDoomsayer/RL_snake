import numpy as np
import environments_fully_observable

env = environments_fully_observable.BaseEnvironment(n_boards=1, board_size=1)
FRUIT_REWARD = env.FRUIT_REWARD
ATE_HIMSELF_REWARD = env.ATE_HIMSELF_REWARD
HIT_WALL_REWARD = env.HIT_WALL_REWARD

class MetricsTracker:
    def __init__(self, n_boards):
        self.n_boards = n_boards
        
        # Current episode accumulators
        self.average_reward = np.zeros(n_boards, dtype=np.float32)
        self.apples_eaten = np.zeros(n_boards, dtype=np.float32)
        self.body_pieces_eaten = np.zeros(n_boards, dtype=np.float32)
        self.deaths = np.zeros(n_boards, dtype=np.float32)
        self.steps_lived = np.zeros(n_boards, dtype=np.float32)
        
        # History for plotting (population averages over time)
        self.history = {
            "avg_reward": [],
            "avg_apples": [],
            "avg_apple_ratio": [],
            "avg_body_eaten": [],
            "avg_body_ratio": [],
            "avg_deaths": [],
            "avg_death_ratio": []
        }

        self.R_FRUIT = FRUIT_REWARD
        self.R_BODY = ATE_HIMSELF_REWARD
        self.R_WALL = HIT_WALL_REWARD

    def update(self, rewards, active_mask):
        """
        Updates cumulative stats based on current rewards and which boards are active.
        
        Args:
            rewards (np.ndarray): array of rewards for each board.
            active_mask (np.ndarray): boolean array indicating which boards are currently active.
        """
        # Ensure we are working with flat numpy arrays
        if hasattr(rewards, "numpy"): rewards = rewards.numpy()
        rewards = rewards.flatten()
        active_mask = active_mask.flatten()

        # Identify events (only for currently active boards) using a small epsilon for float comparison
        apples_mask = (np.abs(rewards - self.R_FRUIT) < 1e-5) * active_mask
        body_mask = (np.abs(rewards - self.R_BODY) < 1e-5) * active_mask
        death_mask = (np.abs(rewards - self.R_WALL) < 1e-5) * active_mask

        self.average_reward += rewards * active_mask
        # Increment counters
        self.apples_eaten += apples_mask
        self.body_pieces_eaten += body_mask
        self.deaths += death_mask
        self.steps_lived += active_mask

        total_steps = np.sum(self.steps_lived)
        
        # Avoid division by zero
        step_denom = total_steps if total_steps > 0 else 1.0

        # Update history with averages
        self.history["avg_reward"].append(np.mean(self.average_reward))
        self.history["avg_apples"].append(np.mean(self.apples_eaten))
        self.history["avg_apple_ratio"].append(np.sum(self.apples_eaten) / step_denom)
        self.history["avg_body_eaten"].append(np.mean(self.body_pieces_eaten))
        self.history["avg_body_ratio"].append(np.sum(self.body_pieces_eaten) / step_denom)
        self.history["avg_deaths"].append(np.mean(self.deaths))
        self.history["avg_death_ratio"].append(np.sum(self.deaths) / step_denom)

    def get_current_averages(self):
        """Returns the most recent calculated averages."""
        return {k: v[-1] if v else 0 for k, v in self.history.items()}