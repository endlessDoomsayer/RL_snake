import tensorflow as tf
import os

class Agent:
    def __init__(self, algorithm_logic, algorithm_name, results_path):
        """
        Wrapper class for the agent that interacts with the environment.

        algorithm_logic: An instance of a class that implements the specific logic for action selection and learning updates.
        """
        self.logic = algorithm_logic

        self.algorithm_name = algorithm_name

        # Folder setup
        self.main_dir = os.path.join(results_path, self.algorithm_name)
        self.checkpoint_dir = os.path.join(self.main_dir, "checkpoints")
        self.stats_dir = os.path.join(self.main_dir, "stats")
        self.train_csv_dir = os.path.join(self.stats_dir, "train")
        self.test_csv_dir = os.path.join(self.stats_dir, "test")

        os.makedirs(self.train_csv_dir, exist_ok=True)
        os.makedirs(self.test_csv_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.best_path = os.path.join(self.main_dir, "best_parameters")
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "checkpoint")

    def get_action(self, state, training=True):
        return self.logic.get_action(state, training)

    def train_step(self, state, action, reward, next_state, done):
        loss = self.logic.train_step(state, action, reward, next_state, done)
        batch_mean_reward = float(tf.reduce_mean(reward))
        return float(loss), batch_mean_reward
    
    def save(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.logic.save_models(folder_path)

    def load(self, folder_path, state_shape, train=False):
        self.logic.load_models(folder_path, state_shape, train)
