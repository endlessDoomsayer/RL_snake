import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, algorithm_logic, algorithm_name, results_path, save_frequency):
        """
        Wrapper class for the agent that interacts with the environment.

        algorithm_logic: An instance of a class that implements the specific logic for action selection and learning updates.
        """
        self.logic = algorithm_logic

        self.algorithm_name = algorithm_name
        self.save_frequency = save_frequency

        # Folder setup
        self.main_dir = os.path.join(results_path, self.algorithm_name)
        self.checkpoint_dir = os.path.join(self.main_dir, "checkpoints")
        self.stats_dir = os.path.join(self.main_dir, "stats")

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.stats_dir, exist_ok=True)
        
        self.best_path = os.path.join(self.main_dir, "best_parameters")
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "checkpoint")

        self.best_reward = -float('inf')
        self.loss_history = []
        self.reward_history = []
        self.total_steps = 0

    def get_action(self, state, training=True):
        # Always returns action indices for the environment
        # and logits for the loss function, if training is True
        return self.logic.get_action(state, training)

    def train_step(self, state, action, reward, next_state, done):
        # Delegates the specific math to the logic class
        loss = self.logic.train_step(state, action, reward, next_state, done)

        self.total_steps += 1
        self.reward_history.append(float(tf.reduce_mean(reward)))
        self.loss_history.append(float(loss))

        if len(self.reward_history) >= 100:
            moving_avg_reward = np.mean(self.reward_history[-100:])
            
            if moving_avg_reward > self.best_reward:
                self.best_reward = moving_avg_reward
                self.save(self.best_path) # Save to main folder
        
        # Periodic Checkpoint
        if self.total_steps % self.save_frequency == 0:
            path = f"{self.checkpoint_prefix}_step_{self.total_steps}"
            self.save(path) # Save to checkpoints folder
            
        return loss
    
    def save(self, folder_path):
        """
        Saves parameters to the specified folder.
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.logic.save_models(folder_path)

    def load(self, folder_path, state_shape, train=False):
        """
        Loads parameters from the specified folder.
        state_shape: (height, width, channels) e.g., (7, 7, 4)
        train: if True, loads weights for training, else loads only the model weights for inference
        """
        self.logic.load_models(folder_path, state_shape, train)

    def save_results_plots(self):
        """Generates and saves two plots: Loss and Reward."""
        # Plot Loss
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history, label='Loss', color='red')
        plt.title(f'Training Loss - {self.total_steps} iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(self.stats_dir, "loss_plot.png"))
        plt.close()

        # Plot Reward (using a moving average to make it readable)
        plt.figure(figsize=(10, 5))
        plt.plot(self.reward_history, label='Reward', color='blue')
        plt.title(f'Training Reward - {self.total_steps} iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Avg Reward')
        plt.savefig(os.path.join(self.stats_dir, "reward_plot.png"))
        plt.close()
        print(f"Graphics saved in {self.stats_dir}")