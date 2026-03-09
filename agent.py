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

        #training history
        self.best_reward = -float('inf')
        self.loss_history = []
        self.reward_history = []
        self.total_steps = 0
        
        #evaluation history
        self.eval_cum_reward_evo = None
        self.eval_apple_ratio_evo = None

    def get_action(self, state, training=True):
        # Always returns action indices for the environment
        # and logits for the loss function, if training is True
        return self.logic.get_action(state, training)

    def train_step(self, state, action, reward, next_state, done):
        """
        Performs the math but DOES NOT update history.
        Returns: (loss, batch_mean_reward)
        """
        loss = self.logic.train_step(state, action, reward, next_state, done)
        batch_mean_reward = float(tf.reduce_mean(reward))
        return float(loss), batch_mean_reward
    
    def record_training_iteration(self, avg_loss, avg_reward):
        """
        Appends the pre-averaged stats to history and handles saving.
        """
        self.total_steps += 1 # We count iterations as "steps" now
        self.loss_history.append(avg_loss)
        self.reward_history.append(avg_reward)

        # Handle Best Model Saving (using the averaged iteration reward)
        if len(self.reward_history) >= 100:
            moving_avg_reward = np.mean(self.reward_history[-100:])
            if moving_avg_reward > self.best_reward:
                self.best_reward = moving_avg_reward
                self.save(self.best_path)
        
        # Periodic Checkpoint
        if self.total_steps % self.save_frequency == 0:
            path = f"{self.checkpoint_prefix}_step_{self.total_steps}"
            self.save(path)

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

    def record_eval_data(self, cum_reward_evo, apple_ratio_evo):
        """
        cum_reward_evo: A list of cumulative rewards for each step.
        apple_ratio_evo: A list of apple-to-step ratios for each step.
        """
        self.eval_cum_reward_evo = cum_reward_evo
        self.eval_apple_ratio_evo = apple_ratio_evo
    
    def save_eval_stats_binary(self, cum_reward_evo, apple_ratio_evo, filename="eval_stats"):
        """
        Saves the evolution data in format (.npz).
        """
        save_path = os.path.join(self.stats_dir, f"{filename}.npz")
        
        # Save multiple arrays into one compressed file
        np.savez_compressed(
            save_path,
            cum_reward_evolution=np.array(cum_reward_evo),
            apple_ratio_evolution=np.array(apple_ratio_evo),
            metadata={
                "algorithm": self.algorithm_name,
                "total_steps": len(cum_reward_evo)
            }
        )
        print(f"Evaluation data saved to: {save_path}")

    def save_evaluation_plots(self):
        """Generates plots for the last evaluation run."""
        if self.eval_cum_reward_evo is None:
            return

        # Plot 1: Cumulative Reward Evolution
        plt.figure(figsize=(10, 5))
        plt.plot(self.eval_cum_reward_evo, color='purple', label='Population Avg')
        plt.title('Evolution of Average Cumulative Reward')
        plt.xlabel('Step')
        plt.ylabel('Cumulative Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.stats_dir, "eval_cum_reward_evolution.png"))
        plt.close()

        # Plot 2: Apple Efficiency Evolution
        plt.figure(figsize=(10, 5))
        plt.plot(self.eval_apple_ratio_evo, color='green', label='Population Efficiency')
        plt.title('Evolution of Apple/Step Ratio')
        plt.xlabel('Step')
        plt.ylabel('Apples / Steps')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.stats_dir, "eval_apple_efficiency_evolution.png"))
        plt.close()

    def save_training_stats_binary(self):
        """Saves numerical training history to an .npz file."""
        save_path = os.path.join(self.stats_dir, "training_stats.npz")
        np.savez_compressed(
            save_path,
            loss_history=np.array(self.loss_history),
            reward_history=np.array(self.reward_history),
            total_steps=self.total_steps
        )
        print(f"Training stats saved to {save_path}")

    def save_training_plots(self):
        window = 100
        def moving_average(data, window_size):
            if len(data) < window_size: return data
            return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

        # Loss Plot
        plt.figure(figsize=(12, 6))
        plt.plot(self.loss_history, color='red', alpha=0.3, label='Loss')
        plt.plot(range(window-1, len(self.loss_history)), moving_average(self.loss_history, window), color='red', label='Trend')
        plt.title('Training Loss (Averaged over Epochs per Iteration)')
        plt.legend(); plt.savefig(os.path.join(self.stats_dir, "loss_plot.png")); plt.close()

        # Reward Plot
        plt.figure(figsize=(12, 6))
        plt.plot(self.reward_history, color='blue', alpha=0.3, label='Reward')
        plt.plot(range(window-1, len(self.reward_history)), moving_average(self.reward_history, window), color='blue', label='Trend')
        plt.title('Training Reward (Averaged over Epochs per Iteration)')
        plt.legend(); plt.savefig(os.path.join(self.stats_dir, "reward_plot.png")); plt.close()

    def load_eval_stats_binary(self, filename="eval_stats"):
        """
        Helper to reload the data later for comparison.
        """
        path = os.path.join(self.stats_dir, f"{filename}.npz")
        data = np.load(path, allow_pickle=True)
        return data['cum_reward_evolution'], data['apple_ratio_evolution']