import tensorflow as tf
import numpy as np
import random
import os
from baselogic import BaseLogic
import models.genericmodel as gm

def create_logic(state_shape, action_dim, n_boards, optimizer, gamma=0.9):
    target_model = gm.GenericNetwork(action_dim)
    q_model = gm.GenericNetwork(action_dim)
    return DQNLogic(q_model, target_model, optimizer, gamma)



class DQNLogic(BaseLogic):
    def __init__(self, q_model, target_model, optimizer, 
                 gamma=0.9, 
                 epsilon_start=1.0, 
                 epsilon_min=0.01, 
                 epsilon_decay=0.995,
                 target_update_freq=500):
        
        self.model = q_model           # The "Live" network we train
        self.target_model = target_model # The "Stable" network for targets
        self.optimizer = optimizer
        self.gamma = gamma
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Synchronization logic
        self.target_update_freq = target_update_freq
        self.steps_count = 0
        
        # Initial sync
        self.update_target_network()

    def update_target_network(self):
        """Copies weights from the live model to the target model."""
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state, training=True):
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        q_values = self.model(state_tensor)
        
        n_boards = state.shape[0]
        
        if training and random.random() < self.epsilon:
            # Exploration: Random action for every board in the batch
            actions = np.random.randint(0, q_values.shape[-1], size=(n_boards, 1))
        else:
            # Exploitation: Best action according to Q-values
            actions = tf.argmax(q_values, axis=1)[:, tf.newaxis]
            
        return tf.cast(actions, tf.int32), q_values

    def train_step(self, state, action, reward, next_state, done):
        state = tf.cast(state, dtype=tf.float32)
        next_state = tf.cast(next_state, dtype=tf.float32)
        reward = tf.cast(reward, tf.float32)
        done = tf.cast(done, tf.float32)

        with tf.GradientTape() as tape:
            # 1. Get current Q-values for the states
            current_q_values = self.model(state)
            
            # 2. Extract the Q-value for the specific action taken
            # We use a mask to pick only the Q(s, a) we care about
            action_mask = tf.one_hot(tf.squeeze(action), current_q_values.shape[-1])
            predicted_q = tf.reduce_sum(current_q_values * action_mask, axis=1, keepdims=True)
            
            # 3. Calculate Target Q-values using the Target Network (Bellman Equation)
            # Target = Reward + Gamma * max(Q_target(next_state)) * (1 - done)
            next_q_values = self.target_model(next_state)
            max_next_q = tf.reduce_max(next_q_values, axis=1, keepdims=True)
            
            target_q = reward + (self.gamma * max_next_q * (1.0 - done))
            
            # 4. Loss: Mean Squared Error between predicted and target
            loss = tf.reduce_mean(tf.square(target_q - predicted_q))
            
        # 5. Optimization
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        # 6. Housekeeping: Update epsilon and target network
        self.steps_count += 1
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        if self.steps_count % self.target_update_freq == 0:
            self.update_target_network()
            
        return loss

    def save_models(self, folder_path):
        self.model.save_weights(os.path.join(folder_path, "model.weights.h5"))
        self.target_model.save_weights(os.path.join(folder_path, "target.weights.h5"))

        metadata = np.array([self.epsilon], dtype=np.float32)
        np.save(os.path.join(folder_path, "metadata.npy"), metadata)

    def load_models(self, folder_path, state_shape, train=True):
        dummy_input = tf.zeros((1, *state_shape))
        
        self.model(dummy_input)
        self.model.load_weights(os.path.join(folder_path, "model.weights.h5"))
        
        if train:
            self.target_model(dummy_input)
            self.target_model.load_weights(os.path.join(folder_path, "target.weights.h5"))
            
            metadata_path = os.path.join(folder_path, "metadata.npy")
            if os.path.exists(metadata_path):
                metadata = np.load(metadata_path)
                self.epsilon = metadata[0]
                self.steps_count = int(metadata[1])
            else:
                print("No metadata found, starting with default epsilon and step count.")
                self.epsilon = 1.0
                self.steps_count = 0
            
        
        print("DQN Weights loaded successfully.")