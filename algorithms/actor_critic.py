import os

import tensorflow as tf
from baselogic import BaseLogic
from genericmodel import GenericNetwork

def create_logic(state_shape, action_dim, n_boards, optimizer, gamma=0.9):
    actor_model = GenericNetwork(action_dim)
    critic_model = GenericNetwork(1)

    return ActorCriticLogic(actor_model, critic_model, optimizer, gamma)


class ActorCriticLogic(BaseLogic):
    def __init__(self, actor, critic, optimizer, gamma=0.9, entropy_beta=0.05):
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.gamma = gamma
        self.entropy_beta = entropy_beta

    def get_action(self, state, training=True):
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        logits = self.actor(state_tensor)
        
        if training:
            action = tf.random.categorical(logits, num_samples=1)
        else:
            action = tf.argmax(logits, axis=1)
        return action, logits

    def train_step(self, state, action, reward, next_state, done):
        state = tf.cast(state, dtype=tf.float32)
        next_state = tf.cast(next_state, dtype=tf.float32)
        reward = tf.cast(reward, tf.float32)
        done = tf.cast(done, tf.float32)
        
        with tf.GradientTape(persistent=True) as tape:
            logits = self.actor(state)
            values = self.critic(state)
            next_values = self.critic(next_state)
            
            # General AC math
            target = reward + (self.gamma * next_values * (1.0 - done))
            advantage = target - values
            
            action_masks = tf.one_hot(tf.squeeze(action), logits.shape[-1])
            log_probs = tf.reduce_sum(tf.nn.log_softmax(logits) * action_masks, axis=1, keepdims=True)
            
            actor_loss = -tf.reduce_mean(log_probs * tf.stop_gradient(advantage))
            
            # Entropy
            probs = tf.nn.softmax(logits)
            entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-9), axis=1)
            actor_loss -= self.entropy_beta * tf.reduce_mean(entropy)

            critic_loss = tf.reduce_mean(tf.square(advantage))
            
        # Standardized Gradient Application
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        del tape
        
        all_grads = actor_grads + critic_grads
        all_vars = self.actor.trainable_variables + self.critic.trainable_variables
        self.optimizer.apply_gradients(zip(all_grads, all_vars))

        return critic_loss # Or (actor_loss + critic_loss)

    def save_models(self, folder_path):
        self.actor.save_weights(os.path.join(folder_path, "actor.weights.h5"))
        self.critic.save_weights(os.path.join(folder_path, "critic.weights.h5"))

    def load_models(self, folder_path, state_shape, train=True):
        dummy_input = tf.zeros((1, *state_shape))
        self.actor(dummy_input)
        self.actor.load_weights(os.path.join(folder_path, "actor.weights.h5"))
        if train and self.critic is not None:
            self.critic(dummy_input)
            self.critic.load_weights(os.path.join(folder_path, "critic.weights.h5"))