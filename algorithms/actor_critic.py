import os
import tensorflow as tf
from baselogic import BaseLogic
from models.MLP import MLP

def create_logic(action_dim, encoder, optimizer, gamma=0.9):
    actor_model = MLP(output_dim=action_dim)
    critic_model = MLP(output_dim=1)
    return ActorCriticLogic(actor_model, critic_model, encoder, optimizer, gamma)

class ActorCriticLogic(BaseLogic):
    def __init__(self, actor, critic, encoder, optimizer, gamma=0.9, entropy_beta=0.01):
        super().__init__(encoder)

        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.gamma = gamma
        self.entropy_beta = entropy_beta

    def get_action(self, state, training=True):
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        logits = self.actor(state_tensor)
        
        if training:
            # Sample from the distribution for exploration
            action = tf.random.categorical(logits, num_samples=1)
        else:
            # Greedy for evaluation
            action = tf.argmax(logits, axis=1)[:, tf.newaxis]
            
        return tf.cast(action, tf.int32), logits

    def train_step(self, state, action, reward, next_state, done):

        state = tf.cast(state, dtype=tf.float32)
        next_state = tf.cast(next_state, dtype=tf.float32)
        reward = tf.cast(reward, tf.float32)
        done = tf.cast(done, tf.float32)
        
        with tf.GradientTape(persistent=True) as tape:
            logits = self.actor(state)
            values = self.critic(state)
            next_values = self.critic(next_state)
            
            # TD Target and Advantage
            # Target = r + gamma * V(s') * (1-done)
            target = reward + (self.gamma * next_values * (1.0 - done))
            advantage = target - values
            advantage = (advantage - tf.reduce_mean(advantage)) / (tf.math.reduce_std(advantage) + 1e-8)

            # Actor Loss
            action_masks = tf.one_hot(tf.squeeze(action), logits.shape[-1])
            # Log-softmax for numerical stability
            log_probs = tf.reduce_sum(tf.nn.log_softmax(logits) * action_masks, axis=1, keepdims=True)
            
            actor_loss = -tf.reduce_mean(log_probs * tf.stop_gradient(advantage))
            
            # Entropy Loss
            probs = tf.nn.softmax(logits)
            entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-9), axis=1)
            actor_loss -= self.entropy_beta * tf.reduce_mean(entropy)

            # Critic Loss
            critic_loss = tf.reduce_mean(tf.square(advantage))
            
            # Total loss to optimize
            total_loss = actor_loss + 0.5*critic_loss
            
        # Gradient Application
        # We optimize both models simultaneously
        trainable_vars = self.actor.trainable_variables + self.critic.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        
        del tape
        
        return float(total_loss)

    def save_models(self, folder_path):
        self.actor.save_weights(os.path.join(folder_path, "actor.weights.h5"))
        self.critic.save_weights(os.path.join(folder_path, "critic.weights.h5"))

    def load_models(self, folder_path, state_shape, train=True):
        dummy_input = tf.zeros((1, state_shape), dtype=tf.float32)
        
        self.actor(dummy_input)
        self.actor.load_weights(os.path.join(folder_path, "actor.weights.h5"))
        
        if train and self.critic is not None:
            self.critic(dummy_input)
            self.critic.load_weights(os.path.join(folder_path, "critic.weights.h5"))