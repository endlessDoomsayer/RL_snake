import tensorflow as tf
from tensorflow.keras import layers


def create_models(state_shape, action_dim):
    actor_model = ActorModel(action_dim)
    critic_model = CriticModel()
    
    # Build the models by passing a dummy input through them
    dummy_input = tf.zeros((1, *state_shape))
    actor_model(dummy_input)
    critic_model(dummy_input)
    
    return actor_model, critic_model

def create_logic(state_shape, action_dim, optimizer, gamma=0.9):
    actor_model, critic_model = create_models(state_shape, action_dim)
    return ActorCriticLogic(actor_model, critic_model, optimizer, gamma)


class ActorModel(tf.keras.Model):
    def __init__(self, action_dim=4):
        super(ActorModel, self).__init__()
        self.flatten = layers.Flatten()
        self.layer_1 = layers.Dense(128, activation='swish')
        self.layer_2 = layers.Dense(128, activation='swish')
        self.layer_3 = layers.Dense(action_dim) # Outputs logits for each action

    def call(self, x):
        x = self.flatten(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        return self.layer_3(x)

class CriticModel(tf.keras.Model):
    def __init__(self):
        super(CriticModel, self).__init__()
        self.flatten = layers.Flatten()
        self.layer_1 = layers.Dense(128, activation='swish')
        self.layer_2 = layers.Dense(128, activation='swish')
        self.layer_3 = layers.Dense(1) # Predicts V(s)

    def call(self, s):
        x = self.flatten(s)
        x = self.layer_1(x)
        x = self.layer_2(x)
        return self.layer_3(x)





class ActorCriticLogic:
    def __init__(self, actor_model, critic_model, optimizer, gamma=0.9):
        self.actor = actor_model    # Outputs probabilities for each action
        self.critic = critic_model  # Outputs a single Value V(s)
        self.optimizer = optimizer
        self.gamma = gamma

    def get_action(self, state, training=True):
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
        logits = self.actor(state_tensor)
        
        if training:
            #stochastic exploration
            action = tf.random.categorical(logits, num_samples=1)
        else:
            #deterministic
            action = tf.argmax(logits, axis=1)
            
        return action, logits

    def train_step(self, state, action, reward, next_state, done):
        state = tf.cast(state, dtype=tf.float32)
        next_state = tf.cast(next_state, dtype=tf.float32)
        reward = tf.cast(reward, tf.float32) # shape (n_boards, 1)
        done = tf.cast(done, tf.float32)     # shape (n_boards, 1)
        
        with tf.GradientTape(persistent=True) as tape:
            # 1. Forward pass
            logits = self.actor(state)
            values = self.critic(state)
            next_values = self.critic(next_state)
            
            # 2. Shared math
            target = reward + (self.gamma * next_values * (1.0 - done))
            advantage = target - values
            
            # 3. Actor Loss (Log Prob * Advantage + Entropy)
            action_masks = tf.one_hot(tf.squeeze(action), 4)
            log_probs = tf.reduce_sum(tf.nn.log_softmax(logits) * action_masks, axis=1, keepdims=True)
            
            # We stop_gradient on advantage so Actor doesn't train the Critic
            actor_loss = -tf.reduce_mean(log_probs * tf.stop_gradient(advantage))
            
            '''
            # Add Entropy for exploration
            probs = tf.nn.softmax(logits)
            entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-9), axis=1)
            actor_loss -= self.entropy_beta * tf.reduce_mean(entropy)
            '''

            # 4. Critic Loss (Mean Squared Error of TD-Error)
            critic_loss = tf.reduce_mean(tf.square(advantage))
            
        # Calculate and apply Actor gradients
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        
        # Calculate and apply Critic gradients
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        
        # Clear the persistent tape
        del tape
        
        all_grads = actor_grads + critic_grads
        all_vars = self.actor.trainable_variables + self.critic.trainable_variables
        
        self.optimizer.apply_gradients(zip(all_grads, all_vars))

        # Return critic loss for history plots
        return critic_loss
    
    def save_models(self, folder_path):
        # Saves two separate files
        self.actor.save_weights(f"{folder_path}/actor.weights.h5")
        self.critic.save_weights(f"{folder_path}/critic.weights.h5")
        #print(f"Successfully saved Actor and Critic weights at {folder_path}")

    def load_models(self, folder_path, state_shape, load_critic=True):
        dummy_input = tf.zeros((1, *state_shape))

        # 2. Build and Load Actor
        self.actor(dummy_input) # Forces TF to initialize weights
        self.actor.load_weights(f"{folder_path}/actor.weights.h5")
        print("Actor weights loaded.")

        # 3. Build and Load Critic (only if needed)
        if load_critic:
            if self.critic is None:
                print("Warning: Attempted to load Critic but no Critic model was provided.")
            else:
                self.critic(dummy_input)
                self.critic.load_weights(f"{folder_path}/critic.weights.h5")
                print("Critic weights loaded.")
