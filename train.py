# %% [markdown]
# # Snake RL Training notebook
# All the details are in the pdf report.
# 
# The following notebook is used to train RL models in both fully observable and partially observable environments.
# 
# Training is done by simulating 1000 parallel environments of grid size 10x10 on 10000 iterations, gamma 0.99.
# 
# Partially observable environments allow for a visibility of radius 2.
# 
# The models are:
# - actor critic
# - dqn
# 
# Each model is trained with the original reward function and with the custom reward function.
# 
# Models are saved in folder "results/[environment type]/[model type]\_[gamma]\_[#rays]\_[reward function]", then used to run the training evaluations, saved in "results/[environment type]/[model type]\_[gamma]\_[#rays]\_[reward function]\stats\train\training_stats.csv" and "results/[environment type]/[model type]\_[gamma]\_[#rays]\_[reward function]\stats\train\evaluation_stats.csv".
# 
# After that, only the 2 best models for each environment (fully or partially observable) are kept (the others are deleted manually to make the final submission lighter, as stated in the assignment).
# 
# To retrieve all the models, please retrain them using this script.

# %%
import numpy as np
from  tqdm import trange
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import keras as K
import os

import environments_fully_observable
import environments_partially_observable
from agent import Agent
import algorithms.actor_critic as ac
import algorithms.dqn as dqn
from buffers.replay_buffer import ReplayBuffer
from raycasting_encoder import RayCastingEncoder
from metrics import MetricsTracker

import utils

tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)

# %%
GAMMA = .99
ITERATIONS = 10000
EVAL_FREQUENCY = 500
TRAIN_SAVE_FREQUENCY = 100
NUM_RAYS = 8
VISIBILITY_RADIUS = 2
N_BOARDS = 1000
GRID_SIZE = 10

# %% [markdown]
# ### Environment definition

# %%
def get_env(env_type='fully_observable', reward_type='classic', n_boards=4, grid_size=10, visibility_radius=2):
    '''
    Generates the environment based on the env_type variable.
    Args:
        env_type: str, type of environment to generate
        reward_type: str, type of reward function to use
        n_boards: int, number of boards to simulate in parallel
        grid_size: int, size of each board (including borders)
        visibility_radius: int, size of the local neighborhood for partially observable environment
    Returns:
        e: environment object
    Raises:
        ValueError: if env_type is not valid
    '''
    if env_type == 'fully_observable':
        if reward_type == 'classic':
            e = environments_fully_observable.OriginalSnakeEnvironment(n_boards, grid_size)
        elif reward_type == 'custom':
            e = environments_fully_observable.CustomSnakeEnvironment(n_boards, grid_size)
    elif env_type == 'partially_observable':
        if reward_type == 'classic':
            e = environments_partially_observable.OriginalSnakeEnvironment(n_boards, grid_size, visibility_radius)
        elif reward_type == 'custom':
            e = environments_partially_observable.CustomSnakeEnvironment(n_boards, grid_size, visibility_radius)
    
    return e

# %% [markdown]
# ### Training definition

# %%
def run_evaluation(agent, env_test, max_steps=200):
    if agent is None:
        raise ValueError("Agent cannot be None for evaluation.")
    if env_test is None:
        raise ValueError("Environment cannot be None for evaluation.")
    
    tracker = MetricsTracker(env_test.n_boards)
    
    active_mask = np.ones(env_test.n_boards, dtype=np.float32)
    raw_state = env_test.to_state()
    
    prog_bar = trange(max_steps, desc="Evaluation Progress", unit="step")

    for iter in prog_bar:
        if hasattr(agent.logic, 'encoder') and agent.logic.encoder is not None:
            state_vec = agent.logic.encoder.encode(raw_state)
        else:
            state_vec = raw_state
        actions, _ = agent.get_action(state_vec, training=False)
        
        rewards = env_test.move(actions)
        
        # Update metrics tracker
        tracker.update(rewards, active_mask)
        
        # Update local active mask for the next step
        # A board dies if reward is -0.1 (Wall)
        deaths = (np.abs(rewards.numpy().flatten() - (-0.1)) < 1e-5)
        active_mask[deaths] = 0.0
        
        if np.sum(active_mask) == 0:
            break
            
        raw_state = env_test.to_state()

    # Retrieve the history for plotting or logging
    stats = tracker.get_current_averages()
    
    return stats, tracker.history

# %%
def train_buffer(agent, env_train, buffer, env_type, reward_type='classic', MIN_BUFFER_SIZE=2000, BATCH_SIZE=512, TRAIN_EPOCHS=4, ITERATIONS=10000, TRAIN_SAVE_FREQUENCY=100, EVAL_FREQUENCY=100):
    progress_bar = trange(ITERATIONS)

    window_losses = []
    window_rewards = []

    best_reward = float('-inf')

    state = env_train.to_state()
    if agent.logic.encoder is not None:
        state = agent.logic.encoder.encode(state)

    for iteration in progress_bar:

        actions, _ = agent.get_action(state, training=True)

        rewards = env_train.move(actions)
        
        next_state = env_train.to_state()
        if agent.logic.encoder is not None:
            next_state = agent.logic.encoder.encode(next_state)
        
        done = np.logical_or(
            np.abs(rewards.numpy() - (-0.1)) < 1e-5,
            np.abs(rewards.numpy() - (-0.2)) < 1e-5
        ).astype(np.float32)

        buffer.push(state, actions.numpy(), rewards.numpy(), next_state, done)

        dead_indices = np.flatnonzero(done)

        if dead_indices.size > 0:
            new_boards = []
            for idx in dead_indices:
                env_train.bodies[idx] = [] # Reset body list
                b = env_train.get_board()
                # Fruit placement
                empty = np.argwhere(b == env_train.EMPTY)
                f = empty[np.random.randint(len(empty))]
                b[f[0], f[1]] = env_train.FRUIT
                new_boards.append(b)
            
            env_train.boards[dead_indices] = np.array(new_boards)

            mini_raw_state = K.utils.to_categorical(env_train.boards[dead_indices], num_classes=5)[..., 1:]

            if agent.logic.encoder is not None:
                mini_raw_state = agent.logic.encoder.encode(mini_raw_state)
            next_state[dead_indices] = mini_raw_state

        state = next_state

        if buffer.size >= MIN_BUFFER_SIZE:
            losses = []
            batch_rewards = []
            for _ in range(TRAIN_EPOCHS):
                s_b, a_b, r_b, ns_b, d_b = buffer.sample(BATCH_SIZE)
                loss_val, b_reward = agent.train_step(s_b, a_b, r_b, ns_b, d_b)
                losses.append(loss_val)
                batch_rewards.append(b_reward)

            window_losses.append(np.mean(losses))
            window_rewards.append(np.mean(batch_rewards))

            if (iteration) % TRAIN_SAVE_FREQUENCY == 0:
                if window_losses and window_rewards:
                    avg_loss = np.mean(window_losses)
                    avg_reward = np.mean(window_rewards)
                    # Save training stats to CSV
                    utils.save_to_csv(filename="training_stats.csv", data_dict={"iteration": iteration, "loss": avg_loss, "reward": avg_reward}, folder=agent.train_csv_dir)

                    window_losses.clear()
                    window_rewards.clear()

            if (iteration) % EVAL_FREQUENCY == 0:
                env_test = get_env(env_type=env_type, reward_type=reward_type, n_boards=N_BOARDS, grid_size=GRID_SIZE, visibility_radius=VISIBILITY_RADIUS)
                stats, _ = run_evaluation(agent, env_test)
                
                avg_reward = stats.get('avg_reward', 0)
                
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    # Save the new best model
                    agent.save(agent.best_path)
                    print(f"\n[Best Model Update] Iteration {iteration}: Avg Reward = {avg_reward:.2f}")

                utils.save_to_csv(filename="evaluation_stats.csv", data_dict={"iteration": iteration, **stats}, folder=agent.train_csv_dir)

                agent.save(os.path.join(agent.checkpoint_dir, f"model_iter{iteration}"))


# %% [markdown]
# ## Fully observable environment

# %%
ENVIRONMENT_TYPE = 'fully_observable'
REWARD_TYPE = 'classic'
RESULTS_PATH = os.path.join("results", ENVIRONMENT_TYPE)

# %%
env_train = get_env(env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, n_boards=N_BOARDS, grid_size=GRID_SIZE, visibility_radius=VISIBILITY_RADIUS)
print(env_train.to_state()[0].shape)

# %%
fig,axs=plt.subplots(1,min(len(env_train.boards), 5), figsize=(10,3))
for ax, board in zip(axs, env_train.boards):
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.imshow(board, origin="lower")

# %% [markdown]
# ### Training 1: DQN, gamma 0.99, 8 rays, classic reward function

# %%
REWARD_TYPE = 'classic'

# %%
env_train = get_env(env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, n_boards=N_BOARDS, grid_size=GRID_SIZE, visibility_radius=VISIBILITY_RADIUS)

encoder = RayCastingEncoder(num_rays=NUM_RAYS)
print("Encoder converts state {} to shape: {}".format(env_train.to_state().shape, encoder.encode(env_train.to_state()).shape))
optimizer = tf.keras.optimizers.Adam(1e-4)
logic = dqn.create_logic(action_dim=4, encoder=encoder, optimizer=optimizer, gamma=GAMMA)
agent = Agent(algorithm_logic=logic, algorithm_name=f"dqn_99_{NUM_RAYS}rays_R{REWARD_TYPE}", results_path=RESULTS_PATH)

# %%
buffer = ReplayBuffer(capacity=10000, state_shape=encoder.output_dim)
MIN_BUFFER_SIZE = 2000
BATCH_SIZE = 512
TRAIN_EPOCHS = 4

train_buffer(agent, env_train, buffer, env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, MIN_BUFFER_SIZE=MIN_BUFFER_SIZE, BATCH_SIZE=BATCH_SIZE, TRAIN_EPOCHS=TRAIN_EPOCHS, ITERATIONS=ITERATIONS, TRAIN_SAVE_FREQUENCY=TRAIN_SAVE_FREQUENCY, EVAL_FREQUENCY=EVAL_FREQUENCY)

# %%
#clear memory after training
del env_train
del agent
del buffer
del logic
del optimizer
del encoder

# %% [markdown]
# ### Training 2: AC, gamma 0.99, 8 rays, classic reward function

# %%
REWARD_TYPE = 'classic'

# %%
env_train = get_env(env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, n_boards=N_BOARDS, grid_size=GRID_SIZE, visibility_radius=VISIBILITY_RADIUS)

encoder = RayCastingEncoder(num_rays=NUM_RAYS)
print("Encoder converts state {} to shape: {}".format(env_train.to_state().shape, encoder.encode(env_train.to_state()).shape))
optimizer = tf.keras.optimizers.Adam(1e-4)
logic = ac.create_logic(action_dim=4, encoder=encoder, optimizer=optimizer, gamma=GAMMA)
agent = Agent(algorithm_logic=logic, algorithm_name=f"actor_critic_99_{NUM_RAYS}rays_R{REWARD_TYPE}", results_path=RESULTS_PATH)

# %%
buffer = ReplayBuffer(capacity=1000, state_shape=encoder.output_dim)
MIN_BUFFER_SIZE = 1000
BATCH_SIZE = 512

TRAIN_EPOCHS = 1

train_buffer(agent, env_train, buffer, env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, MIN_BUFFER_SIZE=MIN_BUFFER_SIZE, BATCH_SIZE=BATCH_SIZE, TRAIN_EPOCHS=TRAIN_EPOCHS, ITERATIONS=ITERATIONS, TRAIN_SAVE_FREQUENCY=TRAIN_SAVE_FREQUENCY, EVAL_FREQUENCY=EVAL_FREQUENCY)

# %%
#clear memory after training
del env_train
del agent
del buffer
del logic
del optimizer
del encoder

# %% [markdown]
# ### Training 3: DQN, gamma 0.99, 8 rays, custom reward function

# %%
REWARD_TYPE = 'custom'

# %%
env_train = get_env(env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, n_boards=N_BOARDS, grid_size=GRID_SIZE, visibility_radius=VISIBILITY_RADIUS)

encoder = RayCastingEncoder(num_rays=NUM_RAYS)
print("Encoder converts state {} to shape: {}".format(env_train.to_state().shape, encoder.encode(env_train.to_state()).shape))
optimizer = tf.keras.optimizers.Adam(1e-4)
logic = dqn.create_logic(action_dim=4, encoder=encoder, optimizer=optimizer, gamma=GAMMA)
agent = Agent(algorithm_logic=logic, algorithm_name=f"dqn_99_{NUM_RAYS}rays_R{REWARD_TYPE}", results_path=RESULTS_PATH)

# %%
buffer = ReplayBuffer(capacity=10000, state_shape=encoder.output_dim)
MIN_BUFFER_SIZE = 2000
BATCH_SIZE = 512
TRAIN_EPOCHS = 4

train_buffer(agent, env_train, buffer, env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, MIN_BUFFER_SIZE=MIN_BUFFER_SIZE, BATCH_SIZE=BATCH_SIZE, TRAIN_EPOCHS=TRAIN_EPOCHS, ITERATIONS=ITERATIONS, TRAIN_SAVE_FREQUENCY=TRAIN_SAVE_FREQUENCY, EVAL_FREQUENCY=EVAL_FREQUENCY)

# %%
#clear memory after training
del env_train
del agent
del buffer
del logic
del optimizer
del encoder

# %% [markdown]
# ### Training 4: AC, gamma 0.99, 8 rays, custom reward function

# %%
REWARD_TYPE = 'custom'

# %%
env_train = get_env(env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, n_boards=N_BOARDS, grid_size=GRID_SIZE, visibility_radius=VISIBILITY_RADIUS)

encoder = RayCastingEncoder(num_rays=NUM_RAYS)
print("Encoder converts state {} to shape: {}".format(env_train.to_state().shape, encoder.encode(env_train.to_state()).shape))
optimizer = tf.keras.optimizers.Adam(1e-4)
logic = ac.create_logic(action_dim=4, encoder=encoder, optimizer=optimizer, gamma=GAMMA)
agent = Agent(algorithm_logic=logic, algorithm_name=f"actor_critic_99_{NUM_RAYS}rays_R{REWARD_TYPE}", results_path=RESULTS_PATH)

# %%
buffer = ReplayBuffer(capacity=1000, state_shape=encoder.output_dim)
MIN_BUFFER_SIZE = 1000
BATCH_SIZE = 512

TRAIN_EPOCHS = 1

train_buffer(agent, env_train, buffer, env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, MIN_BUFFER_SIZE=MIN_BUFFER_SIZE, BATCH_SIZE=BATCH_SIZE, TRAIN_EPOCHS=TRAIN_EPOCHS, ITERATIONS=ITERATIONS, TRAIN_SAVE_FREQUENCY=TRAIN_SAVE_FREQUENCY, EVAL_FREQUENCY=EVAL_FREQUENCY)

# %%
#clear memory after training
del env_train
del agent
del buffer
del logic
del optimizer
del encoder

# %% [markdown]
# ### Training comparison

# %%
AGENTS = {f"dqn_99_8rays_Rclassic", f"actor_critic_99_8rays_Rclassic", f"dqn_99_8rays_Rcustom", f"actor_critic_99_8rays_Rcustom"}

train_comparison = {}
eval_comparison = {}
for agent in AGENTS:
    train_comparison[agent] = f"results/{ENVIRONMENT_TYPE}/{agent}/stats/train/training_stats.csv"
    eval_comparison[agent] = f"results/{ENVIRONMENT_TYPE}/{agent}/stats/train/evaluation_stats.csv"

print(train_comparison)

# %%
utils.get_agents_comparison_plot(agent_paths_dict=train_comparison, output_folder=os.path.join(RESULTS_PATH, "train_results"), prefix="train_comparison")

# %%
utils.get_agents_comparison_plot(agent_paths_dict=eval_comparison, output_folder=os.path.join(RESULTS_PATH, "train_results"), prefix="train_comparison")

# %% [markdown]
# ## Partially observable environment

# %%
ENVIRONMENT_TYPE = 'partially_observable'
RESULTS_PATH = os.path.join("results", ENVIRONMENT_TYPE)
REWARD_TYPE = 'classic'

# %%
env_train = get_env(env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, n_boards=N_BOARDS, grid_size=GRID_SIZE, visibility_radius=VISIBILITY_RADIUS)

print(env_train.to_state()[0].shape)

# %%
fig,axs=plt.subplots(1,min(len(env_train.boards), 5), figsize=(10,3))
for ax, board in zip(axs, env_train.boards):
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.imshow(board, origin="lower")

# %% [markdown]
# ### Training 1: DQN, gamma 0.99, 8 rays, classic reward function

# %%
REWARD_TYPE = 'classic'

# %%
env_train = get_env(env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, n_boards=N_BOARDS, grid_size=GRID_SIZE, visibility_radius=VISIBILITY_RADIUS)

encoder = RayCastingEncoder(num_rays=NUM_RAYS)
print("Encoder converts state {} to shape: {}".format(env_train.to_state().shape, encoder.encode(env_train.to_state()).shape))
optimizer = tf.keras.optimizers.Adam(1e-4)
logic = dqn.create_logic(action_dim=4, encoder=encoder, optimizer=optimizer, gamma=GAMMA)
agent = Agent(algorithm_logic=logic, algorithm_name=f"dqn_99_{NUM_RAYS}rays_R{REWARD_TYPE}", results_path=RESULTS_PATH)

# %%
buffer = ReplayBuffer(capacity=10000, state_shape=encoder.output_dim)
MIN_BUFFER_SIZE = 2000
BATCH_SIZE = 512
TRAIN_EPOCHS = 4

train_buffer(agent, env_train, buffer, env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, MIN_BUFFER_SIZE=MIN_BUFFER_SIZE, BATCH_SIZE=BATCH_SIZE, TRAIN_EPOCHS=TRAIN_EPOCHS, ITERATIONS=ITERATIONS, TRAIN_SAVE_FREQUENCY=TRAIN_SAVE_FREQUENCY, EVAL_FREQUENCY=EVAL_FREQUENCY)

# %%
#clear memory after training
del env_train
del agent
del buffer
del logic
del optimizer
del encoder

# %% [markdown]
# ### Training 2: AC, gamma 0.99, 8 rays, classic reward function

# %%
REWARD_TYPE = 'classic'

# %%
env_train = get_env(env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, n_boards=N_BOARDS, grid_size=GRID_SIZE, visibility_radius=VISIBILITY_RADIUS)

encoder = RayCastingEncoder(num_rays=NUM_RAYS)
print("Encoder converts state {} to shape: {}".format(env_train.to_state().shape, encoder.encode(env_train.to_state()).shape))
optimizer = tf.keras.optimizers.Adam(1e-4)
logic = ac.create_logic(action_dim=4, encoder=encoder, optimizer=optimizer, gamma=GAMMA)
agent = Agent(algorithm_logic=logic, algorithm_name=f"actor_critic_99_{NUM_RAYS}rays_R{REWARD_TYPE}", results_path=RESULTS_PATH)

# %%
buffer = ReplayBuffer(capacity=1000, state_shape=encoder.output_dim)
MIN_BUFFER_SIZE = 1000
BATCH_SIZE = 512

TRAIN_EPOCHS = 1

train_buffer(agent, env_train, buffer, env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, MIN_BUFFER_SIZE=MIN_BUFFER_SIZE, BATCH_SIZE=BATCH_SIZE, TRAIN_EPOCHS=TRAIN_EPOCHS, ITERATIONS=ITERATIONS, TRAIN_SAVE_FREQUENCY=TRAIN_SAVE_FREQUENCY, EVAL_FREQUENCY=EVAL_FREQUENCY)

# %%
#clear memory after training
del env_train
del agent
del buffer
del logic
del optimizer
del encoder

# %% [markdown]
# ### Training 3: DQN, gamma 0.99, 8 rays, custom reward function

# %%
REWARD_TYPE = 'custom'

# %%
env_train = get_env(env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, n_boards=N_BOARDS, grid_size=GRID_SIZE, visibility_radius=VISIBILITY_RADIUS)

encoder = RayCastingEncoder(num_rays=NUM_RAYS)
print("Encoder converts state {} to shape: {}".format(env_train.to_state().shape, encoder.encode(env_train.to_state()).shape))
optimizer = tf.keras.optimizers.Adam(1e-4)
logic = dqn.create_logic(action_dim=4, encoder=encoder, optimizer=optimizer, gamma=GAMMA)
agent = Agent(algorithm_logic=logic, algorithm_name=f"dqn_99_{NUM_RAYS}rays_R{REWARD_TYPE}", results_path=RESULTS_PATH)

# %%
buffer = ReplayBuffer(capacity=10000, state_shape=encoder.output_dim)
MIN_BUFFER_SIZE = 2000
BATCH_SIZE = 512
TRAIN_EPOCHS = 4

train_buffer(agent, env_train, buffer, env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, MIN_BUFFER_SIZE=MIN_BUFFER_SIZE, BATCH_SIZE=BATCH_SIZE, TRAIN_EPOCHS=TRAIN_EPOCHS, ITERATIONS=ITERATIONS, TRAIN_SAVE_FREQUENCY=TRAIN_SAVE_FREQUENCY, EVAL_FREQUENCY=EVAL_FREQUENCY)

# %%
#clear memory after training
del env_train
del agent
del buffer
del logic
del optimizer
del encoder

# %% [markdown]
# ### Training 4: AC, gamma 0.99, 8 rays, custom reward function

# %%
REWARD_TYPE = 'custom'

# %%
env_train = get_env(env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, n_boards=N_BOARDS, grid_size=GRID_SIZE, visibility_radius=VISIBILITY_RADIUS)

encoder = RayCastingEncoder(num_rays=NUM_RAYS)
print("Encoder converts state {} to shape: {}".format(env_train.to_state().shape, encoder.encode(env_train.to_state()).shape))
optimizer = tf.keras.optimizers.Adam(1e-4)
logic = ac.create_logic(action_dim=4, encoder=encoder, optimizer=optimizer, gamma=GAMMA)
agent = Agent(algorithm_logic=logic, algorithm_name=f"actor_critic_99_{NUM_RAYS}rays_R{REWARD_TYPE}", results_path=RESULTS_PATH)

# %%
buffer = ReplayBuffer(capacity=1000, state_shape=encoder.output_dim)
MIN_BUFFER_SIZE = 1000
BATCH_SIZE = 512

TRAIN_EPOCHS = 1

train_buffer(agent, env_train, buffer, env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, MIN_BUFFER_SIZE=MIN_BUFFER_SIZE, BATCH_SIZE=BATCH_SIZE, TRAIN_EPOCHS=TRAIN_EPOCHS, ITERATIONS=ITERATIONS, TRAIN_SAVE_FREQUENCY=TRAIN_SAVE_FREQUENCY, EVAL_FREQUENCY=EVAL_FREQUENCY)

# %%
#clear memory after training
del env_train
del agent
del buffer
del logic
del optimizer
del encoder

# %% [markdown]
# ### Training comparison

# %%
AGENTS = {f"dqn_99_8rays_Rclassic", f"actor_critic_99_8rays_Rclassic", f"dqn_99_8rays_Rcustom", f"actor_critic_99_8rays_Rcustom"}

train_comparison = {}
eval_comparison = {}
for agent in AGENTS:
    train_comparison[agent] = f"results/{ENVIRONMENT_TYPE}/{agent}/stats/train/training_stats.csv"
    eval_comparison[agent] = f"results/{ENVIRONMENT_TYPE}/{agent}/stats/train/evaluation_stats.csv"

print(train_comparison)

# %%
utils.get_agents_comparison_plot(agent_paths_dict=train_comparison, output_folder=os.path.join(RESULTS_PATH, "train_results"), prefix="train_comparison")

# %%
utils.get_agents_comparison_plot(agent_paths_dict=eval_comparison, output_folder=os.path.join(RESULTS_PATH, "train_results"), prefix="train_comparison")


