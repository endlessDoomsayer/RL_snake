# %% [markdown]
# # Snake RL Evaluation notebook
# All the details are in the pdf report.
# 
# The following notebook is used to evaluate RL models and baselines in both fully observable and partially observable environments, using the classic reward function.
# 
# Evaluation is done by simulating 1000 parallel environments of grid size 10x10 and 20x20 on max 2000 steps.
# 
# Partially observable environments allow for a visibility of radius 2.
# 
# The models are:
# - dqn
# - actor critic
# - random
# - hamiltonian cycle
# - greedy
# 
# Evaluation stats are saved in folder "results/[environment type]/[model type]\_[gamma]\_[#rays]\_[reward function]\stats\test\evaluation_stats_[grid size].csv".

# %%
import tensorflow as tf
import numpy as np
import random
import os
from  tqdm import trange
import utils

import environments_fully_observable
import environments_partially_observable
from agent import Agent
import algorithms.actor_critic as ac
import algorithms.dqn as dqn
import algorithms.random as rn
import algorithms.hamiltonian as hm
import algorithms.greedy as gr
from raycasting_encoder import RayCastingEncoder
from metrics import MetricsTracker

tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)

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

# %% [markdown]
# # Grid size 10x10

# %%
GRID_SIZE = 10

# %%
FILENAME = f"evaluation_stats_{GRID_SIZE}.csv"

# %% [markdown]
# ## Fully Observable Environment

# %%
ENVIRONMENT_TYPE = 'fully_observable'
RESULTS_PATH = os.path.join("results", ENVIRONMENT_TYPE)

VISIBILITY_RADIUS = 2 #ignored for fully observable environment

N_BOARDS = 1000

GAMMA = 0.99
BEST_PARAMS_FOLDER = "best_parameters"
NUM_RAYS = 8
REWARD_TYPE = "classic"

# %%
#DQN
encoder = RayCastingEncoder(num_rays=NUM_RAYS)
logic = dqn.create_logic(action_dim=4, encoder=encoder, optimizer=None, gamma=GAMMA)
logic.load_models(folder_path=os.path.join(RESULTS_PATH, f"dqn_99_{NUM_RAYS}rays_R{REWARD_TYPE}", BEST_PARAMS_FOLDER), state_shape=encoder.output_dim, train=False)
agent = Agent(algorithm_logic=logic, algorithm_name=f"dqn_99_{NUM_RAYS}rays_R{REWARD_TYPE}", results_path=RESULTS_PATH)

env_test = get_env(env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, n_boards=N_BOARDS, grid_size=GRID_SIZE, visibility_radius=VISIBILITY_RADIUS)
stats, history = run_evaluation(agent, env_test, max_steps=2000)
print(f"{agent.algorithm_name} performance: {stats}")
utils.save_to_csv(filename=FILENAME, data_dict={**stats}, folder=agent.test_csv_dir)

# %%
#A2C
encoder = RayCastingEncoder(num_rays=NUM_RAYS)
logic = ac.create_logic(action_dim=4, encoder=encoder, optimizer=None, gamma=GAMMA)
logic.load_models(folder_path=os.path.join(RESULTS_PATH, f"actor_critic_99_{NUM_RAYS}rays_R{REWARD_TYPE}", BEST_PARAMS_FOLDER), state_shape=encoder.output_dim, train=False)
agent = Agent(algorithm_logic=logic, algorithm_name=f"actor_critic_99_{NUM_RAYS}rays_R{REWARD_TYPE}", results_path=RESULTS_PATH)

env_test = get_env(env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, n_boards=N_BOARDS, grid_size=GRID_SIZE, visibility_radius=VISIBILITY_RADIUS)
stats, history = run_evaluation(agent, env_test, max_steps=2000)
print(f"{agent.algorithm_name} performance: {stats}")
utils.save_to_csv(filename=FILENAME, data_dict={**stats}, folder=agent.test_csv_dir)

# %%
#RANDOM
logic = rn.create_logic(n_boards=N_BOARDS)
agent = Agent(algorithm_logic=logic, algorithm_name="random_Rclassic", results_path=RESULTS_PATH)

env_test = get_env(env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, n_boards=N_BOARDS, grid_size=GRID_SIZE, visibility_radius=VISIBILITY_RADIUS)
stats, history = run_evaluation(agent, env_test, max_steps=2000)
print(f"{agent.algorithm_name} performance: {stats}")
utils.save_to_csv(filename=FILENAME, data_dict={**stats}, folder=agent.test_csv_dir)

# %%
#Hamiltonian Cycle
logic = hm.create_logic(grid_size=GRID_SIZE, n_boards=N_BOARDS)
agent = Agent(algorithm_logic=logic, algorithm_name="hamiltonian_Rclassic", results_path=RESULTS_PATH)

env_test = get_env(env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, n_boards=N_BOARDS, grid_size=GRID_SIZE, visibility_radius=VISIBILITY_RADIUS)
stats, history = run_evaluation(agent, env_test, max_steps=2000)
print(f"{agent.algorithm_name} performance: {stats}")
utils.save_to_csv(filename=FILENAME, data_dict={**stats}, folder=agent.test_csv_dir)

# %%
#GREEDY
logic = gr.create_logic(n_boards=N_BOARDS)
agent = Agent(algorithm_logic=logic, algorithm_name="greedy_Rclassic", results_path=RESULTS_PATH)

env_test = get_env(env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, n_boards=N_BOARDS, grid_size=GRID_SIZE, visibility_radius=VISIBILITY_RADIUS)
stats, history = run_evaluation(agent, env_test, max_steps=2000)
print(f"{agent.algorithm_name} performance: {stats}")
utils.save_to_csv(filename=FILENAME, data_dict={**stats}, folder=agent.test_csv_dir)

# %% [markdown]
# ## Partially Observable Environment

# %%
ENVIRONMENT_TYPE = 'partially_observable'
RESULTS_PATH = os.path.join("results", ENVIRONMENT_TYPE)

VISIBILITY_RADIUS = 2

N_BOARDS = 1000
GAMMA = 0.99
BEST_PARAMS_FOLDER = "best_parameters"
NUM_RAYS = 8
REWARD_TYPE = "classic"

# %%
#DQN
encoder = RayCastingEncoder(num_rays=NUM_RAYS)
logic = dqn.create_logic(action_dim=4, encoder=encoder, optimizer=None, gamma=GAMMA)
logic.load_models(folder_path=os.path.join(RESULTS_PATH, f"dqn_99_{NUM_RAYS}rays_R{REWARD_TYPE}", BEST_PARAMS_FOLDER), state_shape=encoder.output_dim, train=False)
agent = Agent(algorithm_logic=logic, algorithm_name=f"dqn_99_{NUM_RAYS}rays_R{REWARD_TYPE}", results_path=RESULTS_PATH)

env_test = get_env(env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, n_boards=N_BOARDS, grid_size=GRID_SIZE, visibility_radius=VISIBILITY_RADIUS)
stats, history = run_evaluation(agent, env_test, max_steps=2000)
print(f"{agent.algorithm_name} performance: {stats}")
utils.save_to_csv(filename=FILENAME, data_dict={**stats}, folder=agent.test_csv_dir)

# %%
#A2C
encoder = RayCastingEncoder(num_rays=NUM_RAYS)
logic = ac.create_logic(action_dim=4, encoder=encoder, optimizer=None, gamma=GAMMA)
logic.load_models(folder_path=os.path.join(RESULTS_PATH, f"actor_critic_99_{NUM_RAYS}rays_R{REWARD_TYPE}", BEST_PARAMS_FOLDER), state_shape=encoder.output_dim, train=False)
agent = Agent(algorithm_logic=logic, algorithm_name=f"actor_critic_99_{NUM_RAYS}rays_R{REWARD_TYPE}", results_path=RESULTS_PATH)

env_test = get_env(env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, n_boards=N_BOARDS, grid_size=GRID_SIZE, visibility_radius=VISIBILITY_RADIUS)
stats, history = run_evaluation(agent, env_test, max_steps=2000)
print(f"{agent.algorithm_name} performance: {stats}")
utils.save_to_csv(filename=FILENAME, data_dict={**stats}, folder=agent.test_csv_dir)

# %%
#RANDOM
logic = rn.create_logic(n_boards=N_BOARDS)
agent = Agent(algorithm_logic=logic, algorithm_name="random_Rclassic", results_path=RESULTS_PATH)

env_test = get_env(env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, n_boards=N_BOARDS, grid_size=GRID_SIZE, visibility_radius=VISIBILITY_RADIUS)
stats, history = run_evaluation(agent, env_test, max_steps=2000)
print(f"{agent.algorithm_name} performance: {stats}")
utils.save_to_csv(filename=FILENAME, data_dict={**stats}, folder=agent.test_csv_dir)

# %%
#Hamiltonian Cycle has the same exact performances as the fully observable environment, since it needs to know only the grid size to compute the cycle, and not the actual state of the board.
#Therefore, we do not evaluate it here.
print("Hamiltonian Cycle has the same performance in partially observable environment as in fully observable environment, skipping evaluation.")

# %%
#GREEDY
logic = gr.create_logic(n_boards=N_BOARDS)
agent = Agent(algorithm_logic=logic, algorithm_name="greedy_Rclassic", results_path=RESULTS_PATH)

env_test = get_env(env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, n_boards=N_BOARDS, grid_size=GRID_SIZE, visibility_radius=VISIBILITY_RADIUS)
stats, history = run_evaluation(agent, env_test, max_steps=2000)
print(f"{agent.algorithm_name} performance: {stats}")
utils.save_to_csv(filename=FILENAME, data_dict={**stats}, folder=agent.test_csv_dir)

# %% [markdown]
# # Grid Size 20x20

# %%
GRID_SIZE = 20

# %%
FILENAME = f"evaluation_stats_{GRID_SIZE}.csv"

# %% [markdown]
# ## Fully Observable Environment

# %%
ENVIRONMENT_TYPE = 'fully_observable'
RESULTS_PATH = os.path.join("results", ENVIRONMENT_TYPE)

VISIBILITY_RADIUS = 2 #ignored for fully observable environment

N_BOARDS = 1000

GAMMA = 0.99
BEST_PARAMS_FOLDER = "best_parameters"
NUM_RAYS = 8
REWARD_TYPE = "classic"

# %%
#DQN
encoder = RayCastingEncoder(num_rays=NUM_RAYS)
logic = dqn.create_logic(action_dim=4, encoder=encoder, optimizer=None, gamma=GAMMA)
logic.load_models(folder_path=os.path.join(RESULTS_PATH, f"dqn_99_{NUM_RAYS}rays_R{REWARD_TYPE}", BEST_PARAMS_FOLDER), state_shape=encoder.output_dim, train=False)
agent = Agent(algorithm_logic=logic, algorithm_name=f"dqn_99_{NUM_RAYS}rays_R{REWARD_TYPE}", results_path=RESULTS_PATH)

env_test = get_env(env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, n_boards=N_BOARDS, grid_size=GRID_SIZE, visibility_radius=VISIBILITY_RADIUS)
stats, history = run_evaluation(agent, env_test, max_steps=2000)
print(f"{agent.algorithm_name} performance: {stats}")
utils.save_to_csv(filename=FILENAME, data_dict={**stats}, folder=agent.test_csv_dir)

# %%
#A2C
encoder = RayCastingEncoder(num_rays=NUM_RAYS)
logic = ac.create_logic(action_dim=4, encoder=encoder, optimizer=None, gamma=GAMMA)
logic.load_models(folder_path=os.path.join(RESULTS_PATH, f"actor_critic_99_{NUM_RAYS}rays_R{REWARD_TYPE}", BEST_PARAMS_FOLDER), state_shape=encoder.output_dim, train=False)
agent = Agent(algorithm_logic=logic, algorithm_name=f"actor_critic_99_{NUM_RAYS}rays_R{REWARD_TYPE}", results_path=RESULTS_PATH)

env_test = get_env(env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, n_boards=N_BOARDS, grid_size=GRID_SIZE, visibility_radius=VISIBILITY_RADIUS)
stats, history = run_evaluation(agent, env_test, max_steps=2000)
print(f"{agent.algorithm_name} performance: {stats}")
utils.save_to_csv(filename=FILENAME, data_dict={**stats}, folder=agent.test_csv_dir)

# %%
#RANDOM
logic = rn.create_logic(n_boards=N_BOARDS)
agent = Agent(algorithm_logic=logic, algorithm_name="random_Rclassic", results_path=RESULTS_PATH)

env_test = get_env(env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, n_boards=N_BOARDS, grid_size=GRID_SIZE, visibility_radius=VISIBILITY_RADIUS)
stats, history = run_evaluation(agent, env_test, max_steps=2000)
print(f"{agent.algorithm_name} performance: {stats}")
utils.save_to_csv(filename=FILENAME, data_dict={**stats}, folder=agent.test_csv_dir)

# %%
#Hamiltonian Cycle
logic = hm.create_logic(grid_size=GRID_SIZE, n_boards=N_BOARDS)
agent = Agent(algorithm_logic=logic, algorithm_name="hamiltonian_Rclassic", results_path=RESULTS_PATH)

env_test = get_env(env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, n_boards=N_BOARDS, grid_size=GRID_SIZE, visibility_radius=VISIBILITY_RADIUS)
stats, history = run_evaluation(agent, env_test, max_steps=2000)
print(f"{agent.algorithm_name} performance: {stats}")
utils.save_to_csv(filename=FILENAME, data_dict={**stats}, folder=agent.test_csv_dir)

# %%
#GREEDY
logic = gr.create_logic(n_boards=N_BOARDS)
agent = Agent(algorithm_logic=logic, algorithm_name="greedy_Rclassic", results_path=RESULTS_PATH)

env_test = get_env(env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, n_boards=N_BOARDS, grid_size=GRID_SIZE, visibility_radius=VISIBILITY_RADIUS)
stats, history = run_evaluation(agent, env_test, max_steps=2000)
print(f"{agent.algorithm_name} performance: {stats}")
utils.save_to_csv(filename=FILENAME, data_dict={**stats}, folder=agent.test_csv_dir)

# %% [markdown]
# ## Partially Observable Environment

# %%
ENVIRONMENT_TYPE = 'partially_observable'
RESULTS_PATH = os.path.join("results", ENVIRONMENT_TYPE)

VISIBILITY_RADIUS = 2

N_BOARDS = 1000
GAMMA = 0.99
BEST_PARAMS_FOLDER = "best_parameters"
NUM_RAYS = 8
REWARD_TYPE = "classic"

# %%
#DQN
encoder = RayCastingEncoder(num_rays=NUM_RAYS)
logic = dqn.create_logic(action_dim=4, encoder=encoder, optimizer=None, gamma=GAMMA)
logic.load_models(folder_path=os.path.join(RESULTS_PATH, f"dqn_99_{NUM_RAYS}rays_R{REWARD_TYPE}", BEST_PARAMS_FOLDER), state_shape=encoder.output_dim, train=False)
agent = Agent(algorithm_logic=logic, algorithm_name=f"dqn_99_{NUM_RAYS}rays_R{REWARD_TYPE}", results_path=RESULTS_PATH)

env_test = get_env(env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, n_boards=N_BOARDS, grid_size=GRID_SIZE, visibility_radius=VISIBILITY_RADIUS)
stats, history = run_evaluation(agent, env_test, max_steps=2000)
print(f"{agent.algorithm_name} performance: {stats}")
utils.save_to_csv(filename=FILENAME, data_dict={**stats}, folder=agent.test_csv_dir)

# %%
#A2C
encoder = RayCastingEncoder(num_rays=NUM_RAYS)
logic = ac.create_logic(action_dim=4, encoder=encoder, optimizer=None, gamma=GAMMA)
logic.load_models(folder_path=os.path.join(RESULTS_PATH, f"actor_critic_99_{NUM_RAYS}rays_R{REWARD_TYPE}", BEST_PARAMS_FOLDER), state_shape=encoder.output_dim, train=False)
agent = Agent(algorithm_logic=logic, algorithm_name=f"actor_critic_99_{NUM_RAYS}rays_R{REWARD_TYPE}", results_path=RESULTS_PATH)

env_test = get_env(env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, n_boards=N_BOARDS, grid_size=GRID_SIZE, visibility_radius=VISIBILITY_RADIUS)
stats, history = run_evaluation(agent, env_test, max_steps=2000)
print(f"{agent.algorithm_name} performance: {stats}")
utils.save_to_csv(filename=FILENAME, data_dict={**stats}, folder=agent.test_csv_dir)

# %%
#RANDOM
logic = rn.create_logic(n_boards=N_BOARDS)
agent = Agent(algorithm_logic=logic, algorithm_name="random_Rclassic", results_path=RESULTS_PATH)

env_test = get_env(env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, n_boards=N_BOARDS, grid_size=GRID_SIZE, visibility_radius=VISIBILITY_RADIUS)
stats, history = run_evaluation(agent, env_test, max_steps=2000)
print(f"{agent.algorithm_name} performance: {stats}")
utils.save_to_csv(filename=FILENAME, data_dict={**stats}, folder=agent.test_csv_dir)

# %%
#Hamiltonian Cycle has the same exact performances as the fully observable environment, since it needs to know only the grid size to compute the cycle, and not the actual state of the board.
#Therefore, we do not evaluate it here.
print("Hamiltonian Cycle has the same performance in partially observable environment as in fully observable environment, skipping evaluation.")

# %%
#GREEDY
logic = gr.create_logic(n_boards=N_BOARDS)
agent = Agent(algorithm_logic=logic, algorithm_name="greedy_Rclassic", results_path=RESULTS_PATH)

env_test = get_env(env_type=ENVIRONMENT_TYPE, reward_type=REWARD_TYPE, n_boards=N_BOARDS, grid_size=GRID_SIZE, visibility_radius=VISIBILITY_RADIUS)
stats, history = run_evaluation(agent, env_test, max_steps=2000)
print(f"{agent.algorithm_name} performance: {stats}")
utils.save_to_csv(filename=FILENAME, data_dict={**stats}, folder=agent.test_csv_dir)


