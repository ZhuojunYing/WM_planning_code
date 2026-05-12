"""
config.py
Handles system arguments, GPU setup, hyperparameter definition,
and builds the decision tree matrices used across the project.
"""

import os
import sys
import random
import numpy as np
import tensorflow as tf

# Import your existing helper script
import helper

# ---------------------------------------------------------
# 1. COMMAND LINE ARGUMENTS
# ---------------------------------------------------------
try:
    lambda_string = sys.argv[1]
    lambda_values = [float(x) for x in lambda_string.split(',')]

    alpha_string = sys.argv[2]
    alpha_values = [float(x) for x in alpha_string.split(',')]

    model_dir_name = sys.argv[3]
    sim_dir_name = sys.argv[4]
    epochs_string = sys.argv[5]
    epochs = int(epochs_string)
    epochs_count = 0

    seed = int(sys.argv[6])

    tree_size = int(sys.argv[7])
    # Renamed from 'train' to avoid variable shadowing
    train_mode = sys.argv[8]
    tree_type = str(sys.argv[9])
    num_trials = int(sys.argv[10]) if len(sys.argv) > 10 else 2000
except IndexError:
    print("Error: Missing command-line arguments.")
    print(
        "Usage: python main.py <lambda> <alpha> "
        "<model_dir_name> <simulation_dir_name> "
        "<epochs> <seed> <tree_size: 3|7|13|31|40> "
        "<train/simulate> <tree_type> [num_trials]"
    )
    sys.exit(1)


# ---------------------------------------------------------
# 2. SEED & REPRODUCIBILITY
# ---------------------------------------------------------
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# ---------------------------------------------------------
# 3. GPU CONFIGURATION
# ---------------------------------------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
method = "exponential"
# ---------------------------------------------------------
# 4. MODEL HYPERPARAMETERS
# ---------------------------------------------------------
tree_steps = tree_size
reward_nodes_by_tree_steps = {
    3: 2,
    7: 6,
    13: 12,
    31: 30,
    40: 39,
}
if tree_steps not in reward_nodes_by_tree_steps:
    raise ValueError(
        "tree_size should be the root-inclusive number of tree nodes: "
        "3, 7, 13, 31, or 40."
    )

latent_dim = 16
output_dim = 64
reward_output_dim = reward_nodes_by_tree_steps[tree_steps]
rnn_units = output_dim
time_steps = reward_output_dim
input_dim = 1
num_categories = 9

trials_per_epoch = 200
batch_size = 200
kl_scaler = 5

# Expected maximum path-reward normalizers used by the actor and critic losses.
reward_normalization_by_time_steps = {
    2: 0.75,
    6: 3.58,
    12: 5.11,
    39: 8.0682,
    30: 8.1574,
}
reward_normalization_constant = reward_normalization_by_time_steps[time_steps]

# ---------------------------------------------------------
# 5. DECISION TREE SETUP
# ---------------------------------------------------------
if time_steps == 2:
    decision_tree = {
        '0': {'right': [-1, '1'], 'up': [-1, '2']},
        '1': {},
        '2': {}
    }


elif time_steps == 6:
    decision_tree = {
        '0': {'right': [-1, '1'], 'up': [-1, '4']},
        '1': {'right': [-1, '2'], 'up': [-1, '3']},
        '2': {},
        '3': {},
        '4': {'right': [-1, '5'], 'up': [-1, '6']},
        '5': {},
        '6': {}
    }

elif time_steps == 12:
    decision_tree = {
        '0': {'right': [-1, '1'], 'up': [-1, '5'], 'left': [-1, "9"]},
        '1': {'right': [-1, '2'], 'up': [-1, '3'], 'left': [-1, "4"]},
        '2': {},
        '3': {},
        '4': {},
        '5': {'right': [-1, '6'], 'up': [-1, '7'], 'left': [-1, "8"]},
        '6': {},
        '7': {},
        '8': {},
        '9': {'right': [-1, '10'], 'up': [-1, '11'], 'left': [-1, "12"]},
        '10': {},
        '11': {},
        '12': {}
    }
elif time_steps == 39:
    decision_tree = {
        '0': {'right': [-1, '1'], 'up': [-1, '14'], 'left': [-1, '27']},
        '1': {'right': [-1, '2'], 'up': [-1, '6'], 'left': [-1, '10']},
        '2': {'right': [-1, '3'], 'up': [-1, '4'], 'left': [-1, '5']},
        '3': {},
        '4': {},
        '5': {},
        '6': {'right': [-1, '7'], 'up': [-1, '8'], 'left': [-1, '9']},
        '7': {},
        '8': {},
        '9': {},
        '10': {'right': [-1, '11'], 'up': [-1, '12'], 'left': [-1, '13']},
        '11': {},
        '12': {},
        '13': {},
        '14': {'right': [-1, '15'], 'up': [-1, '19'], 'left': [-1, '23']},
        '15': {'right': [-1, '16'], 'up': [-1, '17'], 'left': [-1, '18']},
        '16': {},
        '17': {},
        '18': {},
        '19': {'right': [-1, '20'], 'up': [-1, '21'], 'left': [-1, '22']},
        '20': {},
        '21': {},
        '22': {},
        '23': {'right': [-1, '24'], 'up': [-1, '25'], 'left': [-1, '26']},
        '24': {},
        '25': {},
        '26': {},
        '27': {'right': [-1, '28'], 'up': [-1, '32'], 'left': [-1, '36']},
        '28': {'right': [-1, '29'], 'up': [-1, '30'], 'left': [-1, '31']},
        '29': {},
        '30': {},
        '31': {},
        '32': {'right': [-1, '33'], 'up': [-1, '34'], 'left': [-1, '35']},
        '33': {},
        '34': {},
        '35': {},
        '36': {'right': [-1, '37'], 'up': [-1, '38'], 'left': [-1, '39']},
        '37': {},
        '38': {},
        '39': {}
    }
else:
    if (tree_type == "deep_breadth"):

        tree = "deep"
        # Your decision tree dictionary
        decision_tree = {
            '0': {'right': [-1, '1'], 'left': [3, '2']},
            '1': {'up': [-1, '3'], 'down': [2, '4']},
            '2': {'up': [-1, '5'], 'down': [2, '6']},
            '3': {'up': [-1, '7'], 'down': [2, '8']},
            '4': {'up': [-1, '9'], 'down': [2, '10']},
            '5': {'up': [-1, '11'], 'down': [2, '12']},
            '6': {'up': [-1, '13'], 'down': [2, '14']},
            '7': {'up': [-1, '15'], 'down': [2, '16']},
            '8': {'up': [-1, '17'], 'down': [2, '18']},
            '9': {'up': [-1, '19'], 'down': [2, '20']},
            '10': {'right': [-1, '21'], 'left': [3, '22']},
            '11': {'up': [-1, '23'], 'down': [2, '24']},
            '12': {'up': [-1, '25'], 'down': [2, '26']},
            '13': {'up': [-1, '27'], 'down': [2, '28']},
            '14': {'up': [-1, '29'], 'down': [2, '30']},
            '15': {},
            '16': {},
            '17': {},
            '18': {},
            '19': {},
            '20': {},
            '21': {},
            '22': {},
            '23': {},
            '24': {},
            '25': {},
            '26': {},
            '27': {},
            '28': {},
            '29': {},
            '30': {}
        }

    elif tree_type == "deep_depth":

        tree = "deep"
        # Your decision tree dictionary
        decision_tree = {
            '0': {'right': [-1, '1'], 'left': [3, '16']},
            '1': {'up': [-1, '2'], 'down': [2, '9']},
            '2': {'up': [-1, '3'], 'down': [2, '6']},
            '3': {'up': [-1, '4'], 'down': [2, '5']},
            '4': {},
            '5': {},
            '6': {'up': [-1, '7'], 'down': [2, '8']},
            '7': {},
            '8': {},
            '9': {'up': [-1, '10'], 'down': [2, '13']},
            '10': {'right': [-1, '11'], 'left': [3, '12']},
            '11': {},
            '12': {},
            '13': {'up': [-1, '14'], 'down': [2, '15']},
            '14': {},
            '15': {},
            '16': {'up': [-1, '17'], 'down': [2, '24']},
            '17': {'up': [-1, '18'], 'down': [2, '21']},
            '18': {'up': [-1, '19'], 'down': [2, '20']},
            '19': {},
            '20': {},
            '21': {'up': [-1, '22'], 'down': [2, '23']},
            '22': {},
            '23': {},
            '24': {'up': [-1, '25'], 'down': [2, '28']},
            '25': {'up': [-1, '26'], 'down': [2, '27']},
            '26': {},
            '27': {},
            '28': {'up': [-1, '29'], 'down': [2, '30']},
            '29': {},
            '30': {}
        }

    elif (tree_type == "wide_breadth"):
        tree = "wide"
        # Your decision tree dictionary
        decision_tree = {
            '0': {
                'right': [-1, '1'], 'left': [3, '2'],
                'up': [-1, '3'], 'down': [2, '4'], 'up1': [-1, '5']
            },
            '1': {
                'up': [-1, '6'], 'down': [2, '7'],
                'right': [-1, '8'], 'left': [3, '9'], 'up1': [-1, '10']
            },
            '2': {
                'up': [-1, '11'], 'down': [2, '12'],
                'right': [-1, '13'], 'left': [3, '14'],
                'up1': [-1, '15']
            },
            '3': {
                'up': [-1, '16'], 'down': [2, '17'],
                'right': [-1, '18'], 'left': [3, '19'],
                'up1': [-1, '20']
            },
            '4': {
                'up': [-1, '21'], 'down': [2, '22'],
                'right': [-1, '23'], 'left': [3, '24'],
                'up1': [-1, '25']
            },
            '5': {
                'up': [-1, '26'], 'down': [2, '27'],
                'right': [-1, '28'], 'left': [3, '29'],
                'up1': [-1, '30']
            },
            '6': {},
            '7': {},
            '8': {},
            '9': {},
            '10': {},
            '11': {},
            '12': {},
            '13': {},
            '14': {},
            '15': {},
            '16': {},
            '17': {},
            '18': {},
            '19': {},
            '20': {},
            '21': {},
            '22': {},
            '23': {},
            '24': {},
            '25': {},
            '26': {},
            '27': {},
            '28': {},
            '29': {},
            '30': {}
        }

    else:

        tree = "wide"
        # Your decision tree dictionary
        decision_tree = {
            '0': {
                'right': [-1, '1'], 'left': [3, '7'],
                'up': [-1, '13'], 'down': [2, '19'],
                'up1': [-1, '25']
            },
            '1': {
                'up': [-1, '2'], 'down': [2, '3'],
                'right': [-1, '4'], 'left': [3, '5'], 'up1': [-1, '6']
            },
            '2': {},
            '3': {},
            '4': {},
            '5': {},
            '6': {},
            '7': {
                'up': [-1, '8'], 'down': [2, '9'],
                'right': [-1, '10'], 'left': [3, '11'],
                'up1': [-1, '12']
            },
            '8': {},
            '9': {},
            '10': {},
            '11': {},
            '12': {},
            '13': {
                'up': [-1, '14'], 'down': [2, '15'],
                'right': [-1, '16'], 'left': [3, '17'],
                'up1': [-1, '18']
            },
            '14': {},
            '15': {},
            '16': {},
            '17': {},
            '18': {},
            '19': {
                'up': [-1, '20'], 'down': [2, '21'],
                'right': [-1, '22'], 'left': [3, '23'],
                'up1': [-1, '24']
            },
            '20': {},
            '21': {},
            '22': {},
            '23': {},
            '24': {},
            '25': {
                'up': [-1, '26'], 'down': [2, '27'],
                'right': [-1, '28'], 'left': [3, '29'],
                'up1': [-1, '30']
            },
            '26': {},
            '27': {},
            '28': {},
            '29': {},
            '30': {}
        }


# ---------------------------------------------------------
# 6. PATH ANALYSIS & TENSOR MATRICES
# ---------------------------------------------------------
results = helper.analyze_tree_paths(decision_tree)

(path_names, path_leaf_dict, sibling_map, node_path_map, node_path_name,
 path_indices, node_indices, est_best_path_map, path_node_map) = results

num_paths = len(path_names)


# Create path map
path_map_np = np.zeros((num_paths, time_steps), dtype=int)
for i in range(num_paths):
    path_map_np[i, :] = [
        1 if f"{n+1}" in path_names[i] else 0 for n in range(time_steps)]
path_map = tf.convert_to_tensor(path_map_np, dtype=tf.float32)

# Create path covariance matrix
path_cov_mat_np = np.zeros((num_paths, num_paths, time_steps), dtype=int)
for i in range(num_paths):
    for j in range(num_paths):
        path_cov_mat_np[i, j, :] = path_map_np[i, :] * path_map_np[j, :]
path_cov_mat = tf.convert_to_tensor(path_cov_mat_np, dtype=tf.float32)

# Create index path map
index_path_map = {path_indices[i]: node_indices[i]
                  for i in range(len(path_indices))}
