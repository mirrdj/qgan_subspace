"""
config.py: the configuration for hamiltonian simulation task

"""

from datetime import datetime

import numpy as np

################################################################
# START OF PARAMETERS TO CHANGE:
################################################################

# Parameter for costs functions and gradients
lamb = float(10)

# Learning Scripts
extra_ancilla = False  # True # False
iterations_epoch = 100
epochs = 10
eta = 1e-1  # initial learning rate
# TODO: Eta its not being used!

# System setting
system_size = 8  # #3 #4
layer = 10  # #20 #15 #10 #4 #3 #2
num_discriminator_layers = 2  # Number of layers for the discriminator circuit

# Set to None for new training or a specific timestamp string to load models
load_ts_for_training = None  # "2025-05-21_00-12-38"
# Example: load_ts_for_training = "2024-01-01_12-00-00"  # Replace with actual timestamp string


################################################################
# END OF PARAMETERS TO CHANGE:
################################################################

# Costs and gradients
s = np.exp(-1 / (2 * lamb)) - 1
cst1 = (s / 2 + 1) ** 2
cst2 = (s / 2) * (s / 2 + 1)
cst3 = (s / 2) ** 2

# Datetime
curr_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# File settings
figure_path = f"./generated_data/{curr_datetime}/figure"
model_gen_path = f"./generated_data/{curr_datetime}/saved_model/{system_size}qubit_model-gen(hs).npz"
model_dis_path = f"./generated_data/{curr_datetime}/saved_model/{system_size}qubit_model-dis(hs).npz"
log_path = f"./generated_data/{curr_datetime}/logs/{system_size}qubit_log.txt"
fid_loss_path = f"./generated_data/{curr_datetime}/fidelities/{system_size}qubit_log_fidelity_loss.npy"
theta_path = f"./generated_data/{curr_datetime}/theta/{system_size}qubit_theta_gen.txt"
