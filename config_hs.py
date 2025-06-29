#!/usr/bin/env python

"""
config_hs.py: the configuration for hamiltonian simulation task

"""

from datetime import datetime

import numpy as np

# Learning Scripts
# from tools.utils import get_maximally_entangled_state, get_maximally_entangled_state_in_subspace

lamb = float(10)
s = np.exp(-1 / (2 * lamb)) - 1
cst1 = (s / 2 + 1) ** 2
cst2 = (s / 2) * (s / 2 + 1)
cst3 = (s / 2) ** 2

# Learning Scripts
initial_eta = 1e-1
epochs = 300
decay = False
eta = initial_eta

# Log
label = "hs"
# fidelities = list()
# losses = list()

# System setting
system_size = 3  # 3 #4
layer = 3  # 20 #15 #10 #4 #3 #2 #4

# input_state = get_maximally_entangled_state(system_size)
# input_state = get_maximally_entangled_state_in_subspace(system_size)

# file settings
run_timestamp: str = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
base_data_path: str = f"./generated_data/{run_timestamp}"
figure_path = f"{base_data_path}/figure"
model_gen_path = f"{base_data_path}/saved_model/{system_size}qubit_model-gen(hs).mdl"
model_dis_path = f"{base_data_path}/saved_model/{system_size}qubit_model-dis(hs).mdl"
