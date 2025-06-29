# Copyright 2025 GIQ, Universitat Autònoma de Barcelona
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
config.py: the configuration for hamiltonian simulation task

"""

from datetime import datetime
from typing import Literal, Optional


################################################################
# CONFIGURATION CLASS
################################################################
class Config:
    def __init__(self):
        ########################################################################
        # CODE CONFIGURATION
        ########################################################################
        self.testing: bool = False  # True for testing mode, or False for single run

        # If testing = False: None for new training, or Timestamp String to load models
        self.load_timestamp: Optional[str] = None

        #######################################################################
        # TRAINING CONFIGURATION
        #######################################################################
        self.epochs: int = 3  # Number of epochs for training
        self.iterations_epoch: int = 100  # Number of iterations per epoch
        self.learning_rate: float = 0.01  # Unified learning rate for optimizers
        self.max_fidelity: float = 0.99  # Target fidelity to stop training
        # TODO: self.steps_gen: int = 1  # Number of steps in a row to update generator (#TODO: Discriminator has to be updated more frequently: 1 to 5)
        # TODO: self.steps_dis: int = 1  # Number of steps in a row to update discriminator

        #######################################################################
        # SYSTEM CONFIGURATION
        #######################################################################
        self.num_qubits: int = 3  # Number of main qubits for the system Hamiltonian

        # Initial state mode:
        # "zero": System starts in |0...0> state.
        # "choi": Uses Choi-Jamiolkowski isomorphism. Doubles num_qubits for the state representation (N_system for each part of Bell pair).
        self.initial_state: Literal["zero", "choi"] = "zero"
        # TODO: Make choi work, since currently doesn;t work well.

        # Ancilla Configuration:
        self.extra_ancilla: bool = False  # If True, adds an ancilla qubit.
        # TODO: We need to make ancilla same or more connected than normal qubits (e.g., ancilla_connectivity = "all" or "linear")
        self.ancilla_mode: Literal["pass", "project", "trace_out"] = (
            "pass"  # How the ancilla is handled. TODO: Implement "project" and "trace_out" modes.
        )

        # Ansatz Configuration: XX_YY_ZZ_Z, ZZ_X_Z or hardware efficient ansatz + Num of layers
        self.ansatz_gen: Literal["XX_YY_ZZ_Z", "ZZ_X_Z", "hardware_eff"] = "XX_YY_ZZ_Z"  # TODO: Implement HW efficient
        self.ansatz_disc: Literal["XX_YY_ZZ_Z", "ZZ_X_Z", "hardware_eff"] = "XX_YY_ZZ_Z"
        self.gen_layers: int = 10  # Number of layers in the generator ansatz
        self.disc_layers: int = 5  # Number of layers for the discriminator circuit

        # Target Hamiltonian: Cluster, Rotated Surface or User-provided hamiltonians. # TODO: Implement Rotated and Custom
        self.target_hamiltonian: Literal["cluster_h", "rotated_surface_h", "custom_h"] = "cluster_h"
        # TODO: self.custom_hamiltonian_obs: list[str] = ["X", "Y", "Z", "I"]  # & their combinations (XX, YXZ, ZZZZ, ...)
        # TODO: self.custom_hamiltonian_coeff: list[int] = [1, 1, 1, 1, 1, 1]  # Coefficients for the custom Hamiltonian obs

        # Cost function: Trace Norm, Squared Trace Norm, Wasserstein, etc...
        # TODO: self.cost_function: Literal["trace_norm", "squared_trace_norm" "wasserstein"] = "wasserstein"
        # Right now, only Wasserstein is implemented I think (from original qgan code)

        #####################################################################
        # SAVING AND LOGGING CONFIGURATION
        #####################################################################

        # Datetime for current run - initialized once
        self.run_timestamp: str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.base_data_path: str = f"./generated_data/{self.run_timestamp}_{self.num_qubits}qubits_{self.initial_state}state_{self.target_hamiltonian}"

        # File path settings (dynamic based on run_timestamp and num_qubits)
        self.figure_path: str = f"{self.base_data_path}/figure"
        self.model_gen_path: str = f"{self.base_data_path}/saved_model/model-gen(hs).npz"
        self.model_dis_path: str = f"{self.base_data_path}/saved_model/model-dis(hs).npz"
        self.log_path: str = f"{self.base_data_path}/logs/log.txt"
        self.fid_loss_path: str = f"{self.base_data_path}/fidelities/log_fidelity_loss.npy"
        self.theta_path: str = f"{self.base_data_path}/theta/theta_gen.txt"


####################################################################
# Global instance of the
####################################################################
CFG = Config()


#####################################################################
# Test Configurations
#####################################################################
test_configurations = [
    {
        "num_qubits": 2,
        "extra_ancilla": False,
        "epochs": 1,
        "iterations_epoch": 3,
        "disc_layers": 1,
        "gen_layers": 1,
        "label_suffix": "c1_2q_1l_noanc_d1_zero_XXYYZZZ",  # Corrected label
        "ansatz_gen": "XX_YY_ZZ_Z",
        "ansatz_disc": "XX_YY_ZZ_Z",
        "initial_state": "zero",  # Test with zero
        "ancilla_mode": "pass",
        "target_hamiltonian": "cluster_h",
    },
    {
        "num_qubits": 2,
        "extra_ancilla": True,
        "epochs": 1,
        "iterations_epoch": 3,
        "disc_layers": 1,
        "gen_layers": 1,
        "label_suffix": "c2_2q_1l_anc_d1_ZZXZ_choi",
        "ansatz_gen": "ZZ_X_Z",
        "ansatz_disc": "ZZ_X_Z",
        "initial_state": "choi",  # Test with choi
        "ancilla_mode": "pass",
        "target_hamiltonian": "cluster_h",
    },
    {
        "num_qubits": 2,
        "extra_ancilla": False,
        "epochs": 1,
        "iterations_epoch": 3,
        "disc_layers": 2,
        "gen_layers": 2,
        "label_suffix": "c3_2q_2l_noanc_d2_XXYYZZZ_zero",
        "ansatz_gen": "XX_YY_ZZ_Z",
        "ansatz_disc": "XX_YY_ZZ_Z",
        "initial_state": "zero",
        "ancilla_mode": "pass",  # Added ancilla_mode
        "target_hamiltonian": "cluster_h",  # Added target_hamiltonian for consistency, assuming it might be needed
    },
    {
        "num_qubits": 3,
        "extra_ancilla": False,
        "epochs": 1,
        "iterations_epoch": 3,
        "disc_layers": 1,
        "gen_layers": 1,
        "label_suffix": "c4_3q_1l_noanc_d1_ZZXZ_choi",
        "ansatz_gen": "ZZ_X_Z",
        "ansatz_disc": "ZZ_X_Z",
        "initial_state": "choi",
        "ancilla_mode": "pass",  # Added ancilla_mode
        "target_hamiltonian": "cluster_h",  # Added target_hamiltonian
    },
    # Add more configurations as needed
    {
        "num_qubits": 2,
        "extra_ancilla": False,
        "epochs": 1,
        "iterations_epoch": 3,
        "disc_layers": 1,
        "gen_layers": 1,
        "label_suffix": "c5_2q_1l_noanc_d1_ZZXZ_zero",
        "ansatz_gen": "ZZ_X_Z",
        "ansatz_disc": "ZZ_X_Z",
        "initial_state": "zero",
        "ancilla_mode": "pass",  # Added ancilla_mode
        "target_hamiltonian": "cluster_h",  # Added target_hamiltonian
    },
]
