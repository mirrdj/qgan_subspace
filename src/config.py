"""
config.py: the configuration for hamiltonian simulation task

"""

from datetime import datetime
from typing import TYPE_CHECKING, List, Literal, Optional

import numpy as np

if TYPE_CHECKING:
    import pennylane as qml


################################################################
# START OF PARAMETERS TO CHANGE:
################################################################


class Config:
    def __init__(self):
        # Parameter for original costs functions and gradients (if still used)
        self.lamb: float = 10.0

        # Learning Scripts
        self.extra_ancilla: bool = False  # If True, adds an ancilla to the generator's output space
        self.iterations_epoch: int = 100
        self.epochs: int = 10

        # Learning rate
        self.learning_rate: float = 0.01  # Unified learning rate for optimizers

        # System setting
        self.system_size: int = 8  # Number of main qubits for the target state
        self.generator_layers: int = 10  # Number of layers in the generator ansatz
        self.discriminator_layers: int = 2  # Number of layers for the discriminator circuit

        # Initial state for the generator
        # "zero": |0...0> state
        # "maximally_entangled": GHZ-like state if applicable, or other forms of maximal entanglement
        self.generator_initial_state_type: Literal["zero", "maximally_entangled"] = "maximally_entangled"

        # Ansatz types
        # "XX_YY_ZZ_Z": A specific ansatz structure
        # "ZZ_X_Z": Another specific ansatz structure
        self.ansatz_gen_type: Literal["XX_YY_ZZ_Z", "ZZ_X_Z"] = "XX_YY_ZZ_Z"  # Updated Literal and default
        self.ansatz_disc_type: Literal["XX_YY_ZZ_Z", "ZZ_X_Z"] = "XX_YY_ZZ_Z"  # Updated Literal and default

        # Target state/Hamiltonian configuration
        # "cluster_state_1d": 1D cluster state
        # "ghz_state": GHZ state
        # "tfim_ground_state": Ground state of Transverse Field Ising Model
        # "custom_hamiltonian": Ground state of a user-defined Hamiltonian
        # "custom_state_vector": A user-provided state vector
        self.target_type: Literal[
            "cluster_state_1d", "ghz_state", "tfim_ground_state", "custom_hamiltonian", "custom_state_vector"
        ] = "cluster_state_1d"

        # Parameters for "tfim_ground_state"
        self.tfim_h_param: float = 1.0  # Transverse field strength for TFIM

        # Parameters for "custom_hamiltonian"
        self.custom_hamiltonian_coeffs: Optional[List[float]] = None
        self.custom_hamiltonian_ops: Optional[List["qml.Observable"]] = None  # Use string literal for qml.Observable

        # Parameter for "custom_state_vector"
        self.custom_target_state_vector: Optional[np.ndarray] = None

        # Cost function for the GAN
        self.cost_function_type: Literal["original_qgan", "wasserstein_like", "fidelity_direct"] = "original_qgan"

        # Set to None for new training or a specific timestamp string to load models
        self.load_ts_for_training: Optional[str] = None

        # Datetime for current run - initialized once
        self.run_timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Derived parameters for 'original_qgan' cost function
        if self.cost_function_type == "original_qgan":
            self.s: float = np.exp(-1 / (2 * self.lamb)) - 1
            self.cst1: float = (self.s / 2 + 1) ** 2
            self.cst2: float = (self.s / 2) * (self.s / 2 + 1)
            self.cst3: float = (self.s / 2) ** 2
        else:
            self.s = self.cst1 = self.cst2 = self.cst3 = None

        # File path settings (dynamic based on run_timestamp and system_size)
        self.base_data_path: str = f"./generated_data/{self.run_timestamp}"
        self.figure_path: str = f"{self.base_data_path}/figure"
        self.saved_model_dir: str = f"{self.base_data_path}/saved_model"
        self.model_gen_path_template: str = f"{self.saved_model_dir}/{{qubits}}qubit_model-gen(hs).npz"
        self.model_dis_path_template: str = f"{self.saved_model_dir}/{{qubits}}qubit_model-dis(hs).npz"
        self.log_path_template: str = f"{self.base_data_path}/logs/{{qubits}}qubit_log.txt"
        self.fid_loss_path_template: str = f"{self.base_data_path}/fidelities/{{qubits}}qubit_log_fidelity_loss.npy"
        self.theta_path_template: str = f"{self.base_data_path}/theta/{{qubits}}qubit_theta_gen.txt"

    def get_model_gen_path(self) -> str:
        return self.model_gen_path_template.format(qubits=self.system_size)

    def get_model_dis_path(self) -> str:
        return self.model_dis_path_template.format(qubits=self.system_size)

    def get_log_path(self) -> str:
        return self.log_path_template.format(qubits=self.system_size)

    def get_fid_loss_path(self) -> str:
        return self.fid_loss_path_template.format(qubits=self.system_size)

    def get_theta_path(self) -> str:
        return self.theta_path_template.format(qubits=self.system_size)


# Global instance of the configuration
CFG = Config()
