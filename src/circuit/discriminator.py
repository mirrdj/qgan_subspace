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

"""Discriminator class for the Quantum GAN using PennyLane."""

import os  # Import os module

import numpy as np  # For initializations or non-gradient parts
import pennylane as qml
from pennylane import numpy as pnp

from circuit.ansatz import get_ansatz_and_shape  # Import the new helper
from training.optimizer import MomentumOptimizer


# TODO: Check that this is similar to the original code
class Discriminator:
    def __init__(
        self, num_qubits: int, num_disc_layers: int, ansatz_type: str, learning_rate: float
    ):  # Added learning_rate
        self.num_qubits = num_qubits  # N_eff
        self.num_disc_layers = num_disc_layers
        self.dev_disc = qml.device("default.qubit", wires=self.num_qubits)

        # Get ansatz and shape function based on type
        self.ansatz_fn, self.params_shape_fn = get_ansatz_and_shape(ansatz_type)

        # Define the ansatz for the discriminator circuit
        params_shape = self.params_shape_fn(self.num_qubits, self.num_disc_layers)
        self.params_disc = pnp.array(np.random.uniform(low=0, high=2 * np.pi, size=params_shape), requires_grad=True)

        self._discriminator_circuit_qnode = self._create_qnode()

        self.optimizer_disc = MomentumOptimizer(learning_rate=learning_rate)  # Pass learning_rate to optimizer

        self.grad_params_disc_fn = qml.grad(self.discriminator_cost_function, argnum=0)

    def _discriminator_circuit(self, state_vector_to_evaluate, disc_circuit_params):
        """
        Defines the discriminator's quantum circuit structure.
        This method is intended to be wrapped by a QNode.
        """
        qml.StatePrep(state_vector_to_evaluate, wires=range(self.num_qubits))

        # Use the selected ansatz function
        self.ansatz_fn(self.num_qubits, self.num_disc_layers, disc_circuit_params)

        return qml.expval(qml.PauliZ(0))

    def _create_qnode(self):
        """Creates and returns the QNode for the discriminator circuit."""
        return qml.QNode(self._discriminator_circuit, self.dev_disc, interface="autograd")

    def discriminator_cost_function(
        self,
        current_params_disc,
        fake_state_vector,
        real_state_vector,
    ):
        """
        Computes the cost for the discriminator.
        Cost to minimize: E_fake - E_real.
        """
        output_fake = self._discriminator_circuit_qnode(fake_state_vector, current_params_disc)
        output_real = self._discriminator_circuit_qnode(real_state_vector, current_params_disc)

        cost = output_fake - output_real
        return cost

    def _calculate_gradients(self, current_params_disc, fake_state_vector, real_state_vector):
        """Calculates gradients for params_disc."""
        grad_val = self.grad_params_disc_fn(current_params_disc, fake_state_vector, real_state_vector)
        return grad_val

    def update_dis(self, generator, real_target_state, input_state_for_generator):
        """
        Performs one update step for the discriminator's parameters.
        """
        fake_state_vector = generator.get_generated_state_vector(
            params_gen=generator.params_gen, input_state_subspace_M_eff_qubits=input_state_for_generator
        )

        fake_state_pnp = pnp.asarray(fake_state_vector, dtype=complex)
        real_target_pnp = pnp.asarray(real_target_state, dtype=complex)

        grad_params_val = self._calculate_gradients(self.params_disc, fake_state_pnp, real_target_pnp)

        current_params_disc_np = (
            self.params_disc.numpy() if hasattr(self.params_disc, "numpy") else np.array(self.params_disc)
        )
        grad_params_val_np = grad_params_val.numpy() if hasattr(grad_params_val, "numpy") else np.array(grad_params_val)

        new_params_disc_flat = self.optimizer_disc.compute_grad(
            current_params_disc_np.flatten(), grad_params_val_np.flatten(), "min"
        )

        self.params_disc = pnp.array(
            pnp.reshape(pnp.array(new_params_disc_flat), self.params_disc.shape), requires_grad=True
        )

    def get_params(self):
        """Returns the discriminator's trainable parameters."""
        return self.params_disc

    def save_model(self, file_path):
        """Saves the discriminator's parameters to a file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create directory if it doesn't exist
        pnp.savez(
            file_path,
            params_disc=self.params_disc.numpy() if hasattr(self.params_disc, "numpy") else np.array(self.params_disc),
        )

    def load_model(self, file_path):
        """Loads the discriminator's parameters from a file."""
        data = pnp.load(file_path, allow_pickle=False)
        loaded_params = data["params_disc"]

        # Get expected shape using the params_shape_fn
        expected_shape = self.params_shape_fn(self.num_qubits, self.num_disc_layers)
        if loaded_params.shape != expected_shape:
            raise ValueError(
                f"Loaded discriminator parameters shape {loaded_params.shape} "
                f"does not match expected shape {expected_shape} for N={self.num_qubits}, layers={self.num_disc_layers}"
            )

        self.params_disc = pnp.array(loaded_params, requires_grad=True)
