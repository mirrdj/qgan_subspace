"""Generator class for the Quantum GAN."""

import os  # Import os module

import numpy as np  # Keep standard numpy for operations not involving params
import pennylane as qml
from pennylane import numpy as pnp  # Use PennyLane's NumPy for automatic differentiation

from circuit.ansatz import get_ansatz_and_shape  # Import the new helper
from training.optimizer import MomentumOptimizer


# TODO: Check that this is similar to the original code
class Generator:
    def __init__(
        self,
        num_qubits: int,  # Number of qubits the generator's ansatz U_G operates on
        layer: int,
        ansatz_type: str,  # New argument for ansatz type
        learning_rate: float,  # New argument for learning rate
    ):
        self.num_qubits = num_qubits
        self.layer = layer

        # Get ansatz and shape function based on type
        self.ansatz_fn, self.params_shape_fn = get_ansatz_and_shape(ansatz_type)

        self.dev_ansatz = qml.device("default.qubit", wires=self.num_qubits)

        params_shape = self.params_shape_fn(self.num_qubits, self.layer)
        # Initialize with a small standard deviation for stability
        self.params_gen = pnp.array(np.random.normal(0, 0.1, params_shape), requires_grad=True)

        self.optimizer = MomentumOptimizer(learning_rate=learning_rate)  # Pass learning_rate to optimizer

        self._ansatz_qnode_for_state = qml.QNode(self._ansatz_circuit_state_prep, self.dev_ansatz, interface="autograd")
        # self._ansatz_qnode_for_unitary is not strictly needed if get_ansatz_unitary defines its own QNode
        # self._ansatz_qnode_for_unitary = qml.QNode(
        #     self._ansatz_circuit_unitary, self.dev_ansatz, interface="autograd"
        # )

    def _ansatz_circuit_state_prep(self, params, input_state_vector):
        """Applies ansatz to a given input_state_vector on self.num_qubits."""
        qml.StatePrep(input_state_vector, wires=range(self.num_qubits))
        self.ansatz_fn(self.num_qubits, self.layer, params)
        return qml.state()

    def _ansatz_circuit_unitary(self, params):
        """Circuit to define operations for unitary matrix construction on self.num_qubits."""
        self.ansatz_fn(self.num_qubits, self.layer, params)
        # qml.matrix() is applied by the caller (get_ansatz_unitary) to a QNode wrapping this.
        # It needs some operation to define the space, Identity is fine.
        return qml.apply(qml.Identity(0))  # Placeholder, actual matrix derived from operations

    def get_ansatz_unitary(self, params_gen):
        """Returns the unitary matrix U_G(params_gen) of the generator's ansatz (acts on self.num_qubits)."""

        @qml.qnode(self.dev_ansatz, interface="autograd")
        def actual_unitary_circuit(params):
            self.ansatz_fn(self.num_qubits, self.layer, params)
            # The qml.matrix() decorator requires the circuit to return an operator or a list of operators.
            # Here, we want the matrix of the entire circuit defined by ansatz_fn.
            # We need to specify the wires for which the matrix is requested.
            # Passing wires positionally due to TypeError with keyword argument.
            return qml.matrix(self.ansatz_fn)

        return actual_unitary_circuit(params=params_gen)

    def get_generated_state_vector(self, params_gen, input_state_vector):
        """
        Generates the fake state vector: U_G(params_gen) |input_state_vector>.
        The |input_state_vector> must be a valid state for self.num_qubits.
        """
        dim_operating_qubits = 2**self.num_qubits
        input_state_pnp = pnp.asarray(input_state_vector, dtype=complex).flatten()

        if len(input_state_pnp) != dim_operating_qubits:
            raise ValueError(
                f"Provided input_state_vector (length {len(input_state_pnp)}) "
                f"does not match expected dimension for num_qubits={self.num_qubits} (2**{self.num_qubits}={dim_operating_qubits})"
            )
        fake_state_vec = self._ansatz_qnode_for_state(params_gen, input_state_pnp)
        return fake_state_vec

    def update_params(self, gradients):
        """Update generator's parameters using provided gradients."""
        current_params_flat_np = (
            self.params_gen.flatten().numpy()
            if hasattr(self.params_gen, "numpy")
            else np.array(self.params_gen.flatten())
        )
        # Ensure gradients are also flat numpy arrays
        gradients_flat_np = (
            gradients.flatten().numpy() if hasattr(gradients, "numpy") else np.array(gradients.flatten())
        )

        if current_params_flat_np.shape != gradients_flat_np.shape:
            raise ValueError(
                f"Shape mismatch between flattened params ({current_params_flat_np.shape}) "
                f"and gradients ({gradients_flat_np.shape}) in Generator.update_params"
            )

        new_params_flat = self.optimizer.compute_grad(current_params_flat_np, gradients_flat_np, "min")
        self.params_gen = pnp.array(pnp.reshape(pnp.array(new_params_flat), self.params_gen.shape), requires_grad=True)

    def get_params(self):
        """Returns the generator's trainable parameters."""
        return self.params_gen

    def save_model(self, file_path):
        """Saves the generator's parameters to a file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create directory if it doesn't exist
        params_to_save = self.params_gen.numpy() if hasattr(self.params_gen, "numpy") else np.array(self.params_gen)
        pnp.savez(file_path, params_gen=params_to_save)

    def load_model(self, file_path):
        """Loads the generator's parameters from a file."""
        data = pnp.load(file_path, allow_pickle=False)
        loaded_params = data["params_gen"]

        expected_shape = self.params_shape_fn(self.num_qubits, self.layer)
        if loaded_params.shape != expected_shape:
            raise ValueError(
                f"Loaded generator parameters shape {loaded_params.shape} "
                f"does not match expected shape {expected_shape} for num_qubits={self.num_qubits}, layers={self.layer}"
            )

        self.params_gen = pnp.array(loaded_params, requires_grad=True)
