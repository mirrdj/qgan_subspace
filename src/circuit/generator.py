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
        num_qubits_N: int,  # N_eff: qubits for the ansatz U_G
        num_qubits_M: int,  # M_eff: qubits for the input subspace state |psi_M>
        layer: int,
        ansatz_type: str,  # New argument for ansatz type
        learning_rate: float,  # New argument for learning rate
    ):
        self.num_qubits_N = num_qubits_N
        self.num_qubits_M = num_qubits_M
        self.layer = layer

        # Get ansatz and shape function based on type
        self.ansatz_fn, self.params_shape_fn = get_ansatz_and_shape(ansatz_type)

        self.dev_ansatz = qml.device("default.qubit", wires=self.num_qubits_N)

        params_shape = self.params_shape_fn(self.num_qubits_N, self.layer)
        # Initialize with a small standard deviation for stability
        self.params_gen = pnp.array(np.random.normal(0, 0.1, params_shape), requires_grad=True)

        self.optimizer = MomentumOptimizer(learning_rate=learning_rate)  # Pass learning_rate to optimizer

        self._ansatz_qnode_for_state = qml.QNode(self._ansatz_circuit_state_prep, self.dev_ansatz, interface="autograd")
        self._ansatz_qnode_for_unitary = qml.QNode(self._ansatz_circuit_unitary, self.dev_ansatz, interface="autograd")

    def _ansatz_circuit_state_prep(self, params, initial_state_vector_N_eff):
        """Applies ansatz to a given initial state vector on N_eff qubits."""
        qml.StatePrep(initial_state_vector_N_eff, wires=range(self.num_qubits_N))
        self.ansatz_fn(self.num_qubits_N, self.layer, params)
        return qml.state()

    def _ansatz_circuit_unitary(self, params):
        """Circuit to get the unitary matrix of the ansatz on N_eff qubits."""
        self.ansatz_fn(self.num_qubits_N, self.layer, params)
        return qml.state()

    def get_ansatz_unitary(self, params_gen):
        """Returns the unitary matrix U_G(params_gen) of the generator's ansatz (acts on N_eff qubits)."""
        return qml.matrix(self._ansatz_qnode_for_unitary)(params=params_gen)

    def get_generated_state_vector(self, params_gen, input_state_subspace_M_eff_qubits):
        """
        Generates the fake state vector: U_G(params_gen) |psi_input_N_eff>,
        where |psi_input_N_eff> = |input_state_subspace_M_eff_qubits (x) |0...0>_{N_eff-M_eff}>.
        """
        dim_N_eff = 2**self.num_qubits_N
        dim_M_eff = 2**self.num_qubits_M

        if self.num_qubits_M > self.num_qubits_N:
            raise ValueError(
                f"M_eff (num_qubits_M={self.num_qubits_M}) cannot be larger than N_eff (num_qubits_N={self.num_qubits_N})"
            )

        initial_state_N_eff_vec = pnp.zeros(dim_N_eff, dtype=complex)
        input_state_subspace_pnp = pnp.asarray(input_state_subspace_M_eff_qubits, dtype=complex).flatten()

        if len(input_state_subspace_pnp) != dim_M_eff:
            raise ValueError(
                f"Provided input_state_subspace_M_eff_qubits (length {len(input_state_subspace_pnp)}) "
                f"does not match expected dimension for M_eff={self.num_qubits_M} (2**{self.num_qubits_M}={dim_M_eff})"
            )

        if self.num_qubits_M == self.num_qubits_N:
            if dim_M_eff != dim_N_eff:
                raise ValueError("Dimension mismatch even when M_eff == N_eff. This shouldn't happen.")
            initial_state_N_eff_vec = input_state_subspace_pnp
        else:
            initial_state_N_eff_vec[0:dim_M_eff] = input_state_subspace_pnp[0:dim_M_eff]

        fake_state_vec = self._ansatz_qnode_for_state(params_gen, initial_state_N_eff_vec)
        return fake_state_vec

    def generator_cost_function(self, params_gen, discriminator, input_state_subspace_M_eff_qubits):
        """
        Computes the cost for the generator.
        Generator wants to maximize discriminator's output D(fake_state).
        So, cost_G = -D(fake_state) to be minimized.
        `discriminator` is the Discriminator object.
        `input_state_subspace_M_eff_qubits` is |psi_M>.
        """
        fake_state_vec = self.get_generated_state_vector(params_gen, input_state_subspace_M_eff_qubits)

        disc_params_fixed = pnp.array(discriminator.params_disc, requires_grad=False)

        disc_output_on_fake = discriminator._discriminator_circuit_qnode(fake_state_vec, disc_params_fixed)

        cost = -disc_output_on_fake
        return cost

    def _calculate_gradients(self, discriminator, input_state_subspace_M_eff_qubits):
        """Calculates gradients of the generator's cost function w.r.t. params_gen."""
        input_state_pnp = pnp.asarray(input_state_subspace_M_eff_qubits, dtype=complex, requires_grad=False)

        grad_fn = qml.grad(self.generator_cost_function, argnum=0)
        gradients = grad_fn(self.params_gen, discriminator, input_state_pnp)
        return gradients

    def update_gen(self, discriminator, input_state_subspace_M_eff_qubits):
        """Update generator's parameters using gradients."""
        gradients = self._calculate_gradients(discriminator, input_state_subspace_M_eff_qubits)

        current_params_flat_np = (
            self.params_gen.flatten().numpy()
            if hasattr(self.params_gen, "numpy")
            else np.array(self.params_gen.flatten())
        )
        gradients_flat_np = (
            gradients.flatten().numpy() if hasattr(gradients, "numpy") else np.array(gradients.flatten())
        )

        new_params_flat = self.optimizer.compute_grad(current_params_flat_np, gradients_flat_np, "min")
        self.params_gen = pnp.array(pnp.reshape(pnp.array(new_params_flat), self.params_gen.shape), requires_grad=True)

    def save_model(self, file_path):
        """Saves the generator's parameters to a file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create directory if it doesn't exist
        params_to_save = self.params_gen.numpy() if hasattr(self.params_gen, "numpy") else np.array(self.params_gen)
        pnp.savez(file_path, params_gen=params_to_save)

    def load_model(self, file_path):
        """Loads the generator's parameters from a file."""
        data = pnp.load(file_path, allow_pickle=False)
        loaded_params = data["params_gen"]

        expected_shape = self.params_shape_fn(self.num_qubits_N, self.layer)
        if loaded_params.shape != expected_shape:
            raise ValueError(
                f"Loaded generator parameters shape {loaded_params.shape} "
                f"does not match expected shape {expected_shape} for N={self.num_qubits_N}, layers={self.layer}"
            )

        self.params_gen = pnp.array(loaded_params, requires_grad=True)
