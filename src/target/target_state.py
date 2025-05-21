# Copyright 2024 PennyLane Team
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

from pennylane import numpy as pnp  # Use PennyLane's numpy


def get_zero_state(num_qubits: int) -> pnp.ndarray:
    """Get the zero quantum state |0...0> as a PennyLane state vector.

    Args:
        num_qubits (int): The number of qubits in the system.

    Returns:
        pnp.ndarray: The zero state vector [1, 0, ..., 0].
    """
    state_vector = pnp.zeros(2**num_qubits, dtype=complex)
    state_vector[0] = 1.0
    return state_vector


def get_ghz_state(num_qubits: int) -> pnp.ndarray:
    """Get the GHZ (Greenberger–Horne–Zeilinger) state for `num_qubits`.
    The state is (|0...0> + |1...1>) / sqrt(2).

    Args:
        num_qubits (int): The number of qubits in the system.

    Returns:
        pnp.ndarray: The GHZ state vector.
    """
    if num_qubits <= 0:
        raise ValueError("Number of qubits must be positive.")

    state_vector = pnp.zeros(2**num_qubits, dtype=complex)
    state_vector[0] = 1.0 / pnp.sqrt(2)
    state_vector[-1] = 1.0 / pnp.sqrt(2)  # Index for |1...1> is 2^N - 1
    return state_vector


def get_maximally_entangled_state(num_qubits_per_register: int) -> pnp.ndarray:
    """Get the maximally entangled state (Bell state generalization) for two registers
    of size `num_qubits_per_register` each, i.e., total `2 * num_qubits_per_register` qubits.
    The state is (1/sqrt(2^N)) * sum_{i=0}^{2^N-1} |i>|i>.

    Args:
        num_qubits_per_register (int): The number of qubits in each of the two registers.

    Returns:
        pnp.ndarray: The maximally entangled state vector.
    """
    total_qubits = 2 * num_qubits_per_register
    dim_register = 2**num_qubits_per_register
    state_vector = pnp.zeros(2**total_qubits, dtype=complex)

    for i in range(dim_register):
        state_vector[i * dim_register + i] = 1.0

    state_vector /= pnp.sqrt(dim_register)
    return state_vector


def get_maximally_entangled_state_in_subspace(num_initial_qubits_per_register: int) -> pnp.ndarray:
    """Get the maximally entangled state for a system where each of the two parties
    has `num_initial_qubits_per_register` plus one ancillary qubit in the |0> state.
    So, total qubits = 2 * (num_initial_qubits_per_register + 1).
    The state is (1/sqrt(2^N)) * sum_{i=0}^{2^N-1} (|i>|0>)_A (|i>|0>)_B,
    where N = num_initial_qubits_per_register.

    Args:
        num_initial_qubits_per_register (int): The number of initial qubits in each party before adding ancillas.

    Returns:
        pnp.ndarray: The maximally entangled state vector in the specified subspace.
    """
    dim_initial_register = 2**num_initial_qubits_per_register
    num_qubits_per_party_with_ancilla = num_initial_qubits_per_register + 1
    dim_party_with_ancilla = 2**num_qubits_per_party_with_ancilla
    total_qubits_with_ancillas = 2 * num_qubits_per_party_with_ancilla

    state_vector = pnp.zeros(2**total_qubits_with_ancillas, dtype=complex)

    for i in range(dim_initial_register):
        idx_party_A = i * 2
        idx_party_B = i * 2
        final_idx = idx_party_A * dim_party_with_ancilla + idx_party_B
        state_vector[final_idx] = 1.0

    state_vector /= pnp.sqrt(dim_initial_register)
    return state_vector


def get_real_denmat(system_size: int, prob_real: list[float], input_states: list[pnp.ndarray]) -> pnp.ndarray:
    """Construct the density matrix for the real state as a mixture of pure states.
    rho_real = sum_i p_i |psi_i><psi_i|

    Args:
        system_size (int): The number of qubits in the system.
        prob_real (list[float]): List of probabilities p_i for each pure state in the mixture.
        input_states (list[pnp.ndarray]): List of pure state vectors |psi_i>.

    Returns:
        pnp.ndarray: The density matrix rho_real.
    """
    if not input_states:
        return pnp.zeros((2**system_size, 2**system_size), dtype=complex)

    density_matrix = pnp.zeros((2**system_size, 2**system_size), dtype=complex)

    for i, prob in enumerate(prob_real):
        state_vec = input_states[i]
        if state_vec.ndim == 1:
            state_vec_col = state_vec[:, pnp.newaxis]
        else:
            state_vec_col = state_vec

        outer_prod = state_vec_col @ pnp.conj(state_vec_col.T)
        density_matrix += prob * outer_prod

    return density_matrix
