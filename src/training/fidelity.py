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

# Cost and Fidelities file

import pennylane.numpy as pnp  # Use PennyLane's numpy

from circuit.generator import Generator


def compute_fidelity(
    gen: Generator, input_to_gen_m_qubits_np: pnp.ndarray, target_real_state_nm_qubits_np: pnp.ndarray
) -> float:
    """Calculate the fidelity between the generated state and the target real state.
    Fidelity = |<target_real_state|generated_state>|^2.

    Args:
        gen (Generator): The generator object.
        input_to_g_m_qubits_np (pnp.ndarray): The input state vector for the generator (M qubits).
        target_real_state_nm_qubits_np (pnp.ndarray): The target real state vector (N+M qubits).

    Returns:
        float: The fidelity.
    """
    # Ensure inputs are PennyLane numpy arrays and detached
    input_to_g_m_pnp = pnp.array(input_to_gen_m_qubits_np, dtype=complex, requires_grad=False).flatten()
    target_real_state_nm_pnp = pnp.array(target_real_state_nm_qubits_np, dtype=complex, requires_grad=False).flatten()

    # Calculate generated state: G |input_to_G>
    # The get_generated_state_vector method handles the tensoring with |0...0>_N-M internally
    # and applies the generator U_G.
    # It expects params_gen to be set within the generator object.
    generated_state_nm_pnp = gen.get_generated_state_vector(
        params_gen=gen.params_gen, input_subspace_state_vector=input_to_g_m_pnp
    )

    # Fidelity: |<target_real_state|generated_state>|^2
    # pnp.vdot(a, b) computes a_conj * b (inner product)
    overlap = pnp.vdot(target_real_state_nm_pnp, generated_state_nm_pnp)
    fidelity = pnp.abs(overlap) ** 2

    return float(pnp.real(fidelity))
