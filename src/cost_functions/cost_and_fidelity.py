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

# Cost and Fidelities file

import pennylane.numpy as pnp  # Use PennyLane's numpy

from discriminator.discriminator import Discriminator  # Changed to absolute import
from generator.generator import Generator  # Changed to absolute import


def compute_cost(
    gen: Generator, dis: Discriminator, real_state_nm_qubits_np: pnp.ndarray, input_to_g_nm_qubits_np: pnp.ndarray
) -> float:
    """Calculate the discriminator's objective value L_D.
    L_D = Tr(rho_real Psi) - Tr(rho_fake Phi) - Regularization.
    This function calls the discriminator's internal cost function which returns -L_D.

    Args:
        gen (Generator): The generator object.
        dis (Discriminator): The discriminator object.
        real_state_nm_qubits_np (pnp.ndarray): The real target state vector (N+M qubits).
        input_to_g_nm_qubits_np (pnp.ndarray): The input state vector for the generator matrix G (N+M qubits,
                                              e.g., |0...0>_N (tensor) |0...0>_M).

    Returns:
        float: The value of the discriminator's objective function L_D.
    """
    # This function is now obsolete due to discriminator redesign.
    # Returning 0.0 as a placeholder.
    # The actual GAN loss should be handled in the training loop based on
    # generator and discriminator costs.
    return 0.0
    # Ensure inputs are PennyLane numpy arrays and detached for value calculation
    # real_state_nm_pnp = pnp.array(real_state_nm_qubits_np, dtype=complex, requires_grad=False).flatten()
    # input_to_g_nm_pnp = pnp.array(input_to_g_nm_qubits_np, dtype=complex, requires_grad=False).flatten()

    # Get current generator matrix G = U_G (tensor) I_M (as PennyLane array, detached)
    # gen.params should already be a pnp.array
    # g_matrix_pnp = gen.get_full_generator_matrix(gen.params_gen) # Corrected params to params_gen
    # g_matrix_val = pnp.array(g_matrix_pnp.numpy(), requires_grad=False)  # Detach

    # dis.alpha and dis.beta are pnp.array(requires_grad=True)
    # The discriminator_cost_function handles requires_grad internally for its arguments.

    # Call the discriminator's cost function. It returns -L_D.
    # neg_l_d = dis.discriminator_cost_function(
    #     dis.alpha, # This attribute no longer exists
    #     dis.beta,  # This attribute no longer exists
    #     g_matrix_val,  # G(theta)
    #     real_state_nm_pnp,  # rho_real (state vector)
    #     input_to_g_nm_pnp,  # input state for G to produce rho_fake
    # )
    # We want to return L_D
    # return float(pnp.real(-neg_l_d))


def compute_fidelity(
    gen: Generator, input_to_g_m_qubits_np: pnp.ndarray, target_real_state_nm_qubits_np: pnp.ndarray
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
    input_to_g_m_pnp = pnp.array(input_to_g_m_qubits_np, dtype=complex, requires_grad=False).flatten()
    target_real_state_nm_pnp = pnp.array(target_real_state_nm_qubits_np, dtype=complex, requires_grad=False).flatten()

    # Calculate generated state: G |input_to_G>
    # The get_generated_state_vector method handles the tensoring with |0...0>_N-M internally
    # and applies the generator U_G.
    # It expects params_gen to be set within the generator object.
    generated_state_nm_pnp = gen.get_generated_state_vector(
        params_gen=gen.params_gen, input_state_subspace_M_eff_qubits=input_to_g_m_pnp
    )

    # Fidelity: |<target_real_state|generated_state>|^2
    # pnp.vdot(a, b) computes a_conj * b (inner product)
    overlap = pnp.vdot(target_real_state_nm_pnp, generated_state_nm_pnp)
    fidelity = pnp.abs(overlap) ** 2

    return float(pnp.real(fidelity))
