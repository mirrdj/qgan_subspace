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

import sys

import numpy as np  # Add standard numpy
import pennylane as qml
import scipy.linalg  # Add scipy.linalg
from pennylane import numpy as pnp  # Use PennyLane's numpy


def construct_target(
    num_qubits: int, Z_terms: bool = False, ZZ_terms: bool = False, ZZZ_terms: bool = False, I_term: bool = False
) -> pnp.ndarray:
    """Construct target Hamiltonian unitary U = exp(-iH).

    Args:
        num_qubits (int): The number of qubits in the system.
        Z_terms (bool): Whether to include single Z terms.
        ZZ_terms (bool): Whether to include ZZ interaction terms.
        ZZZ_terms (bool): Whether to include ZZZ interaction terms.
        I_term (bool): If True, H is the identity matrix, overriding other terms.

    Returns:
        pnp.ndarray: The unitary matrix exp(-iH).
    """
    if I_term:
        H_matrix = pnp.eye(2**num_qubits, dtype=complex)
    else:
        coeffs = []
        obs = []
        if Z_terms:
            for i in range(num_qubits):
                coeffs.append(1.0)
                obs.append(qml.PauliZ(i))
        if ZZ_terms:
            for i in range(num_qubits - 1):
                coeffs.append(1.0)
                obs.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
        if ZZZ_terms:
            for i in range(num_qubits - 2):
                coeffs.append(1.0)
                obs.append(qml.PauliZ(i) @ qml.PauliZ(i + 1) @ qml.PauliZ(i + 2))

        if not obs:  # No terms selected, H is effectively zero
            H_matrix = pnp.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
        else:
            hamiltonian = qml.Hamiltonian(coeffs, obs)
            H_matrix_qml = qml.matrix(hamiltonian, wire_order=range(num_qubits))
            if H_matrix_qml is None:  # Should not happen if obs is not empty
                H_matrix = pnp.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
            else:
                # Ensure H_matrix is a pnp.ndarray
                H_matrix = pnp.array(H_matrix_qml, dtype=complex)

    H_np = np.asarray(H_matrix)  # Convert to standard NumPy array
    exp_H_np = scipy.linalg.expm(-1j * H_np)
    return pnp.array(exp_H_np, dtype=complex)


def construct_clusterH(num_qubits: int) -> pnp.ndarray:
    """Construct cluster Hamiltonian unitary U = exp(-iH).

    Args:
        num_qubits (int): The number of qubits in the system.

    Returns:
        pnp.ndarray: The unitary matrix exp(-iH) for the cluster Hamiltonian.
    """
    coeffs = []
    obs = []

    for i in range(num_qubits - 2):
        coeffs.append(1.0)
        obs.append(qml.PauliX(i) @ qml.PauliZ(i + 1) @ qml.PauliX(i + 2))
        coeffs.append(1.0)
        obs.append(qml.PauliZ(i))

    coeffs.append(1.0)
    obs.append(qml.PauliZ(num_qubits - 2))
    coeffs.append(1.0)
    obs.append(qml.PauliZ(num_qubits - 1))

    if not obs:
        H_matrix = pnp.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    else:
        hamiltonian = qml.Hamiltonian(coeffs, obs)
        H_matrix_qml = qml.matrix(hamiltonian, wire_order=range(num_qubits))
        if H_matrix_qml is None:
            H_matrix = pnp.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
        else:
            H_matrix = pnp.array(H_matrix_qml, dtype=complex)

    H_np = np.asarray(H_matrix)  # Convert to standard NumPy array
    exp_H_np = scipy.linalg.expm(-1j * H_np)
    return pnp.array(exp_H_np, dtype=complex)


def construct_RotatedSurfaceCode(num_qubits: int) -> pnp.ndarray:
    """Construct rotated surface code Hamiltonian unitary U = exp(-iH).

    Args:
        num_qubits (int): The number of qubits in the system. Must be 4 or 9.

    Returns:
        pnp.ndarray: The unitary matrix exp(-iH) for the rotated surface code.
    """
    coeffs = []
    obs = []

    if num_qubits == 4:
        # -XXXX(0,1,2,3)
        coeffs.append(-1.0)
        obs.append(qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliX(2) @ qml.PauliX(3))
        # -ZZ(0,1)
        coeffs.append(-1.0)
        obs.append(qml.PauliZ(0) @ qml.PauliZ(1))
        # -ZZ(2,3)
        coeffs.append(-1.0)
        obs.append(qml.PauliZ(2) @ qml.PauliZ(3))
    elif num_qubits == 9:
        # -XXXX(0,1,3,4)
        coeffs.append(-1.0)
        obs.append(qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliX(3) @ qml.PauliX(4))
        # -XXXX(4,5,7,8)
        coeffs.append(-1.0)
        obs.append(qml.PauliX(4) @ qml.PauliX(5) @ qml.PauliX(7) @ qml.PauliX(8))
        # -XX(2,5)
        coeffs.append(-1.0)
        obs.append(qml.PauliX(2) @ qml.PauliX(5))
        # -XX(3,6)
        coeffs.append(-1.0)
        obs.append(qml.PauliX(3) @ qml.PauliX(6))
        # -ZZZZ(1,2,4,5)
        coeffs.append(-1.0)
        obs.append(qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(4) @ qml.PauliZ(5))
        # -ZZZZ(3,4,6,7)
        coeffs.append(-1.0)
        obs.append(qml.PauliZ(3) @ qml.PauliZ(4) @ qml.PauliZ(6) @ qml.PauliZ(7))
        # -ZZ(0,1)
        coeffs.append(-1.0)
        obs.append(qml.PauliZ(0) @ qml.PauliZ(1))
        # -ZZ(7,8)
        coeffs.append(-1.0)
        obs.append(qml.PauliZ(7) @ qml.PauliZ(8))
    else:
        sys.exit(f"RotatedSurfaceCode not defined for system size {num_qubits}. Only 4 or 9.")

    if not obs:  # Should not happen given the logic, but as a safeguard
        H_matrix = pnp.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    else:
        hamiltonian = qml.Hamiltonian(coeffs, obs)
        H_matrix_qml = qml.matrix(hamiltonian, wire_order=range(num_qubits))
        if H_matrix_qml is None:
            H_matrix = pnp.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
        else:
            H_matrix = pnp.array(H_matrix_qml, dtype=complex)

    H_np = np.asarray(H_matrix)  # Convert to standard NumPy array
    exp_H_np = scipy.linalg.expm(-1j * H_np)
    return pnp.array(exp_H_np, dtype=complex)
