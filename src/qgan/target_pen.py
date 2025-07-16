"""Target hamiltonian module"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import scipy
from scipy import linalg

import sys
from functools import reduce

from config import CFG


##############################################################
# MAIN FUNCTIONS FOR TARGET HAMILTONIAN
##############################################################
def get_target_unitary(target_type: str, size: int) -> np.ndarray:
    """Get the target unitary based on the target type and size.

    Args:
        target_type (str): Type of target Hamiltonian, either cluster_h, rotated_surface_h, ising_h, or custom_h.
        size (int): Size of the system.

    Returns:
        np.ndarray: The target unitary.
    """
    if target_type == "cluster_h":
        return construct_clusterH(size)
    if target_type == "rotated_surface_h":
        return construct_RotatedSurfaceCode(size)
    if target_type == "custom_h":
        return construct_target(size, CFG.custom_hamiltonian_terms, CFG.custom_hamiltonian_strengths)
    raise ValueError(f"Unknown target type: {target_type}. Expected 'cluster_h', 'rotated_surface_h', or 'custom_h'.")


##################################################################
# PREDEFINED TARGETS
##################################################################
def construct_target(size: int, terms: list[str], strengths: list[float]) -> np.ndarray:
    """Construct target Hamiltonian. Specify the terms to include as a list of strings.

    Args:
        size (int): the size of the system.
        terms (list[str]): which terms to include, e.g. ["I", "X", "Y", "Z", "XX", "XZ", "ZZ", "ZZZ", "ZZZZ", "XZX", "XXXX"]
        strengths (list[float]): the strengths of the terms, in the same order as `terms`.

    Returns:
        np.ndarray: the target Hamiltonian.
    """

    obs = []
    coeffs = []

    obs_sequences = to_obs_sequences(terms)
    for idx, obs_sequence in enumerate(obs_sequences):
        for i in range(size - len(obs_sequence) + 1):
            o = apply_obs_sequence(obs_sequence, i)

            # repeat coefficients to match the amount of times
            # the sequence has been applied
            coeffs.append(strengths[idx])
            obs.append(o)


    H = qml.Hamiltonian(coeffs=coeffs, observables=obs)
    return linalg.expm(-1j * qml.matrix(H).real)

def apply_obs_sequence(ob_sequence, start_wire):
    return reduce(lambda a, b: a @ b, [ob(start_wire + i) for i, ob in enumerate(ob_sequence)])

def to_obs_sequences(terms: list[str]) -> list[list[qml.operation.Operator]]:
    """Return the sequence of PennyLane operators """

    obs_sequences = []

    for term in terms:
        obs = []
        for let in term:
            ob = to_pauli(let)
            obs.append(ob)

        obs_sequences.append(obs)

    return obs_sequences

def to_pauli(let: str) -> qml.operation.Operator:
    """Return the corresponding PennyLane operator implementation according to the letter passed as argument."""

    if let == "X":
        return qml.X
    elif let == "Y":
        return qml.Y
    elif let == "Z":
        return qml.Z
    elif let == "I":
        return qml.I
    else:
        raise ValueError(f"No gate for {let}")


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