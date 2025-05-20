##### ansatz for generator


import numpy as np
import pennylane as qml


def construct_qcircuit_XX_YY_ZZ_Z(wires: int, layer: int, params: np.ndarray):
    """Construct a quantum circuit with the ansatz of XX, YY, ZZ rotations and Z rotations using PennyLane.

    Args:
        wires (int): Number of qubits (system_size).
        layer (int): Number of layers in the ansatz.
        params (np.ndarray): Array of rotation parameters. Expected shape: (layer, num_gates_per_layer).
                             num_gates_per_layer = (size * 4) for XX, YY, ZZ, Z on each qubit (with wraparound for 2-qubit gates)
    """
    param_idx = 0
    for j in range(layer):
        for i in range(wires):
            # Two-qubit gates
            qml.IsingXX(params[param_idx], wires=[i, (i + 1) % wires])
            param_idx += 1
            qml.IsingYY(params[param_idx], wires=[i, (i + 1) % wires])
            param_idx += 1
            qml.IsingZZ(params[param_idx], wires=[i, (i + 1) % wires])
            param_idx += 1

            # Single-qubit gate
            qml.RZ(params[param_idx], wires=i)
            param_idx += 1
    # Note: The original code had a specific structure for the last qubit in terms of two-qubit gates
    # and also applied Z to the last qubit separately.
    # The (i + 1) % wires handles the wraparound for a ring connectivity.
    # The original also had a random initialization of theta inside, which is now expected to be passed via `params`.


def get_params_shape_XX_YY_ZZ_Z(wires: int, layer: int):
    """Returns the shape of the parameters needed for construct_qcircuit_XX_YY_ZZ_Z."""
    # Each layer has:
    #   wires * 3 (for XX, YY, ZZ on i, i+1)
    #   wires * 1 (for Z on i)
    # Total params per layer = wires * 4
    return (layer * wires * 4,)


def construct_qcircuit_ZZ_X_Z(wires: int, layer: int, params: np.ndarray):
    """Construct a quantum circuit with the ansatz of ZZ, X and Z rotations using PennyLane.

    Args:
        wires (int): Number of qubits (system_size).
        layer (int): Number of layers in the ansatz.
        params (np.ndarray): Array of rotation parameters. Expected shape: (layer, num_gates_per_layer)
                             num_gates_per_layer = (size * 2) for X, Z on each qubit + (size-1) for ZZ
    """
    param_idx = 0
    for j in range(layer):
        for i in range(wires):
            qml.RX(params[param_idx], wires=i)  # Original used X, assuming RX
            param_idx += 1
            qml.RZ(params[param_idx], wires=i)  # Original used Z, assuming RZ
            param_idx += 1
        for i in range(wires - 1):
            qml.IsingZZ(params[param_idx], wires=[i, i + 1])
            param_idx += 1
    # The original also had a random initialization of theta inside, which is now expected to be passed via `params`.


def get_params_shape_ZZ_X_Z(wires: int, layer: int):
    """Returns the shape of the parameters needed for construct_qcircuit_ZZ_X_Z."""
    # Each layer has:
    #   wires * 2 (for X, Z on each qubit)
    #   (wires - 1) * 1 (for ZZ on i, i+1)
    # Total params per layer = wires * 2 + (wires - 1)
    num_params_per_layer = wires * 2 + (wires - 1)
    return (layer * num_params_per_layer,)
