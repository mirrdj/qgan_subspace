from config import CFG
from src.tools.qobjects.qgates_pen import apply_quantum_gate

import pennylane as qml
import itertools
import numpy as np

# TODO: test if the result of the pennylane implementation of the circuit is the same as the original

def get_ansatz_type_circuit(type_of_ansatz: str) -> callable:
    """Construct the ansatz based on the type specified.

    Args:
        type_of_ansatz (str): Type of ansatz to construct, either 'XX_YY_ZZ_Z' or 'ZZ_X_Z'.

    Returns:
        callable: Function to construct the quantum circuit with the specified ansatz.
    """
    if type_of_ansatz == "XX_YY_ZZ_Z":
        return construct_qcircuit_XX_YY_ZZ_Z

    if type_of_ansatz == "ZZ_X_Z":
        return construct_qcircuit_ZZ_X_Z

    raise ValueError("Invalid type of ansatz specified.")

def count_gates_ZZ_X_Z(size: int, layer: int):
    """
    Count gates in order to make uniform random angles for the gates (0 to 2*pi)
    """
    gate_count = 0
    
    # First 1 qubit gates
    gate_count += (size * 2)

    # Ancilla 1q gates for: total, bridge and disconnected:
    if CFG.extra_ancilla and CFG.do_ancilla_1q_gates:
        gate_count += 2

    # Then 2 qubit gates
    gate_count += (size - 1)

    # Ancilla ancilla coupling (2q) logic for: total and bridge
    if CFG.extra_ancilla:
        if CFG.ancilla_topology == "total":
            gate_count += size
        if CFG.ancilla_topology == "bridge":
            gate_count += 1
        if CFG.ancilla_topology in ["bridge", "ansatz"]:
            gate_count += 1
    #
    return gate_count * layer
        

def construct_qcircuit_ZZ_X_Z(size: int, layer: int, theta):
    dev = qml.device("default.qubit", wires=size)

    # If extra ancilla is used, different than ansatz, we reduce the size by 1,
    # to implement the ancilla logic separately.
    if CFG.extra_ancilla:
        size -= 1

    def circuit():
        idx = 0
        for _ in range(layer):
            # First 1 qubit gates
            for i in range(size):
                apply_quantum_gate("X", i, -1, angle=theta[idx])
                idx += 1
                apply_quantum_gate("Z", i, -1, angle=theta[idx])
                idx += 1

    
            # # Ancilla 1q gates for: total, bridge and disconnected:
            if CFG.extra_ancilla and CFG.do_ancilla_1q_gates:
                apply_quantum_gate("X", size, -1, angle=theta[idx])
                idx += 1
                apply_quantum_gate("Z", size, -1, angle=theta[idx])
                idx += 1

                # Then 2 qubit gates
            for i in range(size - 1):
                apply_quantum_gate("ZZ", i, i + 1, angle=theta[idx])
                idx += 1

            # Ancilla ancilla coupling (2q) logic for: total and bridge
            if CFG.extra_ancilla:
                if CFG.ancilla_topology == "total":
                    for i in range(size):
                        apply_quantum_gate("ZZ", i, size, angle=theta[idx])
                        idx += 1

                if CFG.ancilla_topology == "bridge":
                    apply_quantum_gate("ZZ", 0, size, angle=theta[idx])
                    idx += 1

                if CFG.ancilla_topology in ["bridge", "ansatz"]:
                    qubit_to_connect_to = CFG.ancilla_connect_to if CFG.ancilla_connect_to is not None else size - 1
                    apply_quantum_gate("ZZ", qubit_to_connect_to, size, angle=theta[idx])
                    idx += 1


    return circuit

entg_list = ["XX", "YY", "ZZ"]
def count_gates_XX_YY_ZZ_Z(size: int, layer: int):
    """
    Count gates in order to make uniform random angles for the gates (0 to 2*pi)
    """

    gate_count = 0

    # First 1 qubit gates
    gate_count += size

    # Ancilla 1q gates for: total, bridge and disconnected:
    if CFG.extra_ancilla and CFG.do_ancilla_1q_gates:
        gate_count += 1

    # 2 qubit gates
    gate_count += (size - 1) * len(entg_list)

    # Ancilla coupling (2q) logic for: total and bridge
    if CFG.extra_ancilla:
        if CFG.ancilla_topology == "total":
            gate_count += size * len(entg_list)
        if CFG.ancilla_topology == "bridge":
            gate_count += len(entg_list)
        if CFG.ancilla_topology in ["bridge", "ansatz"]:
            gate_count += len(entg_list)

    return gate_count * layer


def construct_qcircuit_XX_YY_ZZ_Z(size: int, layer: int, theta):
    dev = qml.device("default.qubit", wires=size)

    # If extra ancilla is used, different than ansatz, we reduce the size by 1,
    # to implement the ancilla logic separately.
    if CFG.extra_ancilla:
        size -= 1

    def circuit():
        idx = 0
        for _ in range(layer):
            # First 1 qubit gates
            for i in range(size):
                apply_quantum_gate("Z", i, -1, angle=theta[idx])
                idx += 1

            # Ancilla 1q gates for: total, bridge and disconnected:
            if CFG.extra_ancilla and CFG.do_ancilla_1q_gates:
                apply_quantum_gate("Z", size, -1, angle=theta[idx])
                idx += 1

            # Then 2 qubit gates:
            for i, gate in itertools.product(range(size - 1), entg_list):
                apply_quantum_gate(gate, i, i + 1, angle=theta[idx])
                idx += 1

            # Ancilla ancilla coupling (2q) logic for: total and bridge
            if CFG.extra_ancilla:
                if CFG.ancilla_topology == "total":
                    for i, gate in itertools.product(range(size), entg_list):
                        apply_quantum_gate(gate, i, size, angle=theta[idx])
                        idx += 1

                if CFG.ancilla_topology == "bridge":
                    for gate in entg_list:
                        apply_quantum_gate(gate, 0, size, angle=theta[idx])
                        idx += 1

                if CFG.ancilla_topology in ["bridge", "ansatz"]:
                    qubit_to_connect_to = CFG.ancilla_connect_to if CFG.ancilla_connect_to is not None else size - 1
                    for gate in entg_list:
                        apply_quantum_gate(gate, qubit_to_connect_to, size, angle=theta[idx])
                        idx += 1


    return circuit


