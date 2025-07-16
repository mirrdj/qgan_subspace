import pennylane as qml


##################################################################
#   The parameter shift rule is implemented automatically in PennyLane
#   so there is no need to manually implement it separately as it
#   is done in the original numpy implementation.
##################################################################

def apply_quantum_gate(name, qubit1, qubit2, angle):
    """
    To be called within the quantum circuit to apply the desired quantum gate.
    """
    gate = quantum_gate(name)

    if name in ["XX", "YY", "ZZ"]:
        gate(angle, name, [qubit1, qubit2])
    elif name in ["X", "Y", "Z"]:
        gate(angle, name, qubit1)
    elif name == "G":
        gate(angle)
    elif name == "CNOT":
        return qml.CNOT([qubit1, qubit2])


def quantum_gate(name):
    """
    Retrieve the desired gate.
    """

    if name in ["XX", "YY", "ZZ"]: # TODO: FIX
        return qml.PauliRot
    elif name in ["X", "Y", "Z"]:
        return qml.PauliRot
    elif name == "G":  # TODO: FIX
        return qml.GlobalPhase
    elif name == "CNOT":  # TODO: FIX
        return qml.CNOT
    else:
        raise ValueError("Gate is not defined")
