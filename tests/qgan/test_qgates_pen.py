import sys
import os
import pytest

import numpy as np

# This needs to be before any imports from src to ensure the correct path is set
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from tools.qobjects.qgates_pen import quantum_gate
from tools.qobjects.qgates import QuantumGate

class TestQGates():
    @pytest.mark.parametrize("angle,qubit1,qubit2,name", [
        (np.pi / 3, 0, 1, "ZZ"),
        (np.pi / 4, 0, 1, "YY"),
        (np.pi / 2, 0, 1, "XX"),
        (np.pi / 3, 0, 1, "XX"),
        (np.pi / 4, 0, 1, "XX"),
    ])
    def test_quantum_gate_two_qubit(self, angle, qubit1, qubit2, name):
        qg = QuantumGate(name, qubit1, qubit2, angle=angle)
        matrix_expected = qg.matrix_representation(2, False)
        gate = quantum_gate(name)
        matrix_actual = gate(angle, name, [qubit1, qubit2]).matrix()

        print(matrix_expected)
        print("\n")
        print(matrix_actual)
        print("----------")
        print("\n\n")

        phase = np.vdot(matrix_expected.flatten(), matrix_actual.flatten()) / np.linalg.norm(matrix_expected) ** 2

        assert np.allclose(matrix_expected, phase * matrix_actual)


    @pytest.mark.parametrize("angle,qubit1,qubit2,name", [
        (np.pi / 3, 0, 1, "Z"),
        (np.pi / 4, 0, 1, "X"),
        (np.pi / 2, 0, 1, "Y"),
    ])
    def test_quantum_gate_one_qubit(self, angle, qubit1, qubit2, name):
        qg = QuantumGate(name, qubit1, qubit2, angle=angle)
        matrix_expected = qg.matrix_representation(1, False)
        gate = quantum_gate(name)
        matrix_actual = gate(angle, name, [qubit1]).matrix()

        assert np.allclose(matrix_expected, matrix_actual)

    def test_quantum_gate_global_phase(self):
        name = "G"
        qubit1 = 0
        qubit2 = -1
        angle = np.pi / 4

        qg = QuantumGate(name, qubit1, qubit2, angle=angle)
        matrix_expected = qg.matrix_representation(1, False)
        gate = quantum_gate(name)
        matrix_actual = gate(angle).matrix()

        assert np.allclose(matrix_expected, matrix_actual)

    def test_quantum_gate_cnot(self):
        name = "CNOT"
        qubit1 = 0
        qubit2 = 1
        angle = np.pi / 4

        qg = QuantumGate(name, qubit1, qubit2, angle=angle)
        matrix_expected = qg.matrix_representation(2, False)
        gate = quantum_gate(name)
        matrix_actual = gate(angle).matrix()

        assert np.allclose(matrix_expected, matrix_actual)

