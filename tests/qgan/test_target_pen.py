import sys
import os

import numpy as np
import pennylane as qml

# This needs to be before any imports from src to ensure the correct path is set
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from qgan.target_pen import to_obs_sequences, construct_target, construct_RotatedSurfaceCode, construct_clusterH
from qgan.target import construct_target as oct
from qgan.target import construct_RotatedSurfaceCode as ocrsc
from qgan.target import construct_clusterH as occh

class TestTargetPen():
    def test_to_obs_sequences(self):
        terms = ["ZZZ", "ZZ", "XX", "XZ"]
        obs_expected = [[qml.Z, qml.Z, qml.Z], [qml.Z, qml.Z], [qml.X, qml.X], [qml.X, qml.Z]]
        obs_actual = to_obs_sequences(terms)

        assert obs_expected == obs_actual

        terms = ["I", "ZZ", "XX", "XZ", "XXXX"]
        obs_expected = [[qml.I], [qml.Z, qml.Z], [qml.X, qml.X], [qml.X, qml.Z], [qml.X, qml.X, qml.X, qml.X]]
        obs_actual = to_obs_sequences(terms)

        assert obs_expected == obs_actual


    def test_construct_target(self):
        terms = ["ZZZ", "ZZ", "XX", "XZ"]
        coeffs = [0.1, 0.2, 0.3, 0.4]
        size = 3

        H_expected = oct(size=size, terms=terms, strengths=coeffs)
        H_actual = construct_target(size, terms, coeffs)

        assert np.array_equal(H_actual, H_expected)

    def test_construct_RotatedSurfaceCode(self):
        size = 4
        rsc_expected = ocrsc(size)
        rsc_actual = construct_RotatedSurfaceCode(size)

        assert np.array_equal(rsc_expected, rsc_actual)

    def test_construct_clusterH(self):
        size = 3
        clusterH_expected = occh(size)
        clusterH_actual = construct_clusterH(size)

        assert np.array_equal(clusterH_expected, clusterH_actual)
