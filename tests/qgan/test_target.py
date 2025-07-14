import sys
import os

import numpy as np
# This needs to be before any imports from src to ensure the correct path is set
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from qgan.target import get_target_unitary, get_final_target_state
from qgan.ancilla import get_max_entangled_state_with_ancilla_if_needed
from config import CFG

class TestTarget():
    def test_get_target_unitary(self):
        unitary = get_target_unitary(CFG.target_hamiltonian, CFG.system_size)
        assert unitary is not None
        assert unitary.shape[0] == unitary.shape[1] == 2**CFG.system_size
        # Check that the unitary is unitary
        assert np.isclose(unitary @ unitary.conj().T, np.eye(2**CFG.system_size)).all()

    def test_initialize_target_state(self):
        for ancilla in [False, True]:
            CFG.extra_ancilla = ancilla
            unitary = get_target_unitary(CFG.target_hamiltonian, CFG.system_size)
            _, final_target_state = get_max_entangled_state_with_ancilla_if_needed(CFG.system_size)
            state = get_final_target_state(final_target_state)
            assert state is not None
            assert state.shape[0] == 2**(unitary.shape[0]+(1 if CFG.extra_ancilla and CFG.ancilla_mode == "pass" else 0))
            # Check that the state is a valid quantum state
            assert np.isclose(np.linalg.norm(state), 1.0)
            
