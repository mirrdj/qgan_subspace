import sys
import os

# This needs to be before any imports from src to ensure the correct path is set
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from config import CFG
from qgan.ancilla import (
    get_final_gen_state_for_discriminator,
    get_max_entangled_state_with_ancilla_if_needed,
    project_ancilla_zero,
    trace_out_ancilla,
)
import numpy as np
from qgan.target import get_target_unitary, get_final_target_state
from qgan.generator import Generator
from qgan.discriminator import Discriminator
from qgan.cost_functions import compute_fidelity
from tools.qobjects.qgates import Identity

class TestAncilla():
    
    def test_get_maximally_entangled_state(self):
        for ancilla in [False, True]:
            CFG.extra_ancilla = ancilla
            for ancilla_mode in ["pass", "project"]:
                CFG.ancilla_mode = ancilla_mode
                total_state, final_state = get_max_entangled_state_with_ancilla_if_needed(CFG.system_size)
                for state in [total_state, final_state]:
                    assert state is not None
                    assert len(state.shape) == 2
                assert total_state.shape[0] == 2 ** (CFG.system_size * 2 + (1 if CFG.extra_ancilla else 0))
                assert final_state.shape[0] == 2 ** (CFG.system_size * 2 + (1 if CFG.extra_ancilla and CFG.ancilla_mode == "pass" else 0))
        
    def test_final_gen_state_project(self):
        state = np.ones((2 ** (CFG.system_size * 2 + 1), 1))
        CFG.extra_ancilla = True
        CFG.ancilla_mode = "project"
        result = get_final_gen_state_for_discriminator(state)
        assert result is not None
        assert result.shape[0] == 2 ** (CFG.system_size * 2)  # size halved
        assert result.shape[1] == 1
        expected = state[::2]
        exp_norm = np.linalg.norm(expected)
        assert (result == expected/exp_norm).all() # Keep even indices

    def test_final_gen_state_pass(self):
        state = np.ones((2 ** (CFG.system_size * 2 + 1), 1))
        CFG.extra_ancilla = True
        CFG.ancilla_mode = "pass"
        result = get_final_gen_state_for_discriminator(state)
        assert result is not None
        assert result.shape[0] == 2 ** (CFG.system_size * 2 + 1)  # size unchanged
        assert result.shape[1] == 1
        assert (result == state).all() # Unchanged state

    def test_final_gen_state_trace(self):
        state = np.ones((2 ** (CFG.system_size * 2 + 1), 1))
        CFG.extra_ancilla = True
        CFG.ancilla_mode = "trace"
        result = get_final_gen_state_for_discriminator(state)
        assert result is not None
        assert result.shape[0] == 2 ** (CFG.system_size * 2)  # size halved
        assert result.shape[1] == 1
        # Don't know how to compare state easily

    def test_final_target_state_project(self):
        CFG.extra_ancilla = True
        CFG.ancilla_mode = "project"
        CFG.target_hamiltonian = "cluster_h"
        state = np.ones((2 ** (CFG.system_size * 2), 1)) # No +1 since its in project mode
        result = get_final_target_state(state)
        assert result is not None
        assert result.shape[0] == 2 ** (CFG.system_size * 2)  # size halved
        assert result.shape[1] == 1
        assert (result != state).any() # Changed state

    def test_final_target_state_pass(self):
        CFG.extra_ancilla = True
        CFG.ancilla_mode = "pass"
        CFG.target_hamiltonian = "cluster_h"
        state = np.ones((2 ** (CFG.system_size * 2 + 1), 1)) # +1 since its in pass mode
        result = get_final_target_state(state)
        assert result is not None
        assert result.shape[0] == 2 ** (CFG.system_size * 2 + 1)  # size unchanged
        assert result.shape[1] == 1
        assert (result != state).any() # Changed state

    def test_final_target_state_trace(self):
        CFG.extra_ancilla = True
        CFG.ancilla_mode = "trace"
        CFG.target_hamiltonian = "cluster_h"
        state = np.ones((2 ** (CFG.system_size * 2), 1)) # No +1 since its in trace mode
        result = get_final_target_state(state)
        assert result is not None
        assert result.shape[0] == 2 ** (CFG.system_size * 2)  # size halved
        assert result.shape[1] == 1
        # Don't know how to compare state easily

    ####################################################################################
    # MORE CONCRETE TESTS FOR PROJECTING AND TRACING OUT ANCILLA
    ####################################################################################
    def test_project_ancilla_zero_correct_subspace(self):
        # |00> + |10> = [1,0,1,0] (ancilla is last qubit)
        state = np.array([[1,0,1,0]]).T / np.sqrt(2)
        projected, prob = project_ancilla_zero(state)
        expected = np.array([[1,1]]).T / np.sqrt(2)
        assert np.allclose(projected, expected)
        assert abs(prob - 1.0) < 1e-8

    def test_project_ancilla_zero_zero_norm(self):
        # All zeros
        CFG.system_size = 2
        state = np.zeros((2**(CFG.system_size*2+1),1))
        projected, prob = project_ancilla_zero(state)
        assert np.allclose(projected, np.zeros((2**(CFG.system_size*2),1)))
        assert prob == 0.0

    def test_trace_out_ancilla_pure_state(self):
        # |00> + |11> = [1,0,0,1] (Bell state)
        state = np.array([[1,0,0,1]]).T / np.sqrt(2)
        sampled = trace_out_ancilla(state)
        assert sampled.shape == (2,1)
        assert np.allclose(np.linalg.norm(sampled), 1.0)
    
    ####################################################################################
    # INTEGRATION TESTS FOR ANCILLA LOGIC WITH DISCRIMINATOR AND TARGET
    ####################################################################################
    def test_ancilla_logic_integration_with_Identity_target_and_gen(self):
        # Check that ancilla logic matches for all modes
        CFG.target_hamiltonian = "custom_h"
        CFG.custom_hamiltonian_terms = ["I"]
        CFG.extra_ancilla = True
        CFG.gen_layers = 1
        CFG.gen_ansatz = "ZZ_X_Z"
        CFG.system_size = 2
        
        for topology in ["disconnected", "ansatz", "total"]:
            CFG.ancilla_topology = topology
            
            ###################################################
            # Expected final state, without ancilla.
            ###################################################
            expected_state = np.zeros(2 ** (2 * CFG.system_size), dtype=complex)
            dim_register = 2**CFG.system_size
            for i in range(dim_register):
                expected_state[i * dim_register + i] = 1.0
            expected_state /= np.sqrt(dim_register)
            expected_state_with_no_phase = np.asmatrix(expected_state).T
            # Effect of exp(-1j*I)= exp(-1j) * eigen_state, which is just a phase factor
            expected_state_with_phase = np.exp(-1j) * expected_state_with_no_phase
            
            ###################################################
            # Get, input, target and gen states:
            ###################################################
            total_input_state, final_input_state = get_max_entangled_state_with_ancilla_if_needed(CFG.system_size)
            final_target_state = get_final_target_state(final_input_state)
            gen = Generator(total_input_state)
            
            # Set the generator to the Identity (as target), but with some arbitrary rotations to check ancilla logic:
            for change_gates_in_X_qubit_of_gen in ["none", "ancilla", "first", "second"]:
                # Set the angles of all gates to mimic the identity operation:
                for gate in gen.qc.gates:
                    gate.angle = 0.0
                # Except for the ancilla qubit, where we can to set any same rotation.
                if change_gates_in_X_qubit_of_gen == "ancilla":
                    for gate in gen.qc.gates:
                        if gate.qubit1 == CFG.system_size and (topology == "disconnected" or gate.qubit2 is None):
                            gate.angle = np.random.uniform(0, 2 * np.pi)
                # Change gates not in the ancilla (first and second qubits):
                if change_gates_in_X_qubit_of_gen == "first":
                    for gate in gen.qc.gates:
                        if gate.qubit1== 0 and gate.qubit2 is None:
                            gate.angle = np.random.uniform(0, 2 * np.pi)
                if change_gates_in_X_qubit_of_gen == "second":
                    for gate in gen.qc.gates:
                        if gate.qubit1== 1 and gate.qubit2 is None:
                            gate.angle = np.random.uniform(0, 2 * np.pi)
                    
                total_gen_state = gen.get_total_gen_state()

                ###################################################
                # Comparison:
                ###################################################
                for mode in ['pass', 'project', 'trace']:
                    CFG.ancilla_mode = mode
                    dis = Discriminator() # Since the states are equal, doesn't matter which alpha/beta cost should same.
                    gen_out = get_final_gen_state_for_discriminator(total_gen_state)
                    
                    if mode == 'pass':
                            assert if_exactly_equal_with_allclose_and_fidelity(gen_out, total_gen_state)
                            #
                            
                    # IF GENERATOR WORKS CORRECTLY WITH ANCILLA LOGIC, in the last subspace/qubit, THESE SHOULD WORK:
                    elif change_gates_in_X_qubit_of_gen in ["none", "ancilla"]:                    
                        if mode == 'project':
                            # For 'project' gen will have a random phase, from the expm(-1j*gates):
                            assert if_equal_up_to_global_phase_with_fidelity(gen_out, expected_state_with_no_phase) # The generated case, has also a phase!
                        if mode == 'trace':
                            # For 'trace', gen will have no phase (from the sampling):
                            assert if_exactly_equal_with_allclose_and_fidelity(gen_out, expected_state_with_no_phase)
                        
                        # For both 'project' and 'trace', target will have only exp(-1j) phase exactly:
                        assert if_exactly_equal_with_allclose_and_fidelity(final_target_state, expected_state_with_phase)
                        # Also up to a phase gen and target will match!
                        assert if_equal_up_to_global_phase_with_fidelity(gen_out, final_target_state)
                    
                    # AND THESE SHOULD NOT, SINCE I'M GIVING RANDOM ROTATIONS TO THE ACTUAL GENERATOR GATES (no ancilla)
                    elif change_gates_in_X_qubit_of_gen in ["first", "second"]:
                        if mode == 'project':
                            # For 'project', gen_out should be the expected state, with the same phase
                            assert not if_equal_up_to_global_phase_with_fidelity(gen_out, expected_state_with_phase)
                        if mode == 'trace':
                            # For 'trace', gen_out should be the expected state but with no phase (sampling)
                            assert not if_exactly_equal_with_allclose_and_fidelity(gen_out, expected_state_with_no_phase)
                            
                        # For both 'project' and 'trace', target will have only exp(-1j) phase exactly:
                        assert if_exactly_equal_with_allclose_and_fidelity(final_target_state, expected_state_with_phase)
                        # Also up to a phase gen and target will match!
                        assert not if_equal_up_to_global_phase_with_fidelity(gen_out, final_target_state)
               
    def test_ancilla_logic_integration_with_actual_Hamiltonians_no_gen(self):
        # Check that ancilla logic matches for all modes
        CFG.custom_hamiltonian_terms = ["Z", "ZZ", "ZZZ"]
        CFG.extra_ancilla = True
        
        ###################################################
        # Expected final state, without ancilla.
        ###################################################
        max_entangled_no_ancilla = np.zeros(2 ** (2 * CFG.system_size), dtype=complex)
        dim_register = 2**CFG.system_size
        for i in range(dim_register):
            max_entangled_no_ancilla[i * dim_register + i] = 1.0
        max_entangled_no_ancilla /= np.sqrt(dim_register)
        max_entangled_no_ancilla = np.asmatrix(max_entangled_no_ancilla).T

        ###################################################
        # Get, input and target states:
        ###################################################
        _, final_input_state = get_max_entangled_state_with_ancilla_if_needed(CFG.system_size)
        # Random total_gen_state
        
        for ham in ["cluster_h", "custom_h"]:
            CFG.target_hamiltonian = ham
        
            final_target_state = get_final_target_state(final_input_state)
            
            # Expected Target state without ancilla:
            target_unitary = get_target_unitary(CFG.target_hamiltonian, CFG.system_size)
            target_op = np.kron(Identity(CFG.system_size), target_unitary)
            expected_state = np.matmul(target_op, max_entangled_no_ancilla)

            ###################################################
            # Comparison:
            ###################################################
            for mode in ['pass', 'project', 'trace']:
                CFG.ancilla_mode = mode
                
                if mode == 'project':
                    # For 'project', target_out should be the expected state, with the same phase
                    assert if_exactly_equal_with_allclose_and_fidelity(final_target_state, expected_state)
                elif mode == 'trace':
                    # For, target_out should be the expected state but with no phase (sampling)
                    assert if_exactly_equal_with_allclose_and_fidelity(final_target_state, expected_state)
                # Check normalization
                assert np.isclose(np.linalg.norm(final_target_state), 1.0), "Target output should be normalized, without need to explicitly doing so (T x|0>)"

def if_exactly_equal_with_allclose_and_fidelity(state1, state2):
    """Check if two states are equal using np.allclose and fidelity."""
    return np.allclose(state1, state2) and \
        np.isclose(compute_fidelity(state1, state2), 1.0)
        
        
def if_equal_up_to_global_phase_with_fidelity(state1, state2):
    """Check if two states are equal using fidelity."""
    return np.isclose(compute_fidelity(state1, state2), 1.0)