import itertools
import sys
import os

# This needs to be before any imports from src to ensure the correct path is set
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import pickle
import numpy as np
from qgan.discriminator import Discriminator
from qgan.generator import Generator
from tools.qobjects.qgates import Identity
from config import CFG

class TestGenerator():

    def test_init(self):
        gen = Generator(np.matrix(np.ones((2**(CFG.system_size * 2 + (1 if CFG.extra_ancilla else 0)), 1))))
        assert gen.size == CFG.system_size + (1 if CFG.extra_ancilla else 0)
        assert gen.qc is not None
        
    def test_get_total_gen_state(self):
        gen = Generator(np.matrix(np.ones((2**(CFG.system_size * 2 + (1 if CFG.extra_ancilla else 0)), 1))))
        gen_state = gen.get_total_gen_state()
        assert gen_state is not None
        
        total_size = gen.size+gen.target_size
        assert gen_state.shape == (2**total_size, 1)
        
    def test_get_Untouched_qubits_x_Gen_matrix(self):
        gen = Generator(np.matrix(np.ones((2**(CFG.system_size * 2 + (1 if CFG.extra_ancilla else 0)), 1))))
        gen_matrix = np.kron(Identity(CFG.system_size), gen.qc.get_mat_rep()) # This is the matrix before the product with initial state
        assert gen_matrix is not None
        
        total_size = gen.size+gen.target_size
        assert gen_matrix.shape == (2**total_size, 2**total_size)
        
        # Check that the partial trace of size CFG.system_size is gen.qc.get_mat_rep() 
        partial_trace = gen_matrix[:2**gen.size, :2**gen.size]
        gen_mat_rep = gen.qc.get_mat_rep()
        assert np.allclose(partial_trace, gen_mat_rep)

    def test_update_gen(self):
        final_target_state = np.matrix(np.ones((2**(CFG.system_size * 2 + (1 if CFG.extra_ancilla and CFG.ancilla_mode == "pass" else 0)), 1)))
        gen = Generator(np.matrix(np.ones((2**(CFG.system_size * 2 + (1 if CFG.extra_ancilla else 0)), 1))))
        dis = Discriminator()
        
        old_angles = [g.angle for g in gen.qc.gates].copy()
        gen.update_gen(dis, final_target_state)
        
        # Check that it updates the theta parameters
        new_angles = [g.angle for g in gen.qc.gates].copy()
        assert np.any(old_angles != new_angles)

    def test_all_ansatz_types(self):
        for ansatz in ["XX_YY_ZZ_Z", "ZZ_X_Z"]:
            CFG.gen_ansatz = ansatz
            CFG.gen_layers = 1 
            CFG.extra_ancilla = False
            gen = Generator(np.matrix(np.ones((2**(CFG.system_size * 2 + (1 if CFG.extra_ancilla else 0)), 1))))
            assert gen.qc is not None
            assert len(gen.qc.gates) >= 1

    def test_ancilla_modes_and_topologies(self):
        # Try all combinations of ancilla_mode and ancilla_topology
        modes = ["pass", "project", "trace"]
        topologies = ["disconnected", "ansatz", "bridge", "total"]
        for mode, topo in itertools.product(modes, topologies):
            CFG.extra_ancilla = True
            CFG.ancilla_mode = mode
            CFG.ancilla_topology = topo
            CFG.system_size = 2
            CFG.gen_layers = 1
            gen = Generator(np.matrix(np.ones((2**(CFG.system_size * 2 + (1 if CFG.extra_ancilla else 0)), 1))))
            assert gen.qc is not None
            assert len(gen.qc.gates) >= 1

    def test_save_and_load_with_various_ansatz(self):
        for ansatz in ["XX_YY_ZZ_Z", "ZZ_X_Z"]:
            CFG.gen_ansatz = ansatz
            CFG.gen_layers = 1 
            CFG.extra_ancilla = False
            gen = Generator(np.matrix(np.ones((2**(CFG.system_size * 2 + (1 if CFG.extra_ancilla else 0)), 1))))
            for gate in gen.qc.gates:
                gate.angle = 0.456
            path = f"test_gen_{ansatz}.pkl"
            with open(path, "wb") as f:
                pickle.dump(gen, f)
            gen2 = Generator(np.matrix(np.ones((2**(CFG.system_size * 2 + (1 if CFG.extra_ancilla else 0)), 1))))
            assert gen2.load_model_params(path)
            assert any(g.angle == 0.456 for g in gen2.qc.gates)
            os.remove(path)