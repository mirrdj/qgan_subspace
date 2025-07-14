import sys
import os

import numpy as np

# This needs to be before any imports from src to ensure the correct path is set
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))


from qgan.target import get_final_target_state
from qgan.ancilla import get_final_gen_state_for_discriminator
from qgan.generator import Generator
from qgan.discriminator import Discriminator
from config import CFG
from tools.qobjects.qgates import I, X, Y, Z

class TestDiscriminator():

    def test_init(self):
        dis_size=CFG.system_size * 2 + (1 if CFG.extra_ancilla and CFG.ancilla_mode == "pass" else 0)
        dis = Discriminator()
        # Asserts
        assert dis.size == dis_size
        assert dis.alpha.shape == (dis_size, 4)
        assert dis.beta.shape == (dis_size, 4)
        assert dis.ancilla == CFG.extra_ancilla
        assert dis.ancilla_mode == CFG.ancilla_mode
        assert dis.target_size == CFG.system_size
        assert dis.herm == [I, X, Y, Z]

    def test_getPsi_getPhi(self):
        for extra_ancilla in [False, True]:
            CFG.extra_ancilla = extra_ancilla
            for ancilla_mode in ["pass", "project"]:
                CFG.ancilla_mode = ancilla_mode
               
                # Checks shapes 
                dis = Discriminator()
                psi, phi = dis.get_psi_and_phi()
                assert psi.shape == phi.shape == (2**dis.alpha.shape[0], 2**dis.alpha.shape[0])

    def test_update_dis(self):
        total_input_state = np.matrix(np.ones((2**(CFG.system_size * 2 + (1 if CFG.extra_ancilla else 0)), 1)))
        final_input_state = np.matrix(np.ones((2**(CFG.system_size * 2 + (1 if CFG.extra_ancilla  and CFG.ancilla_mode == "pass" else 0)), 1)))
        dis = Discriminator()
        gen = Generator(total_input_state)
        total_gen_state = gen.get_total_gen_state()
        final_gen_state = get_final_gen_state_for_discriminator(total_gen_state)
        final_target_state = get_final_target_state(final_input_state)
        
        old_alpha = dis.alpha.copy()
        old_beta = dis.beta.copy()
        
        dis.update_dis(final_target_state, final_gen_state)
        
        # Check that it updates the alpha and beta parameters
        assert dis.alpha.shape == (dis.size, 4)
        assert dis.beta.shape == (dis.size, 4)
        assert np.any(dis.alpha != old_alpha)
        assert np.any(dis.beta != old_beta)