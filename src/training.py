# Copyright 2024 PennyLane Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from datetime import datetime

import numpy as np  # For seeding and initial data arrays
import pennylane.numpy as pnp  # For PennyLane operations

from config import CFG, Config
from discriminator.discriminator import Discriminator
from fidelity.fidelity import compute_fidelity
from generator.generator import Generator
from runner.loading_helpers import load_models_if_specified
from runner.training_runner import run_single_training, run_test_configurations
from target.target_hamiltonian import construct_target  # Corrected import
from target.target_state import get_ghz_state, get_zero_state  # Import get_ghz_state
from tools.data_managers import (
    save_fidelity_loss,
    save_theta,
    train_log,
)
from tools.plot_hub import plt_fidelity_vs_iter

np.random.seed()  # Seed for initial parameter generation in Generator/Discriminator


class Training:
    def __init__(self, config_module: Config, train_log_fn, load_timestamp=None):
        """Builds the configuration for the Training. You might wanna comment/discomment lines, for changing the model."""
        self.cf: Config = config_module
        self.train_log = train_log_fn

        # Determine effective qubit numbers based on config
        self.N_main = self.cf.system_size
        self.M_main = self.cf.system_size  # Input subspace size before ancilla consideration

        if self.cf.extra_ancilla:
            self.N_eff = self.N_main + 1  # N_eff: qubits for U_G and target state
            self.M_eff = self.M_main + 1  # M_eff: qubits for the generator's input subspace state |psi_M>
        else:
            self.N_eff = self.N_main
            self.M_eff = self.M_main

        # self.input_state is |psi_M>, the input to the generator's subspace logic.
        # It should be a state vector of M_eff qubits.
        if self.cf.generator_initial_state_type == "zero":
            self.input_state = get_zero_state(self.M_eff)
        elif self.cf.generator_initial_state_type == "maximally_entangled":
            self.train_log(
                f"Using GHZ state as initial state for the generator with {self.M_eff} qubits.\n",
                self.cf.get_log_path(),
            )
            self.input_state = get_ghz_state(self.M_eff)
        else:
            # Default to zero state if type is unknown
            self.train_log(
                f"Warning: Unknown generator_initial_state_type '{self.cf.generator_initial_state_type}'. Defaulting to zero state for {self.M_eff} qubits.\n",
                self.cf.get_log_path(),
            )
            self.input_state = get_zero_state(self.M_eff)

        self.input_state = pnp.array(self.input_state, dtype=complex, requires_grad=False)

        # Target Unitary (acts on N_eff qubits) or state vector
        if self.cf.target_type == "custom_state_vector":
            self.target_unitary = None  # Target is a state vector, not a unitary
            self.train_log(f"Using custom target state vector for {self.N_eff} qubits.\n", self.cf.get_log_path())
        else:
            self.train_log(
                f"Constructing target unitary for type '{self.cf.target_type}' on {self.N_eff} qubits.\n",
                self.cf.get_log_path(),
            )
            # Corrected call: pass N_eff first, then config object, then train_log
            self.target_unitary = construct_target(self.N_eff, self.cf, self.train_log)
            self.target_unitary = pnp.array(self.target_unitary, dtype=complex, requires_grad=False)

        if self.cf.extra_ancilla:
            self.N_tot = 2 * self.cf.system_size
        else:
            self.N_tot = self.cf.system_size

        # Generator
        self.gen = Generator(
            system_size_N=self.N_eff,
            system_size_M=self.M_eff,
            layer=self.cf.generator_layers,
            ansatz_type=self.cf.ansatz_gen_type,  # Pass ansatz type string
            learning_rate=self.cf.learning_rate,  # Pass unified learning rate
        )

        # Real state: U_target |reference_state_N>
        self.real_state = self.initialize_target_state()
        self.real_state = pnp.array(self.real_state, dtype=complex, requires_grad=False)

        # Discriminator (acts on N_eff qubits)
        self.discriminator_total_qubits = self.N_eff
        self.dis = Discriminator(
            system_size=self.N_eff,
            num_disc_layers=self.cf.discriminator_layers,
            ansatz_type=self.cf.ansatz_disc_type,  # Pass ansatz type string
            learning_rate=self.cf.learning_rate,  # Pass unified learning rate
        )

        load_models_if_specified(self, load_timestamp, self.cf, self.train_log)

    def initialize_target_state(self) -> pnp.ndarray:
        """Initialize the target state: U_target |reference_state_N_eff> or custom state vector."""
        if self.cf.target_type == "custom_state_vector":
            if self.cf.custom_target_state_vector is not None:
                # Ensure the custom state vector matches N_eff
                expected_dim = 2**self.N_eff
                if self.cf.custom_target_state_vector.shape != (expected_dim,):
                    error_msg = f"Custom target state vector shape {self.cf.custom_target_state_vector.shape} does not match expected dimension ({expected_dim},) for {self.N_eff} qubits."
                    self.train_log(error_msg + "\n", self.cf.get_log_path())
                    raise ValueError(error_msg)
                # Normalize the custom state vector
                norm = pnp.linalg.norm(self.cf.custom_target_state_vector)
                if not pnp.isclose(norm, 1.0):
                    warn_msg = f"Warning: Custom target state vector norm is {norm}. Normalizing."
                    self.train_log(warn_msg + "\n", self.cf.get_log_path())
                    state_vec = self.cf.custom_target_state_vector / norm
                else:
                    state_vec = self.cf.custom_target_state_vector
                return pnp.array(state_vec, dtype=complex, requires_grad=False)
            error_msg = "Target type is custom_state_vector, but no vector is provided in config."
            self.train_log(error_msg + "\n", self.cf.get_log_path())
            raise ValueError(error_msg)

        # For other target types, assume U_target @ |0...0>
        # Reference state is |0...0> on N_eff qubits
        reference_state_N_eff = get_zero_state(self.N_eff)
        reference_state_N_eff = pnp.array(reference_state_N_eff, requires_grad=False)  # Ensure it's an array

        # self.target_unitary should be already constructed based on target_type
        real_state_vector = self.target_unitary @ reference_state_N_eff
        return real_state_vector

    def run(self):
        """Run the training, saving the data, the model, the logs, and the results plots."""

        f = compute_fidelity(self.gen, self.input_state, self.real_state)

        fidelities_history, losses_G_history, losses_D_history, losses_D_minus_G_history = [], [], [], []
        starttime = datetime.now()
        num_epochs = 0

        initial_fidelity_log = f"Initial Fidelity: {f:.6f}\n"
        self.train_log(initial_fidelity_log, self.cf.get_log_path())

        while f < 0.99:
            fidelities_epoch = np.zeros(self.cf.iterations_epoch)
            losses_G_epoch = np.zeros(self.cf.iterations_epoch)
            losses_D_epoch = np.zeros(self.cf.iterations_epoch)
            losses_D_minus_G_epoch = np.zeros(self.cf.iterations_epoch)
            num_epochs += 1

            epoch_start_time = time.time()

            for iter_idx in range(self.cf.iterations_epoch):
                iter_start_time = time.time()
                loop_header = f"==================================================\nEpoch {num_epochs}, Iteration {iter_idx + 1}, Learning_Rate {self.cf.learning_rate}\n"
                self.train_log(loop_header, self.cf.get_log_path())

                self.gen.update_gen(self.dis, self.input_state)
                self.dis.update_dis(self.gen, self.real_state, self.input_state)

                current_fidelity = compute_fidelity(self.gen, self.input_state, self.real_state)

                cost_G = self.gen.generator_cost_function(self.gen.params_gen, self.dis, self.input_state)

                fake_state_for_D_cost = self.gen.get_generated_state_vector(self.gen.params_gen, self.input_state)
                cost_D = self.dis.discriminator_cost_function(
                    self.dis.params_disc, fake_state_for_D_cost, self.real_state
                )

                cost_D_minus_G = cost_D - cost_G

                fidelities_epoch[iter_idx] = current_fidelity
                losses_G_epoch[iter_idx] = cost_G.numpy() if hasattr(cost_G, "numpy") else cost_G
                losses_D_epoch[iter_idx] = cost_D.numpy() if hasattr(cost_D, "numpy") else cost_D
                losses_D_minus_G_epoch[iter_idx] = (
                    cost_D_minus_G.numpy() if hasattr(cost_D_minus_G, "numpy") else cost_D_minus_G
                )

                iter_time = time.time() - iter_start_time
                iter_log_msg = f"Fidelity: {current_fidelity:.6f}, G_Loss: {cost_G:.6f}, D_Loss: {cost_D:.6f}, D-G Loss: {cost_D_minus_G:.6f}, Iter time: {iter_time:.2f}s\n==================================================\n"
                self.train_log(iter_log_msg, self.cf.get_log_path())

                if (iter_idx + 1) % 10 == 0:
                    endtime = datetime.now()
                    training_duration_hours = (endtime - starttime).total_seconds() / 3600.0
                    log_param_str = (
                        f"Epoch: {num_epochs:4d}, Iter: {iter_idx + 1:4d} | "
                        f"Fidelity: {current_fidelity:.6f} | G_Loss: {cost_G:.6f} | D_Loss: {cost_D:.6f} | D-G Loss: {cost_D_minus_G:.6f} | "
                        f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} | "
                        f"Duration (hrs): {training_duration_hours:.2f}\n"
                    )
                    self.train_log(log_param_str, self.cf.get_log_path())

            f = fidelities_epoch[-1]
            fidelities_history.extend(fidelities_epoch.tolist())
            losses_G_history.extend(losses_G_epoch.tolist())
            losses_D_history.extend(losses_D_epoch.tolist())
            losses_D_minus_G_history.extend(losses_D_minus_G_epoch.tolist())

            epoch_duration = time.time() - epoch_start_time
            epoch_log_msg = f"Epoch {num_epochs} completed. Fidelity: {f:.6f}. Duration: {epoch_duration:.2f}s\n"
            self.train_log(epoch_log_msg, self.cf.get_log_path())

            plot_every = getattr(self.cf, "plot_every_epochs", 1)
            if num_epochs % plot_every == 0:
                plt_fidelity_vs_iter(
                    np.array(fidelities_history),
                    np.array(losses_G_history),
                    np.array(losses_D_history),
                    np.array(losses_D_minus_G_history),
                    self.cf,
                    num_epochs,
                )

            if num_epochs >= self.cf.epochs:
                max_epoch_log = f"Maximum number of epochs ({self.cf.epochs}) reached.\n"
                self.train_log(max_epoch_log, self.cf.get_log_path())
                break

        training_finished_log = "Training finished.\n"
        self.train_log(training_finished_log, self.cf.get_log_path())

        plt_fidelity_vs_iter(
            np.array(fidelities_history),
            np.array(losses_G_history),
            np.array(losses_D_history),
            np.array(losses_D_minus_G_history),
            self.cf,
            num_epochs,
        )

        save_fidelity_loss(np.array(fidelities_history), np.array(losses_G_history), self.cf.get_fid_loss_path())

        self.gen.save_model(self.cf.get_model_gen_path())
        self.dis.save_model(self.cf.get_model_dis_path())

        save_theta(self.gen.params_gen, self.cf.get_theta_path())

        endtime = datetime.now()
        total_training_time_seconds = (endtime - starttime).total_seconds()
        total_time_log = f"Total training time: {total_training_time_seconds:.2f} seconds\nEnd of training script.\n"
        self.train_log(total_time_log, self.cf.get_log_path())


if __name__ == "__main__":
    testing = False  # Set to True to run test configurations

    if not hasattr(CFG, "plot_every_epochs"):
        CFG.plot_every_epochs = 1

    load_timestamp_from_config = CFG.load_ts_for_training

    if testing:
        test_configurations = [
            {
                "system_size": 2,
                "generator_layers": 1,
                "extra_ancilla": False,
                "iterations_epoch": 3,
                "epochs": 1,
                "discriminator_layers": 1,
                "label_suffix": "c1_2q_1l_noanc_d1_ghz_XXYYZZZ_zero",
                "target_type": "ghz_state",
                "ansatz_gen_type": "XX_YY_ZZ_Z",
                "ansatz_disc_type": "XX_YY_ZZ_Z",
                "generator_initial_state_type": "zero",
                "cost_function_type": "original_qgan",
            },
            {
                "system_size": 2,
                "generator_layers": 1,
                "extra_ancilla": True,
                "iterations_epoch": 3,
                "epochs": 1,
                "discriminator_layers": 1,
                "label_suffix": "c2_2q_1l_anc_d1_ghz_ZZXZ_entangled",
                "target_type": "ghz_state",
                "ansatz_gen_type": "ZZ_X_Z",
                "ansatz_disc_type": "ZZ_X_Z",
                "generator_initial_state_type": "maximally_entangled",
                "cost_function_type": "original_qgan",
            },
            {
                "system_size": 2,
                "generator_layers": 2,
                "extra_ancilla": False,
                "iterations_epoch": 3,
                "epochs": 1,
                "discriminator_layers": 2,
                "label_suffix": "c3_2q_2l_noanc_d2_cluster_XXYYZZZ_zero",
                "target_type": "cluster_state_1d",
                "ansatz_gen_type": "XX_YY_ZZ_Z",
                "ansatz_disc_type": "XX_YY_ZZ_Z",
                "generator_initial_state_type": "zero",
                "cost_function_type": "original_qgan",
            },
            {
                "system_size": 3,
                "generator_layers": 1,
                "extra_ancilla": False,
                "iterations_epoch": 3,
                "epochs": 1,
                "discriminator_layers": 1,
                "label_suffix": "c4_3q_1l_noanc_d1_tfim_ZZXZ_entangled",
                "target_type": "tfim_ground_state",
                "tfim_h_param": 0.5,  # Example param for tfim
                "ansatz_gen_type": "ZZ_X_Z",
                "ansatz_disc_type": "ZZ_X_Z",
                "generator_initial_state_type": "maximally_entangled",
                "cost_function_type": "original_qgan",
            },
            # Add more configurations as needed
            {
                "system_size": 2,
                "generator_layers": 1,
                "extra_ancilla": False,
                "iterations_epoch": 3,
                "epochs": 1,
                "discriminator_layers": 1,
                "label_suffix": "c5_2q_1l_noanc_d1_cluster_ZZXZ_zero",
                "target_type": "cluster_state_1d",
                "ansatz_gen_type": "ZZ_X_Z",
                "ansatz_disc_type": "ZZ_X_Z",
                "generator_initial_state_type": "zero",
                "cost_function_type": "original_qgan",
            },
        ]
        run_test_configurations(CFG, train_log, Training, test_configurations)
    else:
        run_single_training(CFG, train_log, Training, load_timestamp=load_timestamp_from_config)
