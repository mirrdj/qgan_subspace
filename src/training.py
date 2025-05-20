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
import traceback
from datetime import datetime

import numpy as np  # For seeding and initial data arrays
import pennylane.numpy as pnp  # For PennyLane operations

import config as cf
from cost_functions.cost_and_fidelity import compute_fidelity
from discriminator.discriminator import Discriminator
from generator import ansatz  # Import the whole module
from generator.generator import Generator
from runner.loading_helpers import load_models_if_specified
from runner.training_runner import run_single_training, run_test_configurations
from target.target_hamiltonian import construct_target
from target.target_state import get_zero_state
from tools.data_managers import (
    save_fidelity_loss,
    save_theta,
    train_log,
)
from tools.plot_hub import plt_fidelity_vs_iter

np.random.seed()  # Seed for initial parameter generation in Generator/Discriminator


class Training:
    def __init__(self, config_module, train_log_fn, load_timestamp=None):
        """Builds the configuration for the Training. You might wanna comment/discomment lines, for changing the model."""
        self.cf = config_module
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
        # It should be a state vector of M_eff qubits. Using |0...0>_M_eff.
        self.input_state = get_zero_state(self.M_eff)

        # Ensure input_state is a pnp.array
        self.input_state = pnp.array(self.input_state, dtype=complex, requires_grad=False)

        # Target Unitary (acts on N_eff qubits)
        self.target_unitary = construct_target(self.N_eff, ZZZ_terms=True)  # Corrected ZZZ to ZZZ_terms
        self.target_unitary = pnp.array(self.target_unitary, dtype=complex, requires_grad=False)

        if self.cf.extra_ancilla:
            self.N_tot = 2 * self.cf.system_size  # Total qubits including ancilla for input state
        else:
            self.N_tot = self.cf.system_size

        # Select ansatz function based on config or default
        # For now, we assume a default or a simple selection mechanism
        # The call to the ansatz function itself is done within the Generator's QNode
        self.ansatz_fn = ansatz.construct_qcircuit_XX_YY_ZZ_Z
        self.params_shape_fn = ansatz.get_params_shape_XX_YY_ZZ_Z

        # Generator
        ansatz_circuit_fn = ansatz.construct_qcircuit_XX_YY_ZZ_Z  # Pass the function itself
        self.gen = Generator(
            system_size_N=self.N_eff,  # Pass N_eff as system_size_N
            system_size_M=self.M_eff,  # Pass M_eff as system_size_M
            layer=self.cf.layer,
            ansatz_fn=ansatz_circuit_fn,
            params_shape_fn=self.params_shape_fn,  # Pass the shape function
        )

        # Real state: U_target |reference_state_N>
        self.real_state = self.initialize_target_state()
        self.real_state = pnp.array(self.real_state, dtype=complex, requires_grad=False)

        # Discriminator (acts on N_eff qubits)
        self.discriminator_total_qubits = self.N_eff
        self.dis = Discriminator(system_size=self.N_eff, num_disc_layers=self.cf.num_discriminator_layers)

        load_models_if_specified(self, load_timestamp, self.cf, self.train_log)

    def initialize_target_state(self) -> pnp.ndarray:
        """Initialize the target state: U_target |reference_state_N_eff>."""
        # Reference state is |0...0> on N_eff qubits
        reference_state_N_eff = get_zero_state(self.N_eff)
        reference_state_N_eff = pnp.array(reference_state_N_eff, requires_grad=False)  # Ensure it's an array

        real_state_vector = self.target_unitary @ reference_state_N_eff
        return real_state_vector

    def run(self):
        """Run the training, saving the data, the model, the logs, and the results plots."""

        # Compute fidelity at initial
        f = compute_fidelity(self.gen, self.input_state, self.real_state)

        # Data storing
        fidelities_history, losses_G_history, losses_D_history, losses_D_minus_G_history = [], [], [], []
        starttime = datetime.now()
        num_epochs = 0

        initial_fidelity_log = f"Initial Fidelity: {f:.6f}\n"
        self.train_log(initial_fidelity_log, self.cf.log_path)

        # Training
        while f < 0.99:  # Target fidelity
            fidelities_epoch = np.zeros(self.cf.iterations_epoch)  # For current epoch
            losses_G_epoch = np.zeros(self.cf.iterations_epoch)  # For current epoch G_loss
            losses_D_epoch = np.zeros(self.cf.iterations_epoch)  # For current epoch D_loss
            losses_D_minus_G_epoch = np.zeros(self.cf.iterations_epoch)  # For D_loss - G_loss
            num_epochs += 1

            epoch_start_time = time.time()

            for iter_idx in range(self.cf.iterations_epoch):
                iter_start_time = time.time()
                loop_header = f"==================================================\nEpoch {num_epochs}, Iteration {iter_idx + 1}, Step_size {self.cf.eta}\n"
                self.train_log(loop_header, self.cf.log_path)

                # Generator gradient descent
                self.gen.update_gen(self.dis, self.input_state)

                # Discriminator gradient ascent
                self.dis.update_dis(self.gen, self.real_state, self.input_state)

                current_fidelity = compute_fidelity(self.gen, self.input_state, self.real_state)

                cost_G = self.gen.generator_cost_function(self.gen.params_gen, self.dis, self.input_state)

                # Generate fake state for Discriminator cost calculation
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
                self.train_log(iter_log_msg, self.cf.log_path)

                if (iter_idx + 1) % 10 == 0:  # Log every 10 iterations
                    endtime = datetime.now()
                    training_duration_hours = (endtime - starttime).total_seconds() / 3600.0
                    log_param_str = (
                        f"Epoch: {num_epochs:4d}, Iter: {iter_idx + 1:4d} | "
                        f"Fidelity: {current_fidelity:.6f} | G_Loss: {cost_G:.6f} | D_Loss: {cost_D:.6f} | D-G Loss: {cost_D_minus_G:.6f} | "
                        f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} | "
                        f"Duration (hrs): {training_duration_hours:.2f}\n"
                    )
                    self.train_log(log_param_str, self.cf.log_path)

            f = fidelities_epoch[-1]
            fidelities_history.extend(fidelities_epoch.tolist())  # Append current epoch's data
            losses_G_history.extend(losses_G_epoch.tolist())
            losses_D_history.extend(losses_D_epoch.tolist())
            losses_D_minus_G_history.extend(losses_D_minus_G_epoch.tolist())

            epoch_duration = time.time() - epoch_start_time
            epoch_log_msg = f"Epoch {num_epochs} completed. Fidelity: {f:.6f}. Duration: {epoch_duration:.2f}s\n"
            self.train_log(epoch_log_msg, self.cf.log_path)

            if num_epochs % self.cf.plot_every_epochs == 0:  # Add cf.plot_every_epochs to config.py (e.g., 1 or 5)
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
                self.train_log(max_epoch_log, self.cf.log_path)
                break

        training_finished_log = "Training finished.\n"
        self.train_log(training_finished_log, self.cf.log_path)

        # Final plot
        plt_fidelity_vs_iter(
            np.array(fidelities_history),
            np.array(losses_G_history),
            np.array(losses_D_history),
            np.array(losses_D_minus_G_history),
            self.cf,
            num_epochs,
        )

        # Save data of fidelity and loss
        save_fidelity_loss(np.array(fidelities_history), np.array(losses_G_history), self.cf.fid_loss_path)

        # Save data of the generator and the discriminator using their own save methods
        self.gen.save_model(self.cf.model_gen_path)
        self.dis.save_model(self.cf.model_dis_path)

        # Output the parameters of the generator
        save_theta(self.gen.params_gen, self.cf.theta_path)

        endtime = datetime.now()
        total_training_time_seconds = (endtime - starttime).total_seconds()
        total_time_log = f"Total training time: {total_training_time_seconds:.2f} seconds\nEnd of training script.\n"
        self.train_log(total_time_log, self.cf.log_path)


if __name__ == "__main__":
    testing = False  # Set this flag to True to run test configurations, False for default run

    # Ensure plot_every_epochs is set, default to 1 if not in config
    if not hasattr(cf, "plot_every_epochs"):
        cf.plot_every_epochs = 1

    # Ensure load_ts_for_training is available, default to None if not in config
    # This will be used by run_single_training if testing is False
    load_timestamp_from_config = getattr(cf, "load_ts_for_training", None)

    if testing:
        # Define test configurations directly here or load from another file if they become too large
        test_configurations = [
            {
                "system_size": 2,
                "layer": 1,
                "extra_ancilla": False,
                "iterations_epoch": 3,
                "epochs": 1,
                "num_discriminator_layers": 1,
                "label_suffix": "c1_2q_1l_noanc_d1",
            },
            {
                "system_size": 2,
                "layer": 1,
                "extra_ancilla": True,
                "iterations_epoch": 3,
                "epochs": 1,
                "num_discriminator_layers": 1,
                "label_suffix": "c2_2q_1l_anc_d1",
            },
            {
                "system_size": 2,
                "layer": 2,
                "extra_ancilla": False,
                "iterations_epoch": 3,
                "epochs": 1,
                "num_discriminator_layers": 2,
                "label_suffix": "c3_2q_2l_noanc_d2",
            },
            {
                "system_size": 3,
                "layer": 1,
                "extra_ancilla": False,
                "iterations_epoch": 3,
                "epochs": 1,
                "num_discriminator_layers": 1,
                "label_suffix": "c4_3q_1l_noanc_d1",
            },
        ]
        run_test_configurations(cf, train_log, Training, test_configurations)
    else:
        run_single_training(cf, train_log, Training, load_timestamp=load_timestamp_from_config)
