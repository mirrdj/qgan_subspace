# Copyright 2025 GIQ, Universitat Autònoma de Barcelona
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

from circuit.discriminator import Discriminator
from circuit.generator import Generator
from circuit.initial_state import get_choi_state, get_zero_state  # Import get_choi_state
from circuit.target_hamiltonian import construct_target  # Corrected import
from config import Config
from tools.data_managers import (
    save_fidelity_loss,
    save_theta,
    train_log,
)
from tools.loading_helpers import load_models_if_specified
from tools.plot_hub import plt_fidelity_vs_iter
from training.fidelity import compute_fidelity  # Reverted to correct import path

np.random.seed()  # Seed for initial parameter generation in Generator/Discriminator


class Training:
    def __init__(self, config_module: Config):
        """Builds the configuration for the Training. You might wanna comment/discomment lines, for changing the model."""
        self.cf: Config = config_module

        # Determine effective qubit numbers based on config
        self.N_main = self.cf.num_qubits
        self.M_main = self.cf.num_qubits  # Input subspace size before ancilla consideration

        if self.cf.extra_ancilla:
            self.N_eff = self.N_main + 1  # N_eff: qubits for U_G and target state
            self.M_eff = self.M_main + 1  # M_eff: qubits for the generator's input subspace state |psi_M>
        else:
            self.N_eff = self.N_main
            self.M_eff = self.M_main

        ################################################################################################################
        # TODO: Change above logic, by a more readable and correct one:
        ################################################################################################################
        # - in_target_q = num_qubits
        # - total_target_q = (num_qubits * 2 if choi else num_qubits) (+ 1 if extra_ancilla in "pass" mode)

        # - in_gen_q = num_qubits (+ 1 if extra_ancilla)
        # - total_gen_q = (num_qubits * 2 if choi else num_qubits) (+ 1 if extra_ancilla)
        # - total_gen_q_after_get_rid_of_ancilla = (num_qubits * 2 if choi else num_qubits) [only applicable if extra_ancilla in "project" or "trace_out" mode]

        # - in_disc_q = (num_qubits * 2 if choi else num_qubits) (+ 1 if extra_ancilla in "pass" mode)
        ################################################################################################################

        # self.input_state is |psi_M>, the input to the generator's subspace logic.
        # It should be a state vector of M_eff qubits.
        if self.cf.initial_state == "zero":
            self.input_state = get_zero_state(self.M_eff)
        elif self.cf.initial_state == "choi":
            train_log(
                f"Using Choi state as initial state for the generator with {self.M_eff} qubits.\n",
                self.cf.log_path,
            )
            self.input_state = get_choi_state(self.M_eff)
            self.in_state = self.input_state[: len(self.input_state) // 2 - 1]
            self.out_state = self.input_state[len(self.input_state) // 2 :]
        else:
            raise ValueError(
                f"Unknown initial state type: {self.cf.initial_state}. Supported types are 'zero' and 'choi'."
            )

        self.input_state = pnp.array(self.input_state, dtype=complex, requires_grad=False)

        # Target Unitary (acts on N_eff qubits) or state vector
        train_log(
            f"Constructing target unitary for type '{self.cf.target_hamiltonian}' on {self.N_eff} qubits.\n",
            self.cf.log_path,
        )
        # Corrected call: pass N_eff first, then config object
        self.target_unitary = construct_target(self.N_eff, self.cf)
        self.target_unitary = pnp.array(self.target_unitary, dtype=complex, requires_grad=False)

        # Real state: U_target |input_state>
        # self.target_unitary should be already constructed based on target_hamiltonian

        # If zero, make the real state the target unitary applied to the input state:
        if self.cf.initial_state == "zero":
            self.real_state = self.target_unitary @ self.input_state

        # If choi, make only half pass to target, and add the rest later:
        elif self.cf.initial_state == "choi":
            self.real_state = self.target_unitary @ self.in_state
            self.real_state = pnp.concatenate(self.real_state, self.out_state)

        self.real_state = pnp.array(self.real_state, dtype=complex, requires_grad=False)

        if self.cf.extra_ancilla:
            train_log(
                "Using an ancilla qubit in the generator and discriminator.\n",
                self.cf.log_path,
            )
            self.in_state = pnp.concatenate((self.in_state, pnp.zeros((1,), dtype=complex)), axis=0)

        # Generator
        self.gen = Generator(
            # input_state=self.in_state, #TODO: Solve which state/qubits is passed to the generator + internal logic
            num_qubits_N=self.N_eff,
            num_qubits_M=self.M_eff,
            layer=self.cf.gen_layers,
            ansatz_type=self.cf.ansatz_gen,  # Pass ansatz type string
            learning_rate=self.cf.learning_rate,  # Pass unified learning rate
        )

        # # TODO: Implement ancilla mode logic:
        # if self.cf.ancilla_mode == "pass":
        #     disc_state = pnp.concatenate(self.in_state, self.out_state)
        # # elif self.cf.ancilla_mode == "project":
        # #     disc_state = pnp.concatenate(project_ancila(self.in_state), self.out_state)
        # # elif self.cf.ancilla_mode == "trace_out":
        # #     disc_state = pnp.concatenate(self.in_state[:-1], self.out_state)
        # else:
        #     raise ValueError(
        #         f"Unknown ancilla mode: {self.cf.ancilla_mode}. Supported modes are 'pass', 'project', and 'trace_out'."
        #     )

        # Discriminator (acts on N_eff qubits)
        self.discriminator_total_qubits = self.N_eff
        self.dis = Discriminator(
            # input_state=disc_state, #TODO: Solve which state/qubits is passed to the discriminator + internal logic
            num_qubits=self.N_eff,
            num_disc_layers=self.cf.disc_layers,
            ansatz_type=self.cf.ansatz_disc,  # Pass ansatz type string
            learning_rate=self.cf.learning_rate,  # Pass unified learning rate
        )

        # Load models if specified (only the params) # TODO: Make this compatible with adding ancilla & choi later
        load_models_if_specified(self, self.cf)

    def run(self):
        """Run the training, saving the data, the model, the logs, and the results plots."""

        f = compute_fidelity(self.gen, self.input_state, self.real_state)

        fidelities_history, losses_G_history, losses_D_history, losses_D_minus_G_history = [], [], [], []
        starttime = datetime.now()
        num_epochs = 0

        initial_fidelity_log = f"Initial Fidelity: {f:.6f}\n"
        train_log(initial_fidelity_log, self.cf.log_path)

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
                train_log(loop_header, self.cf.log_path)

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
                train_log(iter_log_msg, self.cf.log_path)

                if (iter_idx + 1) % 10 == 0:
                    endtime = datetime.now()
                    training_duration_hours = (endtime - starttime).total_seconds() / 3600.0
                    log_param_str = (
                        f"Epoch: {num_epochs:4d}, Iter: {iter_idx + 1:4d} | "
                        f"Fidelity: {current_fidelity:.6f} | G_Loss: {cost_G:.6f} | D_Loss: {cost_D:.6f} | D-G Loss: {cost_D_minus_G:.6f} | "
                        f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} | "
                        f"Duration (hrs): {training_duration_hours:.2f}\n"
                    )
                    train_log(log_param_str, self.cf.log_path)

            f = fidelities_epoch[-1]
            fidelities_history.extend(fidelities_epoch.tolist())
            losses_G_history.extend(losses_G_epoch.tolist())
            losses_D_history.extend(losses_D_epoch.tolist())
            losses_D_minus_G_history.extend(losses_D_minus_G_epoch.tolist())

            epoch_duration = time.time() - epoch_start_time
            epoch_log_msg = f"Epoch {num_epochs} completed. Fidelity: {f:.6f}. Duration: {epoch_duration:.2f}s\n"
            train_log(epoch_log_msg, self.cf.log_path)

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
                train_log(max_epoch_log, self.cf.log_path)
                break

        training_finished_log = "Training finished.\n"
        train_log(training_finished_log, self.cf.log_path)

        plt_fidelity_vs_iter(
            np.array(fidelities_history),
            np.array(losses_G_history),
            np.array(losses_D_history),
            np.array(losses_D_minus_G_history),
            self.cf,
            num_epochs,
        )

        save_fidelity_loss(np.array(fidelities_history), np.array(losses_G_history), self.cf.fid_loss_path)

        self.gen.save_model(self.cf.model_gen_path)
        self.dis.save_model(self.cf.model_dis_path)

        save_theta(self.gen.params_gen, self.cftheta_path)

        endtime = datetime.now()
        total_training_time_seconds = (endtime - starttime).total_seconds()
        total_time_log = f"Total training time: {total_training_time_seconds:.2f} seconds\\nEnd of training script.\\n"
        train_log(total_time_log, self.cf.log_path)
