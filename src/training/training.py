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
import pennylane as qml  # Added import for qml
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

# Removed: from training.fidelity import compute_fidelity


# np.random.seed() # This was already commented or removed, ensure it is.


class Training:
    def __init__(self, config_module: Config):
        """Builds the configuration for the Training, defining qubit numbers,
        initial states, target unitary, and instantiating G and D."""
        self.cf: Config = config_module

        self._initialize_qubit_counts()
        self._prepare_generator_input_state()
        self._prepare_real_target_state()  # This will now store self.main_part_real_state

        # --- Instantiate Generator and Discriminator ---
        train_log(
            f"Instantiating Generator: Operates on num_qubits={self.num_qubits_for_generator_class}. "
            f"Input state for G has {self.num_qubits_generator_input_state_actual} qubits. "  # Should match num_qubits_for_generator_class
            f"Layers: {self.cf.gen_layers}, Ansatz: {self.cf.ansatz_gen}.\\\\n",
            self.cf.log_path,
        )
        if self.num_qubits_for_generator_class != self.num_qubits_generator_input_state_actual:
            # This should ideally not happen with the current logic in _initialize_qubit_counts
            raise ValueError(
                f"Mismatch: Generator class qubits ({self.num_qubits_for_generator_class}) "
                f"!= actual input state qubits ({self.num_qubits_generator_input_state_actual}). Check logic."
            )

        self.gen = Generator(
            num_qubits=self.num_qubits_for_generator_class,  # G operates on this many qubits
            layer=self.cf.gen_layers,
            ansatz_type=self.cf.ansatz_gen,
            learning_rate=self.cf.learning_rate,
        )

        train_log(
            f"Instantiating Discriminator: Input num_qubits_discriminator={self.num_qubits_discriminator} qubits. "
            f"Layers: {self.cf.disc_layers}, Ansatz: {self.cf.ansatz_disc}.\\n",
            self.cf.log_path,
        )
        self.dis = Discriminator(
            num_qubits=self.num_qubits_discriminator,  # Input qubits for D
            num_disc_layers=self.cf.disc_layers,
            ansatz_type=self.cf.ansatz_disc,
            learning_rate=self.cf.learning_rate,
        )

        # Load models if specified (params only). TODO: Ensure compatibility with qubit changes.
        load_models_if_specified(self, self.cf)

    def _initialize_qubit_counts(self):
        """Calculates and logs effective qubit numbers for G and D operations."""
        N_tar = self.cf.num_qubits  # Qubits for the target unitary T
        # Ancilla effect: 1 if extra_ancilla is true and mode is 'pass', 0 otherwise.
        # This is the ancilla that is appended to D's input and T's output.
        has_ancilla_passthrough_effect = 1 if self.cf.extra_ancilla and self.cf.ancilla_mode == "pass" else 0
        # Ancilla effect for G's input: G always sees an ancilla if extra_ancilla is true.
        has_ancilla_for_G_input = 1 if self.cf.extra_ancilla else 0

        # Determine num_qubits_for_generator_class: single num_qubits for Generator class constructor
        if self.cf.initial_state == "choi":
            # G's ansatz acts on N_tar system qubits.
            # Ancilla (if any) is handled externally after G's output is Choi-fied.
            self.num_qubits_for_generator_class = N_tar
        else:  # "zero" mode
            # G's ansatz acts on N_tar system qubits + its own ancilla if present.
            self.num_qubits_for_generator_class = N_tar + has_ancilla_for_G_input

        # Determine num_qubits_generator_input_state_actual: Actual size of the input state vector for G.
        # This must match what the Generator class instance expects.
        self.num_qubits_generator_input_state_actual = self.num_qubits_for_generator_class

        # Determine num_qubits_discriminator: total qubits for Discriminator's input
        if self.cf.initial_state == "choi":
            # D sees (I @ T)|Phi+> (2*N_tar qubits) + passthrough ancilla
            qubits_disc_main_part = 2 * N_tar
        else:  # "zero" mode
            # D sees T|0> (N_tar qubits) + passthrough ancilla
            qubits_disc_main_part = N_tar
        self.num_qubits_discriminator = qubits_disc_main_part + has_ancilla_passthrough_effect

        train_log(
            f"Qubit configuration: Target system (N_tar): {N_tar}, "
            f"Generator class op_qubits: {self.num_qubits_for_generator_class}, "
            f"Generator input state actual_qubits: {self.num_qubits_generator_input_state_actual}, "
            f"Discriminator input qubits: {self.num_qubits_discriminator}.\\\\n"
            f"Initial state: '{self.cf.initial_state}'. Extra ancilla: {self.cf.extra_ancilla} "
            f"(mode: '{self.cf.ancilla_mode}', G_sees_ancilla: {bool(has_ancilla_for_G_input)}, D_passthrough_ancilla: {bool(has_ancilla_passthrough_effect)}).\\\\n",
            self.cf.log_path,
        )

    def _prepare_generator_input_state(self):
        """Prepares self.input_state for the Generator, matching self.num_qubits_generator_input_state_actual."""
        N_tar = self.cf.num_qubits
        has_ancilla_for_G_input = 1 if self.cf.extra_ancilla else 0

        if self.cf.initial_state == "choi":
            # G's ansatz acts on N_tar qubits, input is |0...0>_{N_tar}.
            # self.num_qubits_for_generator_class should be N_tar here.
            if self.num_qubits_generator_input_state_actual != N_tar:
                raise ValueError(
                    f"Mismatch for Choi: G input qubits {self.num_qubits_generator_input_state_actual} != N_tar {N_tar}"
                )
            self.input_state = get_zero_state(N_tar)
            train_log(
                f"Generator input: |0...0> state for {N_tar} system qubits (Choi mode). "
                f"Total G input state qubits: {self.num_qubits_generator_input_state_actual}.\\\\n",
                self.cf.log_path,
            )
        else:  # "zero" mode
            # G's ansatz acts on N_tar + G's ancilla. Input is |0...0>_{N_tar} (x) |0>_anc_G.
            # self.num_qubits_for_generator_class should be N_tar + has_ancilla_for_G_input.
            expected_qubits = N_tar + has_ancilla_for_G_input
            if self.num_qubits_generator_input_state_actual != expected_qubits:
                raise ValueError(
                    f"Mismatch for Zero mode: G input qubits {self.num_qubits_generator_input_state_actual} != expected {expected_qubits}"
                )

            base_input_for_G = get_zero_state(N_tar)
            if has_ancilla_for_G_input:
                ancilla_state_for_G = get_zero_state(1)
                self.input_state = pnp.kron(base_input_for_G, ancilla_state_for_G)
                train_log(
                    f"Generator input: |0...0>_{N_tar} (x) |0>_ancilla_G. "
                    f"Total G input state qubits: {self.num_qubits_generator_input_state_actual}.\\\\n",
                    self.cf.log_path,
                )
            else:
                self.input_state = base_input_for_G
                train_log(
                    f"Generator input: |0...0>_{N_tar}. "
                    f"Total G input state qubits: {self.num_qubits_generator_input_state_actual}.\\\\n",
                    self.cf.log_path,
                )

        self.input_state = pnp.array(self.input_state, dtype=complex, requires_grad=False)

    def _prepare_real_target_state(self):
        """Prepares self.target_unitary and self.real_state for the Discriminator,
        and stores self.main_part_real_state for fidelity calculations."""
        N_tar = self.cf.num_qubits
        has_ancilla_passthrough_effect = 1 if self.cf.extra_ancilla and self.cf.ancilla_mode == "pass" else 0

        # --- Target Unitary (acts on N_tar qubits) ---
        train_log(
            f"Constructing target unitary '{self.cf.target_hamiltonian}' acting on {N_tar} system qubits (N_tar).\\n",
            self.cf.log_path,
        )
        self.target_unitary = construct_target(N_tar, self.cf)  # construct_target should use N_tar
        self.target_unitary = pnp.array(self.target_unitary, dtype=complex, requires_grad=False)

        # --- Construct self.main_part_real_state (core state for comparison) ---
        if self.cf.initial_state == "choi":
            phi_plus_N_tar = get_choi_state(N_tar)
            identity_N_tar = pnp.eye(2**N_tar, dtype=complex)
            operator_on_choi_halves = pnp.kron(identity_N_tar, self.target_unitary)
            self.main_part_real_state = operator_on_choi_halves @ phi_plus_N_tar
            train_log(
                f"Real state (main part for Choi): (I @ T) applied to Choi state for {N_tar} system qubits. "
                f"This part has {2 * N_tar} qubits.\\n",
                self.cf.log_path,
            )
        else:  # "zero" mode
            zero_state_N_tar = get_zero_state(N_tar)
            self.main_part_real_state = self.target_unitary @ zero_state_N_tar
            train_log(
                f"Real state (main part for Zero mode): Target unitary applied to |0...0> for {N_tar} system qubits. "
                f"This part has {N_tar} qubits.\\n",
                self.cf.log_path,
            )
        self.main_part_real_state = pnp.array(self.main_part_real_state, dtype=complex, requires_grad=False)

        # --- Construct self.real_state (actual input for Discriminator D) ---
        if has_ancilla_passthrough_effect:
            ancilla_0_state_passthrough = get_zero_state(1)
            self.real_state = pnp.kron(self.main_part_real_state, ancilla_0_state_passthrough)
            train_log(
                f"Passthrough ancilla |0> tensored to real state for discriminator. "
                f"Total D input state qubits (self.real_state): {self.num_qubits_discriminator}.\\n",
                self.cf.log_path,
            )
        else:
            self.real_state = self.main_part_real_state
            if self.cf.extra_ancilla and self.cf.ancilla_mode != "pass":
                train_log(
                    f"Extra ancilla is True, but ancilla_mode ('{self.cf.ancilla_mode}') is not 'pass'. "
                    f"Passthrough ancilla not added to discriminator's real_state.\\n",
                    self.cf.log_path,
                )
            elif not self.cf.extra_ancilla:
                train_log(
                    f"No extra ancilla. Discriminator input qubits for real_state: {self.num_qubits_discriminator}.\\n",
                    self.cf.log_path,
                )

        self.real_state = pnp.array(self.real_state, dtype=complex, requires_grad=False)

    def _calculate_fake_state_for_D(self, generator_params):
        """Calculates the fake state vector for the discriminator based on generator_params."""
        N_tar = self.cf.num_qubits

        if self.cf.initial_state == "choi":
            # Assumes self.gen.get_ansatz_unitary takes params and returns the unitary matrix
            U_G = self.gen.get_ansatz_unitary(generator_params)
            phi_plus_N_tar = get_choi_state(N_tar)
            identity_N_tar = pnp.eye(2**N_tar, dtype=complex)
            operator_on_choi_halves = pnp.kron(identity_N_tar, U_G)
            main_part_fake_state = operator_on_choi_halves @ phi_plus_N_tar

            if self.cf.extra_ancilla and self.cf.ancilla_mode == "pass":
                ancilla_0_state = get_zero_state(1)
                fake_state = pnp.kron(main_part_fake_state, ancilla_0_state)
            else:
                # If ancilla_mode is not "pass", the passthrough ancilla is not added here.
                # Other ancilla effects (like trace out from main_part_fake_state if U_G acted on more qubits)
                # are not handled here yet for Choi state.
                fake_state = main_part_fake_state

        else:  # "zero" mode
            # Assumes self.gen.get_generated_state_vector takes params and input_state
            # self.input_state is already prepared for G (N_tar or N_tar + G's ancilla)
            generated_output_from_G = self.gen.get_generated_state_vector(generator_params, self.input_state)

            if self.cf.extra_ancilla:
                if self.cf.ancilla_mode == "pass":
                    # G's output (which includes G's ancilla) is passed directly to D.
                    # num_qubits_discriminator was set to N_tar + 1 in this case.
                    fake_state = generated_output_from_G
                else:  # e.g., "trace_out", "project"
                    # G produced N_tar+1 qubits. D expects N_tar (as has_ancilla_passthrough_effect was 0).
                    # This requires tracing out G's ancilla from generated_output_from_G.
                    # This is a critical TODO for other ancilla modes.
                    raise NotImplementedError(
                        f"ancilla_mode='{self.cf.ancilla_mode}' with extra_ancilla=True in 'zero' initial_state "
                        f"requires implemented partial trace for fake state generation. This is not yet supported."
                    )
            else:  # not self.cf.extra_ancilla
                # G produced N_tar qubits. D expects N_tar qubits.
                fake_state = generated_output_from_G

        return pnp.array(fake_state, dtype=complex, requires_grad=False)

    def _calculate_current_fidelity(self):
        """Calculates fidelity between the generated N_tar system state and the target N_tar system state."""
        N_tar = self.cf.num_qubits
        # Assumes self.gen.get_params() exists
        current_gen_params = self.gen.get_params()

        if self.cf.initial_state == "choi":
            # U_G is the unitary from the generator for the N_tar system
            # Assumes self.gen.get_ansatz_unitary takes params
            U_G = self.gen.get_ansatz_unitary(current_gen_params)
            phi_plus_N_tar = get_choi_state(N_tar)
            identity_N_tar = pnp.eye(2**N_tar, dtype=complex)
            operator_on_choi_halves_G = pnp.kron(identity_N_tar, U_G)

            generated_comparison_state_vec = operator_on_choi_halves_G @ phi_plus_N_tar
            # self.main_part_real_state is (I @ T)|Phi+>
            target_comparison_state_vec = self.main_part_real_state

            # Convert state vectors to density matrices
            # Ensure they are PennyLane tensors
            gcs_pnp = pnp.array(generated_comparison_state_vec, requires_grad=False)
            tcs_pnp = pnp.array(target_comparison_state_vec, requires_grad=False)

            dm_generated = pnp.outer(gcs_pnp, pnp.conj(gcs_pnp))
            dm_target = pnp.outer(tcs_pnp, pnp.conj(tcs_pnp))

            return qml.math.fidelity(dm_generated, dm_target)

        # "zero" mode (the 'else' was removed as it's unnecessary after a return)
        # G's raw output state vector (on N_tar or N_tar + G's ancilla)
        # Assumes self.gen.get_generated_state_vector takes params and input_state
        raw_generated_state = self.gen.get_generated_state_vector(current_gen_params, self.input_state)

        # target_comparison_state_vec_N_tar is T|0>_{N_tar} (stored in self.main_part_real_state)
        target_comparison_state_vec_N_tar = self.main_part_real_state
        # Convert target state to density matrix (it's always on N_tar qubits)
        tcs_N_tar_pnp = pnp.array(target_comparison_state_vec_N_tar, requires_grad=False)
        dm_target_N_tar = pnp.outer(tcs_N_tar_pnp, pnp.conj(tcs_N_tar_pnp))

        dm_generated_N_tar = None  # Initialize

        if self.cf.extra_ancilla:
            # raw_generated_state is potentially on N_tar + G's ancilla qubits
            # Use int(round(...)) for robustness converting float from log2 to int
            num_total_qubits_raw_G_output = int(round(pnp.log2(raw_generated_state.shape[0])))
            num_ancilla_G = num_total_qubits_raw_G_output - N_tar

            if num_ancilla_G > 0:
                train_log(
                    f"Fidelity for 'zero' mode with 'extra_ancilla=True': "
                    f"Tracing out {num_ancilla_G} ancilla qubit(s) from G's output "
                    f"(total {num_total_qubits_raw_G_output} qubits, system {N_tar} qubits).",
                    self.cf.log_path,
                )
                # Convert raw_generated_state (vector) to its density matrix
                raw_dm_generated_full = pnp.outer(
                    pnp.array(raw_generated_state, requires_grad=False),
                    pnp.conj(pnp.array(raw_generated_state, requires_grad=False))
                )
                # Wires are 0-indexed. Ancilla(s) are assumed to be the last ones.
                wires_to_trace_out = list(range(N_tar, num_total_qubits_raw_G_output))
                dm_generated_N_tar = qml.math.partial_trace(raw_dm_generated_full, wires_to_trace_out)

            elif num_ancilla_G == 0:
                # G's output is already N_tar, despite extra_ancilla=True.
                train_log(
                    f"Warning: Fidelity for 'zero' mode with 'extra_ancilla=True', but G's output "
                    f"({num_total_qubits_raw_G_output} qubits) matches N_tar ({N_tar}). "
                    "No partial trace performed. Check G's internal ancilla handling and qubit definitions.",
                    self.cf.log_path,
                )
                gcs_pnp = pnp.array(raw_generated_state, requires_grad=False)
                dm_generated_N_tar = pnp.outer(gcs_pnp, pnp.conj(gcs_pnp))

            else:  # num_ancilla_G < 0
                # G's output is smaller than N_tar, which is an error condition.
                train_log(
                    f"ERROR: Fidelity calculation in 'zero' mode with 'extra_ancilla=True'. "
                    f"G's output ({num_total_qubits_raw_G_output} qubits) is unexpectedly smaller than N_tar ({N_tar}). "
                    "Cannot compute valid fidelity. Returning 0.0.",
                    self.cf.log_path,
                )
                return 0.0

        else:  # No extra_ancilla for G, raw_generated_state is on N_tar qubits
            gcs_pnp = pnp.array(raw_generated_state, requires_grad=False)
            dm_generated_N_tar = pnp.outer(gcs_pnp, pnp.conj(gcs_pnp))

        return qml.math.fidelity(dm_generated_N_tar, dm_target_N_tar)

    def run(self):
        """Run the training, saving the data, the model, the logs, and the results plots."""

        f = self._calculate_current_fidelity()  # Initial fidelity

        fidelities_history, losses_G_history, losses_D_history = [], [], []  # Removed D_minus_G
        starttime = datetime.now()
        num_epochs = 0

        initial_fidelity_log = f"Initial Fidelity: {f:.6f}\\n"
        train_log(initial_fidelity_log, self.cf.log_path)

        # Loop until max_fidelity is reached or max_epochs
        while f < self.cf.max_fidelity:
            fidelities_epoch = np.zeros(self.cf.iterations_epoch)
            losses_G_epoch = np.zeros(self.cf.iterations_epoch)
            losses_D_epoch = np.zeros(self.cf.iterations_epoch)
            num_epochs += 1

            epoch_start_time = time.time()

            for iter_idx in range(self.cf.iterations_epoch):
                iter_start_time = time.time()
                loop_header = f"==================================================\\nEpoch {num_epochs}, Iteration {iter_idx + 1}, Learning_Rate {self.cf.learning_rate}\\n"
                train_log(loop_header, self.cf.log_path)

                # Get current parameters
                current_params_G = self.gen.get_params()
                current_params_D = self.dis.get_params()

                # 1. Prepare fake_state_for_D based on current_params_G
                fake_state_for_D = self._calculate_fake_state_for_D(current_params_G)
                # Ensure it's not differentiable for D's gradient calculation w.r.t D_params
                fake_state_for_D = pnp.array(fake_state_for_D, requires_grad=False)
                self.real_state = pnp.array(self.real_state, requires_grad=False)

                # 2. Discriminator update
                def _discriminator_objective(params_D_iter, fake_state_iter, real_state_iter):
                    # D circuit QNode should be used here
                    output_fake = self.dis._discriminator_circuit_qnode(fake_state_iter, params_D_iter)
                    output_real = self.dis._discriminator_circuit_qnode(real_state_iter, params_D_iter)
                    # D tries to make D(real) large (e.g., -> 1) and D(fake) small (e.g., -> -1)
                    # So D minimizes D(fake) - D(real)
                    return output_fake - output_real

                grad_D_fn = qml.grad(_discriminator_objective, argnum=0)
                gradients_D = grad_D_fn(current_params_D, fake_state_for_D, self.real_state)
                cost_D_val = _discriminator_objective(current_params_D, fake_state_for_D, self.real_state)
                self.dis.update_params(gradients_D)
                updated_params_D = self.dis.get_params()

                # 3. Generator update
                def _generator_objective(params_G_iter, params_D_eff_iter):
                    # Calculate fake state based on params_G_iter
                    _fake_state_for_G_obj = self._calculate_fake_state_for_D(params_G_iter)
                    # Ensure it's not differentiable for G's gradient calculation w.r.t D_params
                    _fake_state_for_G_obj = pnp.array(_fake_state_for_G_obj, requires_grad=False)

                    # G tries to make D(fake) large (e.g. -> 1)
                    # So G minimizes -D(fake)
                    output_fake_for_G = self.dis._discriminator_circuit_qnode(_fake_state_for_G_obj, params_D_eff_iter)
                    return -output_fake_for_G

                grad_G_fn = qml.grad(_generator_objective, argnum=0)
                gradients_G = grad_G_fn(current_params_G, updated_params_D)  # Use updated D params
                cost_G_val = _generator_objective(current_params_G, updated_params_D)
                self.gen.update_params(gradients_G)

                # 4. Fidelity and Logging
                current_fidelity = self._calculate_current_fidelity()

                fidelities_epoch[iter_idx] = current_fidelity
                losses_G_epoch[iter_idx] = cost_G_val.numpy() if hasattr(cost_G_val, "numpy") else cost_G_val
                losses_D_epoch[iter_idx] = cost_D_val.numpy() if hasattr(cost_D_val, "numpy") else cost_D_val

                iter_time = time.time() - iter_start_time
                iter_log_msg = (
                    f"Fidelity: {current_fidelity:.6f}, G_Loss: {cost_G_val:.6f}, D_Loss: {cost_D_val:.6f}, "
                    f"Iter time: {iter_time:.2f}s\\n==================================================\\n"
                )
                train_log(iter_log_msg, self.cf.log_path)

            f = fidelities_epoch[-1]
            fidelities_history.extend(fidelities_epoch.tolist())
            losses_G_history.extend(losses_G_epoch.tolist())
            losses_D_history.extend(losses_D_epoch.tolist())

            epoch_duration = time.time() - epoch_start_time
            epoch_log_msg = f"Epoch {num_epochs} completed. Fidelity: {f:.6f}. Duration: {epoch_duration:.2f}s\\n"
            train_log(epoch_log_msg, self.cf.log_path)

            plot_every = getattr(self.cf, "plot_every_epochs", 1)
            if num_epochs % plot_every == 0:
                plt_fidelity_vs_iter(
                    np.array(fidelities_history),
                    np.array(losses_G_history),
                    np.array(losses_D_history),
                    None,  # No D_minus_G loss anymore
                    self.cf,
                    num_epochs,
                )

            if num_epochs >= self.cf.epochs:
                max_epoch_log = f"Maximum number of epochs ({self.cf.epochs}) reached.\\n"
                train_log(max_epoch_log, self.cf.log_path)
                break

        training_finished_log = "Training finished.\\n"
        train_log(training_finished_log, self.cf.log_path)

        plt_fidelity_vs_iter(
            np.array(fidelities_history),
            np.array(losses_G_history),
            np.array(losses_D_history),
            None,  # No D_minus_G loss anymore
            self.cf,
            num_epochs,
        )

        # Reverted save_fidelity_loss to original signature (fidelities, G_loss, path)
        # D_loss is plotted and logged, but not saved by this specific function call.
        save_fidelity_loss(np.array(fidelities_history), np.array(losses_G_history), self.cf.fid_loss_path)

        self.gen.save_model(self.cf.model_gen_path)
        self.dis.save_model(self.cf.model_dis_path)

        # Save theta parameters using get_params()
        save_theta(self.gen.get_params(), self.cf.theta_path)

        endtime = datetime.now()
        total_training_time_seconds = (endtime - starttime).total_seconds()
        total_time_log = f"Total training time: {total_training_time_seconds:.2f} seconds\\nEnd of training script.\\n"
        train_log(total_time_log, self.cf.log_path)
