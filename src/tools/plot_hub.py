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

"""
plot_hub.py: the plot tool

"""

import os

import matplotlib as mpl

mpl.use("Agg")  # Ensure backend is set before pyplot import for non-GUI environments
import matplotlib.pyplot as plt
import numpy as np  # Import numpy for array operations


def plt_fidelity_vs_iter(
    fidelities: np.ndarray,
    losses_G: np.ndarray,
    losses_D: np.ndarray,
    losses_D_minus_G: np.ndarray,
    config,
    indx: int = 0,
):
    """Plots fidelity and various losses vs. iteration number and saves the figure.

    Args:
        fidelities (np.ndarray): Array of fidelity values.
        losses_G (np.ndarray): Array of Generator loss values.
        losses_D (np.ndarray): Array of Discriminator loss values.
        losses_D_minus_G (np.ndarray): Array of (Discriminator Loss - Generator Loss) values.
        config: Configuration object with attributes like figure_path, num_qubits, label.
        indx (int): Index for the figure filename.
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))  # Changed to 1 row, 2 columns

    # Ensure all inputs are numpy arrays for plotting
    fidelities_np = np.asarray(fidelities)
    losses_G_np = np.asarray(losses_G)
    losses_D_np = np.asarray(losses_D)
    losses_D_minus_G_np = np.asarray(losses_D_minus_G) if losses_D_minus_G is not None else np.array([])

    # Left subplot: Fidelity vs. Iteration
    axs[0].plot(range(len(fidelities_np)), fidelities_np)
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Fidelity")
    axs[0].set_title("Fidelity vs. Iteration")

    # Right subplot: All three losses vs. Iteration
    axs[1].plot(range(len(losses_G_np)), losses_G_np, color="blue", label="Generator Loss (cost_G)")
    axs[1].plot(range(len(losses_D_np)), losses_D_np, color="red", label="Discriminator Loss (cost_D)")
    if losses_D_minus_G_np.size > 0:
        axs[1].plot(range(len(losses_D_minus_G_np)), losses_D_minus_G_np, color="green", label="cost_D - cost_G")
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Loss")
    axs[1].set_title("Losses vs. Iteration")
    axs[1].legend()  # Add legend to distinguish the loss curves

    plt.tight_layout()

    # Save the figure
    fig_path = os.path.join(config.figure_path, f"{config.num_qubits}qubit_{config.gen_layers}_{indx}.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path)
    plt.close(fig)  # Close the figure to free memory


def plt_fidelity_vs_iter_projection(
    fidelities: np.ndarray, losses: np.ndarray, probability_up: np.ndarray, config, indx: int = 0
):
    """Plots fidelity, loss, and ancilla qubit probability vs. iteration number.

    Args:
        fidelities (np.ndarray): Array of fidelity values.
        losses (np.ndarray): Array of loss values.
        probability_up (np.ndarray): Array of probabilities for the ancilla qubit being in state |0> or |1>.
        config: Configuration object.
        indx (int): Index for the figure filename.
    """
    fig = plt.figure(figsize=(18, 5))  # Adjusted figsize
    axs1 = plt.subplot(131)  # Changed to 1 row, 3 columns
    axs2 = plt.subplot(132)
    axs3 = plt.subplot(133)

    # Ensure inputs are numpy arrays
    fidelities_np = np.asarray(fidelities)
    losses_np = np.asarray(losses)
    probability_up_np = np.asarray(probability_up)

    axs1.plot(range(len(fidelities_np)), fidelities_np)
    axs1.set_xlabel("Iteration")
    axs1.set_ylabel("Fidelity")
    axs1.set_title("Fidelity vs. Iteration")

    axs2.plot(range(len(losses_np)), losses_np)
    axs2.set_xlabel("Iteration")
    axs2.set_ylabel("Discriminator Loss")
    axs2.set_title("Loss vs. Iteration")

    axs3.plot(range(len(probability_up_np)), probability_up_np)
    axs3.set_xlabel("Iteration")
    axs3.set_ylabel("Ancilla Qubit Up Probability")  # Clarified label
    axs3.set_title("Ancilla Up Probability vs. Iteration")

    plt.tight_layout()

    # Save the figure
    fig_path = os.path.join(config.figure_path, f"{config.num_qubits}qubit_{config.label}_projection_{indx}.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path)
    plt.close(fig)  # Close the figure
