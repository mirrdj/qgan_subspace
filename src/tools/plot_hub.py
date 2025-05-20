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

"""
plot_hub.py: the plot tool

"""

import os

import matplotlib as mpl

mpl.use("Agg")  # Ensure backend is set before pyplot import for non-GUI environments
import matplotlib.pyplot as plt
import numpy as np  # Import numpy for array operations


def plt_fidelity_vs_iter(fidelities: np.ndarray, losses: np.ndarray, config, indx: int = 0):
    """Plots fidelity and loss vs. iteration number and saves the figure.

    Args:
        fidelities (np.ndarray): Array of fidelity values.
        losses (np.ndarray): Array of loss values.
        config: Configuration object with attributes like figure_path, system_size, label.
        indx (int): Index for the figure filename.
    """
    fig, (axs1, axs2) = plt.subplots(1, 2, figsize=(12, 5))  # Adjusted figsize for better layout

    # Ensure fidelities and losses are numpy arrays for plotting
    fidelities_np = np.asarray(fidelities)
    losses_np = np.asarray(losses)

    axs1.plot(range(len(fidelities_np)), fidelities_np)
    axs1.set_xlabel("Iteration")  # Changed from Epoch to Iteration to match typical GAN training logs
    axs1.set_ylabel("Fidelity")  # Simplified label
    axs1.set_title("Fidelity vs. Iteration")  # Added title

    axs2.plot(range(len(losses_np)), losses_np)
    axs2.set_xlabel("Iteration")  # Changed from Epoch to Iteration
    axs2.set_ylabel("Discriminator Loss")  # Clarified loss type
    axs2.set_title("Loss vs. Iteration")  # Added title

    plt.tight_layout()

    # Save the figure
    fig_path = os.path.join(config.figure_path, f"{config.system_size}qubit_{config.layer}_{indx}.png")
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
    fig_path = os.path.join(config.figure_path, f"{config.system_size}qubit_{config.label}_projection_{indx}.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path)
    plt.close(fig)  # Close the figure
