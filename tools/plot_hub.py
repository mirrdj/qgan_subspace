#!/usr/bin/env python

"""
plot_hub.py: the plot tool

"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plt_fidelity_vs_iter(fidelities, losses, config, indx=0):
    fig, (axs1, axs2) = plt.subplots(1, 2)
    axs1.plot(range(len(fidelities)), fidelities)
    axs1.set_xlabel("Epoch")
    axs1.set_ylabel("Fidelity between real and fake states")
    axs2.plot(range(len(losses)), losses)
    axs2.set_xlabel("Epoch")
    axs2.set_ylabel("Wasserstein Loss")
    plt.tight_layout()
    fig_path = f"{config.figure_path}/{config.system_size}qubit_{config.label}_{indx}.png"
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path)


def plt_fidelity_vs_iter_projection(fidelities, losses, probability_up, config, indx=0):
    # fig, (axs1, axs2) = plt.subplots(1, 2)
    fig = plt.figure()
    axs1 = plt.subplot(121)
    axs2 = plt.subplot(222)
    axs3 = plt.subplot(224)

    axs1.plot(range(len(fidelities)), fidelities)
    axs1.set_xlabel("Epoch")
    axs1.set_ylabel("Fidelity between real and fake states")
    axs2.plot(range(len(losses)), losses)
    axs2.set_xlabel("Epoch")
    axs2.set_ylabel("Wasserstein Loss")
    axs3.plot(range(len(probability_up)), probability_up)
    axs3.set_xlabel("Epoch")
    axs3.set_ylabel("Probability of additonal qubit being up")

    plt.tight_layout()

    fig_path = f"{config.figure_path}/{config.system_size}qubit_{config.label}_{indx}.png"
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path)
