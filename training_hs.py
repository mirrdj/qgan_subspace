import os
import sys
import time
from datetime import datetime

import numpy as np
import scipy.io as scio

import config_hs as cf
from model_hs import Discriminator, Generator, compute_cost, compute_fidelity
from tools.plot_hub import plt_fidelity_vs_iter
from tools.qcircuit import *
from tools.utils import (
    get_maximally_entangled_state,
    get_maximally_entangled_state_in_subspace,
    get_zero_state,
    save_model,
    train_log,
)

np.random.seed()

##### target gates


def term_XXXX(size, qubit1, qubit2, qubit3, qubit4):
    matrix = 1
    for i in range(size):
        if (qubit1 == i) or (qubit2 == i) or (qubit3 == i) or (qubit4 == i):
            matrix = np.kron(matrix, X)
        else:
            matrix = np.kron(matrix, I)
    return matrix


def term_ZZZZ(size, qubit1, qubit2, qubit3, qubit4):
    matrix = 1
    for i in range(size):
        if (qubit1 == i) or (qubit2 == i) or (qubit3 == i) or (qubit4 == i):
            matrix = np.kron(matrix, Z)
        else:
            matrix = np.kron(matrix, I)
    return matrix


def term_ZZZ(size, qubit1, qubit2, qubit3):
    matrix = 1
    for i in range(size):
        if (qubit1 == i) or (qubit2 == i) or (qubit3 == i):
            matrix = np.kron(matrix, Z)
        else:
            matrix = np.kron(matrix, I)
    return matrix


def term_XZX(size, qubit1, qubit2, qubit3):
    matrix = 1
    for i in range(size):
        if (qubit1 == i) or (qubit3 == i):
            matrix = np.kron(matrix, X)
        elif qubit2 == i:
            matrix = np.kron(matrix, Z)
        else:
            matrix = np.kron(matrix, I)
    return matrix


def term_XX(size, qubit1, qubit2):
    matrix = 1
    for i in range(size):
        if (qubit1 == i) or (qubit2 == i):
            matrix = np.kron(matrix, X)
        else:
            matrix = np.kron(matrix, I)
    return matrix


def term_ZZ(size, qubit1, qubit2):
    matrix = 1
    for i in range(size):
        if (qubit1 == i) or (qubit2 == i):
            matrix = np.kron(matrix, Z)
        else:
            matrix = np.kron(matrix, I)
    return matrix


def term_Z(size, qubit1):
    matrix = 1
    for i in range(size):
        if qubit1 == i:
            matrix = np.kron(matrix, Z)
        else:
            matrix = np.kron(matrix, I)
    return matrix


def construct_target(size):
    H = np.zeros([2**size, 2**size])

    # for i in range(size):
    #     H += term_Z(size,i)

    for i in range(size - 1):
        H += term_ZZ(size, i, i + 1)

    # for i in range(size-2):
    #     H += term_ZZZ(size,i,i+1,i+2)

    # H = np.identity(2**size)

    return linalg.expm(-1j * H)


def construct_clusterH(size):
    H = np.zeros([2**size, 2**size])
    for i in range(size - 2):
        H += term_XZX(size, i, i + 1, i + 2)
        H += term_Z(size, i)
    H += term_Z(size, size - 2)
    H += term_Z(size, size - 1)
    return linalg.expm(-1j * H)


def construct_RotatedSurfaceCode(size):
    H = np.zeros([2**size, 2**size])

    if size == 4:
        H += -term_XXXX(size, 0, 1, 2, 3)
        H += -term_ZZ(size, 0, 1)
        H += -term_ZZ(size, 2, 3)
    elif size == 9:
        H += -term_XXXX(size, 0, 1, 3, 4)
        H += -term_XXXX(size, 4, 5, 7, 8)
        H += -term_XX(size, 2, 5)
        H += -term_XX(size, 3, 6)
        H += -term_ZZZZ(size, 1, 2, 4, 5)
        H += -term_ZZZZ(size, 3, 4, 6, 7)
        H += -term_ZZ(size, 0, 1)
        H += -term_ZZ(size, 7, 8)
    else:
        sys.exit("system size is not 2*2 or 3*3 either")

    return linalg.expm(-1j * H)


##### design generator


def construct_qcircuit_XYZandfieldZ(qc, size, layer):
    entg_list = ["XX", "YY", "ZZ"]
    for j in range(layer):
        for i in range(size):
            if i < size - 1:
                for gate in entg_list:
                    qc.add_gate(Quantum_Gate(gate, i, i + 1, angle=0.5000 * np.pi))
                qc.add_gate(Quantum_Gate("Z", i, angle=0.5000 * np.pi))
        for gate in entg_list:
            qc.add_gate(Quantum_Gate(gate, 0, size - 1, angle=0.5000 * np.pi))
        qc.add_gate(Quantum_Gate("Z", size - 1, angle=0.5000 * np.pi))
        # qc.add_gate(Quantum_Gate("G", None, angle=0.5000 * np.pi))

    theta = np.random.randn(len(qc.gates))
    for i in range(len(qc.gates)):
        qc.gates[i].angle = theta[i]

    return qc


def construct_qcircuit_ZZ_XZ(qc, size, layer):
    for j in range(layer):
        for i in range(size):
            qc.add_gate(Quantum_Gate("X", i, angle=0.5000 * np.pi))
            qc.add_gate(Quantum_Gate("Z", i, angle=0.5000 * np.pi))
        for i in range(size - 1):
            qc.add_gate(Quantum_Gate("ZZ", i, i + 1, angle=0.5000 * np.pi))

    theta = np.random.randn(len(qc.gates))
    for i in range(len(qc.gates)):
        qc.gates[i].angle = theta[i]

    return qc


##### main


def main():
    # preparation for Choi state
    input_state = get_maximally_entangled_state(cf.system_size)
    # input_state = get_maximally_entangled_state_in_subspace(cf.system_size) # 2*(size+1), B0, B1, A0, A1

    # define target gates
    # target_unitary = scio.loadmat('./exp_ideal_{}_qubit.mat'.format(cf.system_size))['exp_ideal']
    target_unitary = construct_target(cf.system_size)
    # target_unitary = construct_clusterH(cf.system_size)
    # target_unitary = construct_RotatedSurfaceCode(cf.system_size)

    # define target state
    real_state = np.matmul(np.kron(target_unitary, Identity(cf.system_size)), input_state)
    # real_state = np.matmul(np.kron(np.kron(np.kron(target_unitary, Identity(1)), Identity(cf.system_size)), Identity(1)), input_state)

    # define generator
    gen = Generator(cf.system_size)
    # gen.set_qcircuit(construct_qcircuit_XYZandfieldZ(gen.qc, cf.system_size, cf.layer))
    gen.set_qcircuit(construct_qcircuit_ZZ_XZ(gen.qc, cf.system_size, cf.layer))

    # define discriminator
    herm = [I, X, Y, Z]
    dis = Discriminator(herm, cf.system_size * 2)
    # dis = Discriminator(herm, (cf.system_size+1)*2)

    # compute fidelity at initial
    f = compute_fidelity(gen, input_state, real_state, input_state)

    fidelities = np.zeros(cf.epochs)
    losses = np.zeros(cf.epochs)
    fidelities_history = []
    losses_history = []
    starttime = datetime.now()
    num_epochs = 0

    while f < 0.99:
        # while (f < 0.95):
        fidelities[:] = 0.0
        losses[:] = 0.0
        num_epochs += 1
        for iter in range(cf.epochs):
            print("==================================================")
            print("Epoch {}, Step_size {}".format(iter + 1 + (num_epochs - 1) * cf.epochs, cf.eta))

            # Generator gradient descent
            gen.update_gen(dis, real_state, input_state)
            # print("Loss after generator step: {}".format(compute_cost(gen, dis, real_state, input_state)))

            # Discriminator gradient ascent
            dis.update_dis(gen, real_state, input_state)
            # print("Loss after discriminator step: {}".format(compute_cost(gen, dis, real_state, input_state)))

            fidelities[iter] = compute_fidelity(gen, input_state, real_state)
            losses[iter] = compute_cost(gen, dis, real_state, input_state)

            print("Fidelity between real and fake state: {}".format(fidelities[iter]))
            print("==================================================")

            if iter % 10 == 0:
                # save log
                endtime = datetime.now()
                training_duration = (endtime - starttime).seconds / float(3600)
                param = "epoches:{:4d} | fidelity:{:8f} | time:{:10s} | duration:{:8f}\n".format(
                    iter,
                    round(fidelities[iter], 6),
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    round(training_duration, 2),
                )
                train_log(param, "./{}qubit_log_hs.txt".format(cf.system_size))

        f = compute_fidelity(gen, input_state, real_state)
        fidelities_history = np.append(fidelities_history, fidelities)
        losses_history = np.append(losses_history, losses)
        plt_fidelity_vs_iter(fidelities_history, losses_history, cf, num_epochs)

        if num_epochs >= 10:
            print("The number of epochs exceeds 10.")
            break

    # save data of fidelity and loss
    if os.path.exists("qubit_log_hs_fidelity_loss.npy"):
        os.remove("qubit_log_hs_fidelity_loss.npy")
    with open("qubit_log_hs_fidelity_loss.npy", "wb") as f:
        np.save(f, fidelities_history)
        np.save(f, losses_history)

    # save data of the generator and the discriminator
    save_model(gen, cf.model_gen_path)
    save_model(dis, cf.model_dis_path)

    # output the parameters of the generator
    array_angle = np.zeros(len(gen.qc.gates))
    for i in range(len(gen.qc.gates)):
        array_angle[i] = gen.qc.gates[i].angle
    np.savetxt("theta_gen.txt", array_angle)

    endtime = datetime.now()
    print("{} seconds".format((endtime - starttime).seconds))
    print("end")


main()
