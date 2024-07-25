import sys
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import json
import math
import argparse
from numpy import linalg as LA
from gates import NumberPreserving_2, Spin_2, Spin_3, create_singlet

def create_Heisenberg(N, J1):
    H = sum([J1 * qml.PauliZ(i) @ qml.PauliZ((i + 1) % N) for i in range(N)])
    H += sum([J1 * qml.PauliX(i) @ qml.PauliX((i + 1) % N) for i in range(N)])
    H += sum([J1 * qml.PauliY(i) @ qml.PauliY((i + 1) % N) for i in range(N)])

    return H

def prepare_init_state(N):
    for i in range(0, N, 2):
        create_singlet(i, i+1)

def create_equivariant_circuit(N, num_blocks, H):
    # total params = 2Nl
    def circuit_2qubits_eq(params):
        prepare_init_state(N)
        k = 0
        for l in range(num_blocks):
            for i in range(0, N, 2):
                Spin_2(params[k], wires=[i, (i + 1) % N])
            k += 1
            for i in range(1, N, 2):
                Spin_2(params[k], wires=[i, (i + 1) % N])
            k += 1

        return qml.expval(H)

    return circuit_2qubits_eq

def create_non_equivariant_circuit(N, num_blocks, H):
    # total params = 2Nl
    def circuit_2qubits_noneq(params):
        prepare_init_state(N)
        k = 0
        for l in range(num_blocks):
            for i in range(0, N, 2):
                qml.IsingXX(params[k], wires=[i, (i + 1) % N])
                qml.IsingYY(params[k+1], wires=[i, (i + 1) % N])
                qml.IsingZZ(params[k+2], wires=[i, (i + 1) % N])

            for i in range(1, N, 2):
                qml.IsingXX(params[k+3], wires=[i, (i + 1) % N])
                qml.IsingYY(params[k+4], wires=[i, (i + 1) % N])
                qml.IsingZZ(params[k+5], wires=[i, (i + 1) % N])

            for i in range(N):
                qml.RY(params[k+6], wires = i)
            k += 7

        return qml.expval(H)

    return circuit_2qubits_noneq


if __name__ == "__main__":
    J1 = 1.0

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-qubits', type=int, required=True, dest="num_qubits")
    parser.add_argument('--gate', type=str, required=True, choices=["eq", "noneq"])
    parser.add_argument('--num-blocks', type=int, required=True, dest="num_blocks")
    parser.add_argument('--init-scale', type=float, default=1.0, dest="init_scale")
    parser.add_argument('--learning-rate', type=float, default=5e-3, dest="learning_rate")

    args = parser.parse_args()
    args = vars(args)

    print(json.dumps(args), file=sys.stderr)

    N = args["num_qubits"]
    ham = create_Heisenberg(N, J1)

    print(ham)
    print(qml.commutator(ham, sum(qml.PauliY(i) for i in range(N))))

    ham_sparse = qml.SparseHamiltonian(ham.sparse_matrix(), wires=range(N))
    dev = qml.device("lightning.qubit", wires=N)

    adam_step = args["learning_rate"]
    num_blocks = args["num_blocks"]
    init_scale = args["init_scale"]
    epochs = 2000
    opt = qml.AdamOptimizer(stepsize=adam_step)

    if args["gate"] == "eq":
        circuit = qml.QNode(create_equivariant_circuit(N, num_blocks, ham_sparse), dev, diff_method="adjoint")
        init = init_scale*math.pi/(2*num_blocks)
        params = init * pnp.random.rand(2 * num_blocks)
    elif args["gate"] == "noneq":
        circuit = qml.QNode(create_non_equivariant_circuit(N, num_blocks, ham_sparse), dev, diff_method="adjoint")
        init = init_scale*math.pi/(7*num_blocks)
        params = init * pnp.random.rand(7 * num_blocks)
    else:
        raise ValueError("spin gate size not supported")

    for epoch in range(epochs):
        params, cost = opt.step_and_cost(circuit, params)
        print(f"{epoch}\t{cost}")

    st = dev.state
    np.save("converged_state.npy", st)
