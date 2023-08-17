import sys
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import json
import math
import argparse
from gates import Spin_2, Spin_3, create_singlet

def create_Heisenberg(N, J1, J2):
    H = sum([J1 * qml.PauliZ(i) @ qml.PauliZ((i + 1) % N) for i in range(N)])
    H += sum([J1 * qml.PauliX(i) @ qml.PauliX((i + 1) % N) for i in range(N)])
    H += sum([J1 * qml.PauliY(i) @ qml.PauliY((i + 1) % N) for i in range(N)])

    H += sum([J2 * qml.PauliZ(i) @ qml.PauliZ((i + 2) % N) for i in range(N)])
    H += sum([J2 * qml.PauliX(i) @ qml.PauliX((i + 2) % N) for i in range(N)])
    H += sum([J2 * qml.PauliY(i) @ qml.PauliY((i + 2) % N) for i in range(N)])

    return H

def prepare_init_state(N):
    for i in range(0, N, 2):
        create_singlet(i, i+1)

def create_u2_circuit(N, num_blocks, H):
    # total params = 2Nl
    def circuit_2qubits(params):
        prepare_init_state(N)
        k = 0
        for l in range(num_blocks):
            for i in range(0, N, 2):
                Spin_2(params[k], wires=[i, (i + 1) % N])
                k += 1

            for i in range(1, N, 2):
                Spin_2(params[k], wires=[i, (i + 1) % N])
                k += 1

            for i in range(0, N):
                Spin_2(params[k], wires=[i, (i + 2) % N])
                k += 1

        return qml.expval(H)

    return circuit_2qubits


def create_u3_circuit(N, num_blocks, H):
    # total params = 4Nl
    def circuit_3qubits(params):
        prepare_init_state(N)
        k = 0
        for l in range(num_blocks):
            for i in range(0, N):
                Spin_3(params[k], params[k + 1], params[k + 2], params[k+3], wires=[i, (i + 1) % N, (i + 2) % N])
                k += 4

        return qml.expval(H)

    return circuit_3qubits


if __name__ == "__main__":
    J1 = 1.0

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-qubits', type=int, required=True, dest="num_qubits")
    parser.add_argument('--gate', type=int, required=True)
    parser.add_argument('--num-blocks', type=int, required=True, dest="num_blocks")
    parser.add_argument('--j2', type=float, required=True)
    parser.add_argument('--init-scale', type=float, default=1.0, dest="init_scale")
    parser.add_argument('--learning-rate', type=float, default=5e-3, dest="learning_rate")

    args = parser.parse_args()
    args = vars(args)

    print(json.dumps(args), file=sys.stderr)

    N = args["num_qubits"]
    J2 = args["j2"]
    ham = create_Heisenberg(N, J1, J2)
    ham_sparse = qml.SparseHamiltonian(ham.sparse_matrix(), wires=range(N))
    dev = qml.device("lightning.qubit", wires=N)

    adam_step = args["learning_rate"]
    num_blocks = args["num_blocks"]
    init_scale = args["init_scale"]
    epochs = 2000
    opt = qml.AdamOptimizer(stepsize=adam_step)

    if args["gate"] == 2:
        circuit = qml.QNode(create_u2_circuit(N, num_blocks, ham_sparse), dev, diff_method="adjoint")
        init = init_scale*math.pi/(2*N*num_blocks)
        params = init * pnp.random.rand(2 * N * num_blocks)
    elif args["gate"] == 3:
        circuit = qml.QNode(create_u3_circuit(N, num_blocks, ham_sparse), dev, diff_method="adjoint")
        init = init_scale*math.pi/(4*N*num_blocks)
        params = init * pnp.random.rand(4 * N * num_blocks)
    else:
        raise ValueError("spin gate size not supported")

    for epoch in range(epochs):
        params, cost = opt.step_and_cost(circuit, params)
        print(f"{epoch}\t{cost}")

    st = dev.state
    np.save("converged_state.npy", st)
