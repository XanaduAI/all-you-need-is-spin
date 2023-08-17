import sys
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import json
import math
import argparse
from scipy.sparse import linalg as LA

from gates import Spin_3, create_singlet

UPPER_TRIANGLES = {
    "a": [0, 1, 2],
    "b": [12, 3, 4],
    "c": [5, 7, 8],
    "d": [6, 9, 10],
    "e": [16, 11, 17],
    "f": [13, 14, 15],
}
LOWER_TRIANGLES = {
    "A": [0, 7, 17],
    "B": [5, 3, 2],
    "C": [6, 14, 4],
    "D": [12, 9, 8],
    "E": [13, 11, 10],
    "F": [16, 1, 15]
}

def create_Kagome18():
    edges = [
        (0, 17), (0, 7), #2
        (0, 1), (0, 2), (3, 12), (4, 12),  #4
        (1, 2), (2, 3), (3, 4), #3
        (2, 5), (3, 5), (4, 6), (6, 14), #4
        (5, 7), (5, 8), (6,9), (6, 10), (11, 16), #5
        (7, 17), (7, 8), (8, 9), (9, 10), (10, 11), (11, 17), #6
        (8, 12), (9, 12), (10, 13), (11, 13), #4
        (13, 14), (13, 15), #2
        (4, 14), (14, 15), (1, 15), #2
        (15, 16), (1, 16), #2
        (16, 17) #1 total 35
    ]
    J1 = 1.0
    H = sum([J1 * qml.PauliZ(i) @ qml.PauliZ(j) for i, j in edges])
    H += sum([J1 * qml.PauliX(i) @ qml.PauliX(j) for i, j in edges])
    H += sum([J1 * qml.PauliY(i) @ qml.PauliY(j) for i, j in edges])

    return H

def create_u3_circuit(N, nblocks, H):
    singlets = [(0, 17), (1, 2), (3, 4), (5, 7), (6, 9), (11, 16), (8, 12), (10, 13), (14, 15)]

    def circuit_3qubits(params):
        for singlet in singlets:
            create_singlet(*singlet)

        for block_idx in range(nblocks):
            for ut_idx, wires in enumerate(UPPER_TRIANGLES.values()):
                start_idx = 48*block_idx + 4*ut_idx 
                Spin_3(params[start_idx + 0], params[start_idx + 1], params[start_idx + 2], params[start_idx + 3], wires=wires)

            for lt_idx, wires in enumerate(LOWER_TRIANGLES.values()):
                start_idx = 48*block_idx + 4*lt_idx + 24
                Spin_3(params[start_idx + 0], params[start_idx + 1], params[start_idx + 2], params[start_idx + 3], wires=wires)

        return qml.expval(H)

    return circuit_3qubits

def create_u3_circuit_param_shared(N, nblocks, H):
    singlets = [(0, 17), (1, 2), (3, 4), (5, 7), (6, 9), (11, 16), (8, 12), (10, 13), (14, 15)]

    def circuit_3qubits_param_shared(params):
        for singlet in singlets:
            create_singlet(*singlet)

        for block_idx in range(nblocks):
            start_idx = 16*block_idx
            for k in ["a", "b", "f"]:
                Spin_3(params[start_idx + 0], params[start_idx + 1], params[start_idx + 2], params[start_idx + 3], wires=UPPER_TRIANGLES[k])
            start_idx += 4

            for k in ["A", "D", "E"]:
                Spin_3(params[start_idx + 0], params[start_idx + 1], params[start_idx + 2], params[start_idx + 3], wires=LOWER_TRIANGLES[k])
            start_idx += 4

            for k in ["c", "d", "e"]:
                Spin_3(params[start_idx + 0], params[start_idx + 1], params[start_idx + 2], params[start_idx + 3], wires=UPPER_TRIANGLES[k])
            start_idx += 4

            for k in ["B", "C", "F"]:
                Spin_3(params[start_idx + 0], params[start_idx + 1], params[start_idx + 2], params[start_idx + 3], wires=LOWER_TRIANGLES[k])

        return qml.expval(H)

    return circuit_3qubits_param_shared

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-blocks', type=int, required=True, dest="num_blocks")
    parser.add_argument('--init-scale', type=float, default=1.0, dest="init_scale")
    parser.add_argument('--learning-rate', type=float, default=1e-2, dest="learning_rate")
    parser.add_argument('--share-param', action='store_true', dest="share_param")
    parser.add_argument('--no-share-param', dest='share_param', action='store_false')

    args = parser.parse_args()
    args = vars(args)

    print(json.dumps(args), file=sys.stderr)

    N = 18
    ham = create_Kagome18()
    ham_sparse = qml.SparseHamiltonian(ham.sparse_matrix(), wires=range(N))

    dev = qml.device("lightning.qubit", wires=N)

    num_blocks = args["num_blocks"]
    init_scale = args["init_scale"]
    adam_step = args["learning_rate"]
    epochs = 2000
    opt = qml.AdamOptimizer(stepsize = adam_step)

    if args["share_param"]:
        circuit = qml.QNode(create_u3_circuit_param_shared(N, num_blocks, ham_sparse), dev, diff_method="adjoint")
        init = init_scale*math.pi/(48*num_blocks)
        params = init * pnp.random.rand(16*num_blocks)
    else:
        circuit = qml.QNode(create_u3_circuit(N, num_blocks, ham_sparse), dev, diff_method="adjoint")
        init = init_scale*math.pi/(48*num_blocks)
        params = init * pnp.random.rand(48*num_blocks)

    for epoch in range(epochs):
        params, cost = opt.step_and_cost(circuit, params)
        print(f"{epoch}\t{cost}")

    st = dev.state
    np.save("converged_state.npy", st)
