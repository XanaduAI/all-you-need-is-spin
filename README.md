# All you need is spin: SU(2) equivariant variational quantum circuits based on spin-networks

This repository contains code to reproduce the experiments reported in the paper East, Alonso-Linaje and Park: "All you ned is spin: SU(2) equivariant variational quantum circuits based on spin-networks".

<img src="https://github.com/XanaduAI/all-you-need-is-spin/blob/main/static/one-dim-j1j2.png?raw=true" style="width:65em;">
<img src="https://github.com/XanaduAI/all-you-need-is-spin/blob/main/static/kagome18.png?raw=true" style="width:30em;">

## Usage
In the `python_src` directory there are two scripts to simulate the variational algorithm with SU(2) equivariant gates on two different problems. For example, the following will simulate the algorithm with two-qubit vertex gates (given by `--gate` argument) for 20 qubits (`--num_qubits`), $J_2=0.44$ (`--j2`), and four blocks of gates (`--num_blocks`; $p$ in the paper). The default learning rate `5e-3` is used.
```python
python3 ./python_src/heisenberg_1d.py --num-qubits 20 --gate 2 --num-blocks 4 --j2 0.44
```

Similarly, one can simulate the model on the Kagome lattice with $n=18$ qubits. 
```python
python3 ./python_src/kagome_lattice.py --num-qubits 20 --gate 3 --num-blocks 5 --no-share-param
```
The final parameter, `--no-share-param`, controls whether we want to share parameters among gates. We have used `--no-share-param` for all data presented in the paper.


## Warnings
Currently, the feature we used for the simulations is not fully integrated with the latest release of PennLane-Lightning. It is instead implemented in [merge_mat_sparse_adj](https://github.com/PennyLaneAI/pennylane-lightning/tree/merge_mat_sparse_adj) branch of PennyLane-Lightning. Thus users need to compile this branch from source. The recommended way is using a Python virtual environment populated with the packages in `requirements.txt`. After cloning the repository, you may run 
```bash
$ python3 -m venv env
$ source ./env/bin/activate
$ pip install -r ./requirements.txt
```

You need a compiler with proper C++20 supports. See the [PennyLane-Lightning](https://github.com/PennyLaneAI/pennylane-lightning) repository for further instructions.
