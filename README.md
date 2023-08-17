# All you need is spin: SU(2) equivariant variational quantum circuits based on spin-networks

<img src="https://github.com/XanaduAI/all-you-need-is-spin/blob/main/static/one-dim-j1j2.png?raw=true" style="width:65em;">
<img src="https://github.com/XanaduAI/all-you-need-is-spin/blob/main/static/kagome18.png?raw=true" style="width:30em;">

## Usage
In the `python_src` directory, there are two scripts. These scripts can simulate the variational algorithm with SU(2) equivariant gates. For example, the following will simulate the algorithm with two-qubit vertex gates (given by `--gate` argument) for 20 qubits (`--num_qubits`), $J_2=0.44$ (`--j2`), and four blocks of gates (`--num_blocks`; $p$ in the paper). The default learning rate `5e-3` is used.
```python
python3 ./python_src/heisenberg_1d.py --num-qubits 20 --gate 2 --num-blocks 4 --j2 0.44
```

Similarly, one can simulate the model on the Kagome lattice with $n=18$ qubits. 
```python
python3 ./python_src/kagome_lattice.py --num-qubits 20 --gate 3 --num-blocks 5 --no-share-param
```
The final parameter, `--no-share-param`, controls whether we want to share parameters among gates. We have used `--no-share-param` for all data presented in the paper.


## Warnings
Currently, the feature we used here is not fully integrated with the latest release of PennLane-Lightning. Thus it is required to compile PennyLane-Lightning from the source. The recommended way is using Python virtual environment with `requirements.txt`. So after cloning the repository, you may run 
```bash
$ python3 -m venv env
$ source ./env/bin/activate
$ pip install -r ./requirements.txt
```

You need a compiler with proper C++20 supports. See [PennyLane-Lightning](https://github.com/PennyLaneAI/pennylane-lightning) repository for further instructions.
