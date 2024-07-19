import pennylane as qml
import numpy as np

class NumberPreserving_2(qml.operation.Operation):
    num_wires = 2

    @staticmethod
    def compute_decomposition(theta, phi, wires):
        return [qml.CNOT(wires[::-1]),
                qml.RZ(-phi - np.pi, wires[1]),
                qml.RY(-theta - np.pi/2, wires[1]),
                qml.CNOT(wires),
                qml.RY(theta + np.pi/2, wires[1]),
                qml.RZ(phi + np.pi, wires[1]),
                qml.CNOT(wires[::-1])]

class Spin_2(qml.operation.Operation):
    num_wires = 2

    @staticmethod
    def compute_decomposition(theta, wires):
        schur2 = np.array([
            [1, 0, 0, 0],
            [0, 1/np.sqrt(2), 1/np.sqrt(2), 0],
            [0, 0, 0, 1],
            [0, 1/np.sqrt(2), -1/np.sqrt(2), 0]
        ])
    
        return [qml.QubitUnitary(schur2, wires = wires),
                qml.ControlledPhaseShift(theta, wires = wires),
                qml.adjoint(qml.QubitUnitary)(schur2, wires = wires)]

class Spin_3(qml.operation.Operation):
    num_wires = 3
    
    @staticmethod
    def compute_decomposition(theta0, theta1, theta2, theta3, wires):
        schur = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1/np.sqrt(3), 1/np.sqrt(3), 0, 1/np.sqrt(3), 0, 0, 0],
            [0, 0, 0, 1/np.sqrt(3), 0, 1/np.sqrt(3), 1/np.sqrt(3), 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, -1/np.sqrt(2), 0, 1/np.sqrt(2), 0, 0, 0],
            [0, 0, 0, -1/np.sqrt(2), 0, 1/np.sqrt(2), 0, 0],
            [0, np.sqrt(2)/np.sqrt(3), -1/np.sqrt(6), 0, -1/np.sqrt(6), 0, 0, 0],
            [0, 0, 0, 1/np.sqrt(6), 0, 1/np.sqrt(6), -np.sqrt(2)/np.sqrt(3), 0]
        ])
    
        return [qml.QubitUnitary(schur, wires = wires),
                qml.CRot(theta1, theta2, theta3, wires = wires[:2]),
                qml.PhaseShift(theta0, wires=wires[0]),
                qml.adjoint(qml.QubitUnitary)(schur, wires = wires)]

def create_singlet(i, j):
    qml.Hadamard(wires=i)
    qml.PauliZ(wires=i)
    qml.CNOT(wires=[i, j])
    qml.PauliX(wires=j)
