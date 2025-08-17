import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_parametrized_circuit(x, n_qubits, depth, theta: ParameterVector) -> QuantumCircuit:

    qc = QuantumCircuit(n_qubits)

    for i, xi in enumerate(x):
        if i>=n_qubits:
            break

        qc.rx(np.pi * float(xi), i)

    idx = 0
    for _ in range(depth):
        for q in range(n_qubits):
            qc.ry(theta[idx], q); idx += 1
            qc.rz(theta[idx], q); idx += 1

        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)

        if n_qubits > 1:
            qc.cx(n_qubits - 1, 0)

    return qc


def observable_z0(n_qubits) -> SparsePauliOp:
    label = ["I"] * n_qubits
    label[0] = "Z"
    return  SparsePauliOp("".join(label), coeffs=[1.0])



