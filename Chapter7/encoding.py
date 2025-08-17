from qiskit import QuantumCircuit, transpile
from qiskit.compiler import assemble
import numpy as np
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
from qiskit.visualization import plot_bloch_multivector

# Dane klasyczne (do zakodowania)
X = np.array([0.1, 0.4, 0.7, 1.0])

# Normalizacja do zakresu [0, π]
X_min, X_max = X.min(), X.max()
theta = (X - X_min) / (X_max - X_min) * np.pi

print("Zakodowane kąty θ:", theta)

# Tworzymy obwód kwantowy
qc = QuantumCircuit(4)

# Kodowanie danych przez rotacje Ry(θ)
for i in range(4):
    qc.ry(theta[i], i)

# Rysujemy obwód
qc.draw('mpl')

# Symulacja stanu końcowego
simulator = AerSimulator()
compiled_circuit = transpile(qc, simulator)
job = simulator.run(assemble(compiled_circuit))
result = job.result()
statevector = result.get_statevector()

# Bloch sfera
plot_bloch_multivector(statevector)
plt.show()