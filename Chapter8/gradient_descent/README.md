# QNN with Gradient Descent (Qiskit 1.0.2)

Minimal quantum neural network trained on XOR using vanilla gradient descent.
Supports:
- **Finite differences** (numerical)
- **Parameter-shift** (analytic for RX/RY/RZ)

Prediction is ⟨Z⟩ on qubit 0 (range [-1, 1]); labels are mapped to {-1, +1}.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # (or use your env manager)
pip install -r requirements.txt
