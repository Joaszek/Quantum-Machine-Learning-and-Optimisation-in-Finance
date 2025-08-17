import numpy as np
from qiskit.circuit import ParameterVector

from .circuits import  build_parametrized_circuit, observable_z0
from .gradients import predict, parameter_shift, finite_difference
from .loss import mse_loss


def train_qnn(estimator, X, y, n_qubits=2, depth=2, lr=0.4, epochs=300, grad_method="parameter_shift", seed=7):
    rng = np.random.default_rng(seed)

    p = depth * n_qubits * 2
    theta_vals = rng.uniform(-np.pi, np.pi, size=p)

    def circuit_template_fn(x, theta_params: ParameterVector):
        return build_parametrized_circuit(x, n_qubits, depth, theta_params)

    obs =observable_z0(n_qubits)

    grad_fn = {
        "parameter_shift": parameter_shift,
        "finite_difference": finite_difference
    }.get(grad_method)

    if grad_fn is None:
        raise ValueError("grad_method must be 'parameter_shift' or 'finite_difference'")

    for epoch in range(1, epochs+1):
        preds = [predict(estimator, circuit_template_fn, obs, theta_vals, x) for x in X]
        loss = mse_loss(preds, y)

        N = len(X)
        grad = np.zeros_like(theta_vals, dtype=float)
        for xi, yi, fi in zip(X, y, preds):
            df = grad_fn(estimator, circuit_template_fn, obs, theta_vals, xi)
            grad += (2.0 / N) * (fi - yi) * df

        theta_vals -= lr * grad

        if epoch % 25 == 0 or epoch == 1:
            preds_cls = np.sign(preds)
            acc = np.mean(preds_cls == y)
            print(f"Epoch {epoch:3d} | loss={loss:.4f} | acc={acc:.2f}")

    preds = [predict(estimator, circuit_template_fn, obs, theta_vals, x) for x in X]
    preds_cls = np.sign(preds)
    return {
        "theta": theta_vals,
        "preds": np.array(preds),
        "preds_cls": preds_cls.astype(int),
        "targets": y.astype(int),
    }