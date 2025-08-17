import numpy as np
from qiskit.circuit import ParameterVector

def predict(estimator, circuit_template_fn, obs, theta_vals, x):
    theta_params = ParameterVector("θ", len(theta_vals))
    qc = circuit_template_fn(x, theta_params)
    res = estimator.run(
        circuits=[qc],
        observables=[obs],
        parameter_values=[list(theta_vals)]
    ).result()
    return float(res.values[0])

def parameter_shift(estimator, circuit_template_fn, obs, theta_vals, x):
    grad = np.zeros_like(theta_vals, dtype=float)
    for k in range(len(theta_vals)):
        tp = theta_vals.copy(); tp[k] += np.pi/2
        tm = theta_vals.copy(); tm[k] -= np.pi/2
        f_plus = predict(estimator, circuit_template_fn, obs, tp, x)
        f_minus = predict(estimator, circuit_template_fn, obs, tm, x)
        grad[k] = 0.5 * (f_plus - f_minus)

    return grad

def finite_difference(estimator, circuit_template_fn, obs, theta_vals, x, eps=1e-4):
    grad = np.zeros_like(theta_vals, dtype=float)
    for k in range(len(theta_vals)):
        tp = theta_vals.copy(); tp[k] += eps
        tm = theta_vals.copy(); tm[k] -= eps
        f_plus = predict(estimator, circuit_template_fn, obs, tp, x)
        f_minus = predict(estimator, circuit_template_fn, obs, tm, x)
        grad[k] = (f_plus - f_minus) / (2.0 * eps)
    return grad