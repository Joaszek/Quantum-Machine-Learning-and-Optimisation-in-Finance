import pandas as pd

from qiskit_algorithms import QAOA
from qiskit.primitives import Sampler
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo

N_ASSETS = 8
M_SELECTED = 3
RISK_AVERSION = 0.5

def read_data():
    mean_returns = pd.read_csv("../data/mean_returns.csv", index_col=0)["mean_return"].values
    cov_matrix = pd.read_csv("../data/covariance_matrix.csv", index_col=0).values

    return mean_returns, cov_matrix

def build_qubo_problem():

    qp = QuadraticProgram("PortfolioOptimization")

    mean_returns, cov_matrix = read_data()

    for i in range(N_ASSETS):
        qp.binary_var(name=f"x_{i}")

    linear = {f"x_{i}": -mean_returns[i] for i in range(N_ASSETS)}
    quadratic = {
        (f"x_{i}", f"x_{j}"): float(RISK_AVERSION * cov_matrix[i][j])
        for i in range(N_ASSETS) for j in range(N_ASSETS)
    }

    qp.minimize(linear=linear, quadratic=quadratic)

    qp.linear_constraint(
        linear={f"x_{i}": 1 for i in range(N_ASSETS)},
        sense="==",
        rhs=M_SELECTED,
        name="cardinally_constraint"
    )

    return qp

def solve_qubo(qp: QuadraticProgram):

    qubo_converter = QuadraticProgramToQubo()
    qubo = qubo_converter.convert(qp)

    sampler = Sampler()

    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=100), reps=2)

    optimizer = MinimumEigenOptimizer(qaoa)
    result = optimizer.solve(qubo)

    return result, qubo