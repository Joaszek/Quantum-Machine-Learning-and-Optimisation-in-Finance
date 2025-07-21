import numpy as np
from qiskit.primitives import Sampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


class QBoostClassifier:
    def __init__(self, n_classifiers, regularization, max_depth, maxiter, reps):
        self.n_classifiers = n_classifiers
        self.regularization = regularization
        self.max_depth = max_depth
        self.maxiter = maxiter
        self.reps = reps
        self.selected_indices = []
        self.alpha_weights = []
        self.base_classifiers = []

    def build_qubo(self, predictions, y, n, m):

        Q = np.zeros((n,n))

        for i in range(n):
            for j in range(n):
                Q[i][j] = np.dot(predictions[i]-y, predictions[j]-y)/m

        for i in range(n):
            Q[i][i] += self.regularization

        qp = QuadraticProgram()

        for i in range(n):
            qp.binary_var(name=f"w_{i}")

        linear = {f"w_{i}": float(Q[i][i]) for i in range(n)}
        quadratic = {(f"w_{i}", f"w_{j}"): float(Q[i][j]) for i in range(n) for j in range(n)}

        qp.minimize(linear=linear, quadratic=quadratic)

        sampler = Sampler()
        qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=self.maxiter), reps=self.reps)
        solver = MinimumEigenOptimizer(qaoa)

        result = solver.solve(qp)

        self.selected_indices = [i for i, val in enumerate(result.x) if val == 1]
        self.alpha_weights = [1.0 if i in self.selected_indices else 0.0 for i in range(n)]


    def fit(self, X, y):
        predictions = []
        for i in range(self.n_classifiers):
            clf = DecisionTreeClassifier(max_depth=self.max_depth, random_state=i)
            clf.fit(X, y)
            pred = clf.predict(X)
            predictions.append(pred)
            self.base_classifiers.append(clf)

        predictions = np.array(predictions)
        y = np.array(y)
        n = self.n_classifiers
        m = len(y)

        self.build_qubo(predictions, y, n, m)

    def predict(self, X):
        pred_matrix = np.array([clf.predict(X) for clf in self.base_classifiers])
        weighted_matrix = np.dot(self.alpha_weights, pred_matrix)
        return (weighted_matrix >= (0.5*sum(self.alpha_weights))).astype(int)
