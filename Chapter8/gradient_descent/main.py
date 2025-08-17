import argparse
import numpy as np
from qiskit_aer.primitives import Estimator

from data.data import xor_dataset
from src.train import train_qnn


def parse_args():
    ap = argparse.ArgumentParser(description="QNN with Gradient Descent (Qiskit 1.0.2)")
    ap.add_argument("--grad", choices=["parameter_shift", "finite_difference"], default="parameter_shift")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.4)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--qubits", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    return ap.parse_args()


def main():
    args = parse_args()
    estimator = Estimator()
    X, y = xor_dataset()

    result = train_qnn(
        estimator,
        X, y,
        n_qubits=args.qubits,
        depth=args.depth,
        lr=args.lr,
        epochs=args.epochs,
        grad_method=args.grad,
        seed=args.seed,
    )

    print("\nFinal predictions (⟨Z⟩):", np.round(result["preds"], 3).tolist())
    print("Final classes (sign):     ", result["preds_cls"].tolist())
    print("Targets:                  ", result["targets"].tolist())


if __name__ == "__main__":
    main()
