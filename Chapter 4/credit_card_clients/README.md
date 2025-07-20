# QBoost Classifier on Credit Default Dataset (Qiskit 1.0.2)

This project implements a quantum-enhanced boosting classifier (QBoost) using [Qiskit 1.0.2](https://qiskit.org/) and applies it to the **"Default of Credit Card Clients"** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).

---

## Project Overview

QBoost is a quantum-enhanced ensemble learning algorithm that selects an optimal subset of weak classifiers using a Quadratic Unconstrained Binary Optimization (QUBO) formulation. This QUBO is solved using QAOA (Quantum Approximate Optimization Algorithm) on a gate-based quantum simulator.

We compare QBoost against classical models using standard classification metrics.

---

## Project Structure
qboost-credit-default/\
├── data/\
│ └── default_of_credit_card_clients.csv # Raw dataset (UCI)\
├── src/\
│ ├── data_loader.py # Loads and preprocesses data\
│ ├── qboost_model.py # QBoost implementation using QAOA\
│ └── benchmark.py # Classical model evaluation utilities\
├── notebooks/\
│ └── notebooks.ipynb # Optional EDA or result analysis\
├── main.py # Main script for training and evaluation\
├── requirements.txt\
└── README.md\

## Dataset

- **Source**: UCI Machine Learning Repository
- **Size**: 30,000 client records
- **Target**: Binary classification – will the client default on their credit card payment next month?

Download the dataset:

https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

And place it into the data folder
---

## How to Run

### 1. Install requirements

```bash
pip install -r requirements.txt
```
### 2. Run the app in the Jupyter Notebook