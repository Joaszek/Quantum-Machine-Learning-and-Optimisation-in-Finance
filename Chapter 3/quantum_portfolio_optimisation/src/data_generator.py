import numpy as np
import pandas as pd
import os


def generate_data(n_assets: int = 8, seed: int = 42):
    data_dir = "../data"

    mean_returns_path = os.path.join(data_dir, "mean_returns.csv")
    cov_matrix_path = os.path.join(data_dir, "covariance_matrix.csv")

    os.makedirs(data_dir, exist_ok=True)

    if not os.path.exists(mean_returns_path) or not os.path.exists(cov_matrix_path):
        np.random.seed(seed)
        mean_returns = np.random.uniform(0.05, 0.15, n_assets)

        A = np.random.randn(n_assets, n_assets)
        cov_matrix = np.dot(A, A.T)
        cov_matrix /= np.max(np.abs(cov_matrix))

        pd.DataFrame(mean_returns, columns=["mean_return"]).to_csv(mean_returns_path, index_label="asset")
        pd.DataFrame(cov_matrix).to_csv(cov_matrix_path, index_label="asset")

        print("Generated and saved new data.")
    else:
        print("Data files already exist. Skipping generation.")
