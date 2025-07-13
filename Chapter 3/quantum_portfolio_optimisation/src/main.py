from data_generator import generate_data
import os
import pandas as pd
from genetic_optimizer import genetic_algorithm
from qubo_optimizer import build_qubo_problem, solve_qubo

if __name__ == "__main__":

    generate_data()

    os.makedirs("results", exist_ok=True)

    print("=== Genetic Algorithm ===")
    best_solution, indices, cost = genetic_algorithm()
    print("\nBest portfolio (indices):", indices)
    print("Best cost (objective):", cost)
    pd.DataFrame({"selected_asset_indices": indices}).to_csv("results/best_ga_solution.csv", index=False)

    print("\n=== QUBO Optimization ===")
    qp = build_qubo_problem()
    result, qubo = solve_qubo(qp)

    x = result.x
    selected_indices = [i for i, val in enumerate(x) if val > 0.5]
    cost = result.fval
    print("\nQUBO solution:", selected_indices)
    print("Objective value (cost):", cost)
    pd.DataFrame({"selected_asset_indices": selected_indices}).to_csv("results/best_qubo_solution.csv", index=False)
