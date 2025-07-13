import numpy as np
import pandas as pd
import os

N_ASSETS=8
M_SELECTED=3
POPULATION_SIZE=50
N_ITERATIONS=100
TOP_K=10
SEED=42

np.random.seed(SEED)

def read_data():
    mean_returns = pd.read_csv("../data/mean_returns.csv", index_col=0)["mean_return"].values
    cov_matrix = pd.read_csv("../data/covariance_matrix.csv", index_col=0).values

    return mean_returns, cov_matrix

def calculate_cost(solution):

    mean_returns, cov_matrix = read_data()

    indices = np.where(solution == 1)[0]
    if len(indices) == 0:
        return np.inf

    mean = np.sum(mean_returns[indices]) / len(indices)
    risk = np.dot(solution.T, np.dot(cov_matrix, solution)) / len(indices)**2
    return -mean + risk

def  generate_initial_population():
    population = []
    for _ in range(POPULATION_SIZE):
        solution = np.zeros(N_ASSETS, dtype=int)
        selected_indices = np.random.choice(N_ASSETS, size=M_SELECTED, replace=False)
        solution[selected_indices] = 1
        population.append(solution)

    return population

def mutate(solution):
    ones = np.where(solution == 1)[0]
    zeros = np.where(solution == 0)[0]
    if len(ones) == 0 or len(zeros) == 0:
        return solution.copy()

    i = np.random.choice(ones)
    j = np.random.choice(zeros)
    new_solution = solution.copy()
    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution

def genetic_algorithm():
    population = generate_initial_population()
    population = sorted(population, key=calculate_cost)

    for iteration in range(N_ITERATIONS):
        new_population = []
        top_k = population[:TOP_K]

        for parent in top_k:
            for _ in range(POPULATION_SIZE // TOP_K):
                child = mutate(parent)
                new_population.append(child)

        population = sorted(new_population, key=calculate_cost)
        best_cost = calculate_cost(population[0])
        print(f"Iteration {iteration + 1}: Best cost = {best_cost:.6f}")

    best_solution = population[0]
    best_indices = np.where(best_solution == 1)[0]
    best_cost = calculate_cost(best_solution)

    return best_solution, best_indices, best_cost