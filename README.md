# Courier Route Optimization

This project implements and compares several optimization algorithms to solve a courier route optimization problem. The algorithms used include:

1. Genetic Algorithm (GA)
2. Simulated Annealing (SA)
3. Simulated Annealing with Local Search (SAL)
4. Tabu Search (TS)

The goal is to minimize the total travel distance of couriers while considering constraints such as package capacity and delivery time windows.

## Table of Contents

- [Project Description](#project-description)
- [Algorithms](#algorithms)
- [Visualization](#visualization)
- [Results](#results)
- [License](#license)

## Project Description

The problem involves a fixed number of couriers delivering a set number of packages within a city. Each courier has a different speed and package capacity. Packages have specific delivery time windows that need to be adhered to. The objective is to find the optimal routes for couriers to minimize the total distance traveled while respecting the constraints.

## Algorithms

### Genetic Algorithm (GA)

A population-based search algorithm inspired by the process of natural selection. It uses operations like selection, crossover, and mutation to evolve solutions over generations.

### Simulated Annealing (SA)

A probabilistic technique for approximating the global optimum of a given function. It mimics the cooling process of materials to escape local optima by allowing worse solutions based on a temperature parameter.

### Simulated Annealing with Local Search (SAL)

An enhanced version of Simulated Annealing that incorporates local search techniques to improve the quality of solutions.

### Tabu Search (TS)

An iterative search method that uses a tabu list to avoid cycles and encourages exploration of new areas in the solution space.

## Visualization

The script generates visualizations for each algorithm showing the best routes found for the couriers. Additionally, it produces plots comparing the performance of the algorithms in terms of objective function value over generations and execution time.

## Results

The results section will include:

1. Visualizations of the best routes for each algorithm.
2. A plot showing the objective function value over generations for each algorithm.
3. A comparison of execution times for the algorithms.
4. A comparison of the best objective function values obtained by each algorithm.
5. A plot showing the number of generations needed to achieve the best solution for each algorithm.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to contribute to this project by submitting issues or pull requests. For major changes, please open an issue first to discuss what you would like to change.

If you have any questions or suggestions, please contact the project maintainer.

Happy optimizing!
