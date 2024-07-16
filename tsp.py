import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import time
from collections import deque

# Parametry problemu
NUM_COURIERS = 3
NUM_PACKAGES = 20
CITY_SIZE = 100
VELOCITIES = [1, 0.8, 1.2]  # Prędkości kurierów
BASE_LOCATION = np.array([50, 50])  # Lokalizacja bazy
PACKAGE_CAPACITY = 7  # Maksymalna liczba paczek na kuriera
TIME_WINDOW_PENALTY = 100  # Kara za naruszenie okien czasowych

# Generowanie losowych współrzędnych paczek
np.random.seed(42)
packages = np.random.randint(0, CITY_SIZE, size=(NUM_PACKAGES, 2))

# Generowanie losowych okien czasowych
time_windows = np.random.randint(0, CITY_SIZE // 2, size=(NUM_PACKAGES, 2))
time_windows = np.sort(time_windows, axis=1)

# Tworzenie klas fitness i osobników
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("indices", np.random.permutation, NUM_PACKAGES)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Funkcja oceny (funkcja celu)
def evaluate(individual):
    total_distance = 0
    penalty = 0
    for courier in range(NUM_COURIERS):
        courier_packages = individual[courier::NUM_COURIERS]
        if len(courier_packages) > PACKAGE_CAPACITY:
            penalty += (len(courier_packages) - PACKAGE_CAPACITY) * TIME_WINDOW_PENALTY

        if len(courier_packages) < 1:
            continue
        # Start od bazy
        prev_location = BASE_LOCATION
        time = 0
        for i in range(len(courier_packages)):
            p = packages[courier_packages[i]]
            time += np.linalg.norm(prev_location - p) / VELOCITIES[courier]
            if time < time_windows[courier_packages[i]][0] or time > time_windows[courier_packages[i]][1]:
                penalty += TIME_WINDOW_PENALTY
            total_distance += np.linalg.norm(prev_location - p) / VELOCITIES[courier]
            prev_location = p
        # Powrót do bazy
        total_distance += np.linalg.norm(prev_location - BASE_LOCATION) / VELOCITIES[courier]
    return total_distance + penalty,

# Rejestracja funkcji w toolboxie
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Funkcja główna dla GA
def run_ga():
    population = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=1000, 
                                               stats=stats, halloffame=hof, verbose=False)

    return population, logbook, hof

# Implementacja SA
def sa(max_iterations=1000):
    def neighbor(individual):
        new_ind = individual[:]
        i, j = np.random.randint(0, len(individual), 2)
        new_ind[i], new_ind[j] = new_ind[j], new_ind[i]
        return new_ind

    initial_solution = toolbox.individual()
    best = current = initial_solution
    best_score = current_score = evaluate(current)[0]
    temp = 1.0
    cooling_rate = 0.003

    logbook = tools.Logbook()
    logbook.header = ["gen", "avg", "std", "min", "max"]
    scores = []

    gen = 0
    while temp > 1e-3 and gen < max_iterations:
        new_solution = neighbor(current)
        new_score = evaluate(new_solution)[0]

        if new_score < current_score or np.random.rand() < np.exp((current_score - new_score) / temp):
            current = new_solution
            current_score = new_score

        if new_score < best_score:
            best = new_solution
            best_score = new_score

        scores.append(current_score)
        logbook.record(gen=gen, avg=np.mean(scores), std=np.std(scores),
                       min=np.min(scores), max=np.max(scores))
        temp *= 1 - cooling_rate
        gen += 1

    return best, logbook, best

# Implementacja algorytmu wyżarzania symulowanego z przybliżonym przeszukiwaniem lokalnym (SAL)
def sal(max_iterations=1000):
    def neighbor(individual):
        new_ind = individual[:]
        i, j = np.random.randint(0, len(individual), 2)
        new_ind[i], new_ind[j] = new_ind[j], new_ind[i]
        return new_ind

    initial_solution = toolbox.individual()
    best = current = initial_solution
    best_score = current_score = evaluate(current)[0]
    temp = 1.0
    cooling_rate = 0.003

    logbook = tools.Logbook()
    logbook.header = ["gen", "avg", "std", "min", "max"]
    scores = []

    gen = 0
    while temp > 1e-3 and gen < max_iterations:
        new_solution = neighbor(current)
        new_score = evaluate(new_solution)[0]

        if new_score < current_score or np.random.rand() < np.exp((current_score - new_score) / temp):
            current = new_solution
            current_score = new_score

        if new_score < best_score:
            best = new_solution
            best_score = new_score

        scores.append(current_score)
        logbook.record(gen=gen, avg=np.mean(scores), std=np.std(scores),
                       min=np.min(scores), max=np.max(scores))
        temp *= 1 - cooling_rate
        gen += 1

    return best, logbook, best

# Implementacja Tabu Search (TS)
def tabu_search(max_iterations=1000, tabu_tenure=10):
    def neighbor(individual):
        new_ind = individual[:]
        i, j = np.random.randint(0, len(individual), 2)
        new_ind[i], new_ind[j] = new_ind[j], new_ind[i]
        return new_ind

    initial_solution = toolbox.individual()
    best = current = initial_solution
    best_score = current_score = evaluate(current)[0]

    tabu_list = deque(maxlen=tabu_tenure)
    logbook = tools.Logbook()
    logbook.header = ["gen", "avg", "std", "min", "max"]
    scores = []

    for gen in range(max_iterations):
        neighborhood = [neighbor(current) for _ in range(100)]
        neighborhood_scores = [evaluate(ind)[0] for ind in neighborhood]

        for i in range(len(neighborhood)):
            if neighborhood[i] in tabu_list:
                neighborhood_scores[i] += TIME_WINDOW_PENALTY

        best_neighbor = neighborhood[np.argmin(neighborhood_scores)]
        best_neighbor_score = min(neighborhood_scores)

        if best_neighbor_score < best_score:
            best = best_neighbor
            best_score = best_neighbor_score

        current = best_neighbor
        current_score = best_neighbor_score

        tabu_list.append(current)
        scores.append(current_score)
        logbook.record(gen=gen, avg=np.mean(scores), std=np.std(scores),
                       min=np.min(scores), max=np.max(scores))

    return best, logbook, best

if __name__ == "__main__":
    # Uruchomienie GA
    start_time = time.time()
    pop_ga, log_ga, hof_ga = run_ga()
    ga_time = time.time() - start_time

    # Uruchomienie SA
    start_time = time.time()
    best_sa, log_sa, hof_sa = sa()
    sa_time = time.time() - start_time

    # Uruchomienie SAL
    start_time = time.time()
    best_sal, log_sal, hof_sal = sal()
    sal_time = time.time() - start_time

    # Uruchomienie TS
    start_time = time.time()
    best_ts, log_ts, hof_ts = tabu_search()
    ts_time = time.time() - start_time

    # Wizualizacja najlepszej trasy dla GA
    best_ind_ga = hof_ga[0]
    plt.figure(figsize=(10, 6))
    for courier in range(NUM_COURIERS):
        courier_packages = best_ind_ga[courier::NUM_COURIERS]
        courier_route = np.vstack((BASE_LOCATION, packages[courier_packages], BASE_LOCATION))
        plt.plot(courier_route[:, 0], courier_route[:, 1], marker='o', label=f'Kurier {courier + 1}')
    plt.scatter(packages[:, 0], packages[:, 1], c='red')
    plt.scatter(BASE_LOCATION[0], BASE_LOCATION[1], c='blue', marker='s', s=100, label='Baza')
    plt.legend()
    plt.xlabel('Współrzędna X')
    plt.ylabel('Współrzędna Y')
    plt.title('Najlepsze trasy kurierów (GA)')
    plt.show()

    # Wizualizacja najlepszej trasy dla SA
    best_ind_sa = best_sa
    plt.figure(figsize=(10, 6))
    for courier in range(NUM_COURIERS):
        courier_packages = best_ind_sa[courier::NUM_COURIERS]
        courier_route = np.vstack((BASE_LOCATION, packages[courier_packages], BASE_LOCATION))
        plt.plot(courier_route[:, 0], courier_route[:, 1], marker='o', label=f'Kurier {courier + 1}')
    plt.scatter(packages[:, 0], packages[:, 1], c='red')
    plt.scatter(BASE_LOCATION[0], BASE_LOCATION[1], c='blue', marker='s', s=100, label='Baza')
    plt.legend()
    plt.xlabel('Współrzędna X')
    plt.ylabel('Współrzędna Y')
    plt.title('Najlepsze trasy kurierów (SA)')
    plt.show()

    # Wizualizacja najlepszej trasy dla SAL
    best_ind_sal = best_sal
    plt.figure(figsize=(10, 6))
    for courier in range(NUM_COURIERS):
        courier_packages = best_ind_sal[courier::NUM_COURIERS]
        courier_route = np.vstack((BASE_LOCATION, packages[courier_packages], BASE_LOCATION))
        plt.plot(courier_route[:, 0], courier_route[:, 1], marker='o', label=f'Kurier {courier + 1}')
    plt.scatter(packages[:, 0], packages[:, 1], c='red')
    plt.scatter(BASE_LOCATION[0], BASE_LOCATION[1], c='blue', marker='s', s=100, label='Baza')
    plt.legend()
    plt.xlabel('Współrzędna X')
    plt.ylabel('Współrzędna Y')
    plt.title('Najlepsze trasy kurierów (SAL)')
    plt.show()

    # Wizualizacja najlepszej trasy dla TS
    best_ind_ts = best_ts
    plt.figure(figsize=(10, 6))
    for courier in range(NUM_COURIERS):
        courier_packages = best_ind_ts[courier::NUM_COURIERS]
        courier_route = np.vstack((BASE_LOCATION, packages[courier_packages], BASE_LOCATION))
        plt.plot(courier_route[:, 0], courier_route[:, 1], marker='o', label=f'Kurier {courier + 1}')
    plt.scatter(packages[:, 0], packages[:, 1], c='red')
    plt.scatter(BASE_LOCATION[0], BASE_LOCATION[1], c='blue', marker='s', s=100, label='Baza')
    plt.legend()
    plt.xlabel('Współrzędna X')
    plt.ylabel('Współrzędna Y')
    plt.title('Najlepsze trasy kurierów (TS)')
    plt.show()

    # Analiza statystyk dla GA
    gen_ga = log_ga.select("gen")
    avg_ga = log_ga.select("avg")
    std_ga = log_ga.select("std")
    min_ga = log_ga.select("min")
    max_ga = log_ga.select("max")

    # Analiza statystyk dla SA
    gen_sa = log_sa.select("gen")
    avg_sa = log_sa.select("avg")
    std_sa = log_sa.select("std")
    min_sa = log_sa.select("min")
    max_sa = log_sa.select("max")

    # Analiza statystyk dla SAL
    gen_sal = log_sal.select("gen")
    avg_sal = log_sal.select("avg")
    std_sal = log_sal.select("std")
    min_sal = log_sal.select("min")
    max_sal = log_sal.select("max")

    # Analiza statystyk dla TS
    gen_ts = log_ts.select("gen")
    avg_ts = log_ts.select("avg")
    std_ts = log_ts.select("std")
    min_ts = log_ts.select("min")
    max_ts = log_ts.select("max")

    # Wykres z przebiegami wartości funkcji celu dla GA, SA, SAL i TS
    plt.figure(figsize=(12, 6))
    plt.plot(gen_ga, avg_ga, label='GA - Średnia wartość funkcji celu')
    plt.fill_between(gen_ga, np.array(avg_ga) - np.array(std_ga), np.array(avg_ga) + np.array(std_ga), alpha=0.2)

    plt.plot(gen_sa, avg_sa, label='SA - Średnia wartość funkcji celu')
    plt.fill_between(gen_sa, np.array(avg_sa) - np.array(std_sa), np.array(avg_sa) + np.array(std_sa), alpha=0.2)

    plt.plot(gen_sal, avg_sal, label='SAL - Średnia wartość funkcji celu')
    plt.fill_between(gen_sal, np.array(avg_sal) - np.array(std_sal), np.array(avg_sal) + np.array(std_sal), alpha=0.2)

    plt.plot(gen_ts, avg_ts, label='TS - Średnia wartość funkcji celu')
    plt.fill_between(gen_ts, np.array(avg_ts) - np.array(std_ts), np.array(avg_ts) + np.array(std_ts), alpha=0.2)

    plt.xlabel('Generacja')
    plt.ylabel('Wartość funkcji celu')
    plt.title('Zmiana wartości funkcji celu w zależności od generacji dla GA, SA, SAL i TS')
    plt.legend()
    plt.show()

    # Porównanie czasów wykonania
    times = [ga_time, sa_time, sal_time, ts_time]
    labels = ['GA', 'SA', 'SAL', 'TS']
    plt.figure(figsize=(10, 6))
    plt.bar(labels, times, color=['blue', 'green', 'red', 'purple'])
    plt.xlabel('Algorytm')
    plt.ylabel('Czas wykonania (s)')
    plt.title('Porównanie czasów wykonania algorytmów')
    plt.show()

    # Porównanie najlepszych wartości funkcji celu
    best_scores = [min_ga[-1], min_sa[-1], min_sal[-1], min_ts[-1]]
    plt.figure(figsize=(10, 6))
    plt.bar(labels, best_scores, color=['blue', 'green', 'red', 'purple'])
    plt.xlabel('Algorytm')
    plt.ylabel('Najlepsza wartość funkcji celu')
    plt.title('Porównanie najlepszych wartości funkcji celu algorytmów')
    plt.show()

    # Średnia liczba iteracji do osiągnięcia najlepszego rozwiązania
    best_gen_ga = np.argmin(log_ga.select("min"))
    best_gen_sa = np.argmin(log_sa.select("min"))
    best_gen_sal = np.argmin(log_sal.select("min"))
    best_gen_ts = np.argmin(log_ts.select("min"))

    best_gens = [best_gen_ga, best_gen_sa, best_gen_sal, best_gen_ts]
    plt.figure(figsize=(10, 6))
    plt.bar(labels, best_gens, color=['blue', 'green', 'red', 'purple'])
    plt.xlabel('Algorytm')
    plt.ylabel('Generacja')
    plt.title('Średnia liczba iteracji do osiągnięcia najlepszego rozwiązania')
    plt.show()
