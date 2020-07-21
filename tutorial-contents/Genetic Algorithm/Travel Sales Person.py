# Adding rank and tournament selections by Choi, T
# Adding one- and two-point crossovers by Choi, T

"""
Visualize Genetic Algorithm to find the shortest path for travel sales problem.
Visit my tutorial website for more: https://morvanzhou.github.io/tutorials/
"""

import matplotlib.pyplot as plt
import numpy as np

N_CITIES = 20   # DNA size
CROSS_RATE = 0.1
MUTATE_RATE = 0.02
POP_SIZE = 500
N_GENERATIONS = 500


class GA(object):
    def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size, ):
        self.DNA_size = DNA_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size

        self.pop = np.vstack([np.random.permutation(DNA_size)
                              for _ in range(pop_size)])

    def translateDNA(self, DNA, city_position):  # get cities' coord in order
        line_x = np.empty_like(DNA, dtype=np.float64)
        line_y = np.empty_like(DNA, dtype=np.float64)
        for i, d in enumerate(DNA):
            city_coord = city_position[d]
            line_x[i, :] = city_coord[:, 0]
            line_y[i, :] = city_coord[:, 1]
        return line_x, line_y

    def get_fitness(self, line_x, line_y):
        total_distance = np.empty((line_x.shape[0],), dtype=np.float64)
        for i, (xs, ys) in enumerate(zip(line_x, line_y)):
            total_distance[i] = np.sum(
                np.sqrt(np.square(np.diff(xs)) + np.square(np.diff(ys))))
        fitness = np.exp(self.DNA_size * 2 / total_distance)
        return fitness, total_distance

    def select(self, fitness):
        idx = np.random.choice(np.arange(
            self.pop_size), size=self.pop_size, replace=True, p=fitness / fitness.sum())
        return self.pop[idx]

    # Added by Choi, T for EA lectures
    def rank_select(self, fitness):
        # Efficient method to calculate the rank vector of a list in Python
        # https://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
        def rank_simple(vector):
            return sorted(range(len(vector)), key=vector.__getitem__)

        def rankdata(a):
            n = len(a)
            ivec = rank_simple(a)
            svec = [a[rank] for rank in ivec]
            sumranks = 0
            dupcount = 0
            newarray = [0]*n
            for i in range(n):
                sumranks += i
                dupcount += 1
                if i == n-1 or svec[i] != svec[i+1]:
                    averank = sumranks / float(dupcount) + 1
                    for j in range(i-dupcount+1, i+1):
                        newarray[ivec[j]] = averank
                    sumranks = 0
                    dupcount = 0
            return newarray

        rank_fitness = rankdata(fitness)
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True,
                               p=list(map(lambda x: x / sum(rank_fitness), rank_fitness)))
        return self.pop[idx]

    # Added by Choi, T for EA lectures
    def tournament_select(self, fitness, tournament_size=2):
        idx = []
        for _ in range(self.pop_size):
            participants = np.random.choice(
                np.arange(self.pop_size), size=tournament_size, replace=False)
            participants_fitness = list(np.array(fitness)[participants])
            winner = participants_fitness.index(max(participants_fitness))
            idx.append(participants[winner])
        return self.pop[idx]

    def crossover(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            # select another individual from pop
            i_ = np.random.randint(0, self.pop_size, size=1)
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(
                np.bool)    # choose crossover points
            # find the city number
            keep_city = parent[~cross_points]
            swap_city = pop[i_, np.isin(
                pop[i_].ravel(), keep_city, invert=True)]
            parent[:] = np.concatenate((keep_city, swap_city))
        return parent

    def one_point_crossover(self, parent, pop):
        if np.random.rand() < CROSS_RATE:
            # select another individual from pop
            i_ = np.random.randint(0, self.pop_size, size=1)
            j_ = np.random.randint(2, self.DNA_size - 2, size=1)
            flag = True if np.random.randint(0, 2) < 0.5 else False
            cross_points = [flag] * self.DNA_size
            cross_points[int(j_):] = [not flag] * len(cross_points[int(j_):])
            # find the city number
            keep_city = parent[~np.array(cross_points)]
            swap_city = pop[i_, np.isin(
                pop[i_].ravel(), keep_city, invert=True)]
            parent[:] = np.concatenate((keep_city, swap_city))
        return parent

    def two_point_crossover(self, parent, pop):
        if np.random.rand() < CROSS_RATE:
            # select another individual from pop
            i_ = np.random.randint(0, self.pop_size, size=1)
            j_ = np.sort(np.random.choice(
                np.arange(self.DNA_size) - 2, size=2, replace=False) + 1)
            flag = True if np.random.randint(0, 2) < 0.5 else False
            cross_points = [flag] * self.DNA_size
            cross_points[int(j_[0]):int(j_[1])] = [not flag] * \
                len(cross_points[int(j_[0]):int(j_[1])])
            # find the city number
            keep_city = parent[~np.array(cross_points)]
            swap_city = pop[i_, np.isin(
                pop[i_].ravel(), keep_city, invert=True)]
            parent[:] = np.concatenate((keep_city, swap_city))
        return parent

    def mutate(self, child):
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                swap_point = np.random.randint(0, self.DNA_size)
                swapA, swapB = child[point], child[swap_point]
                child[point], child[swap_point] = swapB, swapA
        return child

    def evolve(self, fitness):
        pop = self.tournament_select(fitness)
        pop_copy = pop.copy()
        for parent in pop:  # for every parent
            child = self.two_point_crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop


class TravelSalesPerson(object):
    def __init__(self, n_cities):
        self.city_position = np.random.rand(n_cities, 2)
        plt.ion()

    def plotting(self, lx, ly, total_d):
        plt.cla()
        plt.scatter(self.city_position[:, 0].T,
                    self.city_position[:, 1].T, s=100, c='k')
        plt.plot(lx.T, ly.T, 'r-')
        plt.text(-0.05, -0.05, "Total distance=%.2f" %
                 total_d, fontdict={'size': 20, 'color': 'red'})
        plt.xlim((-0.1, 1.1))
        plt.ylim((-0.1, 1.1))
        plt.pause(0.01)


ga = GA(DNA_size=N_CITIES, cross_rate=CROSS_RATE,
        mutation_rate=MUTATE_RATE, pop_size=POP_SIZE)

env = TravelSalesPerson(N_CITIES)
for generation in range(N_GENERATIONS):
    lx, ly = ga.translateDNA(ga.pop, env.city_position)
    fitness, total_distance = ga.get_fitness(lx, ly)
    ga.evolve(fitness)
    best_idx = np.argmax(fitness)
    print('Gen:', generation, '| best fit: %.2f' % fitness[best_idx],)

    env.plotting(lx[best_idx], ly[best_idx], total_distance[best_idx])

plt.ioff()
plt.show()
