# Adding rank and tournament selections by Choi, T
# Adding one- and two-point crossovers by Choi, T
# Adding sharing method by Choi, T

"""
Visualize Genetic Algorithm to find a path to the target.
Visit my tutorial website for more: https://morvanzhou.github.io/tutorials/
"""

import matplotlib.pyplot as plt
import numpy as np

N_MOVES = 450
DNA_SIZE = N_MOVES*2
DIRECTION_BOUND = [0, 1]
CROSS_RATE = 0.8
MUTATE_RATE = 0.003
POP_SIZE = 100
N_GENERATIONS = 100
GOAL_POINT = [10, 10]
START_POINT = [0, 0]
OBSTACLE_LINE = np.array([[1, 5], [9, 5]])


class GA(object):
    def __init__(self, DNA_size, DNA_bound, cross_rate, mutation_rate, pop_size, ):
        self.DNA_size = DNA_size
        DNA_bound[1] += 1
        self.DNA_bound = DNA_bound
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size

        self.pop = np.random.randint(*DNA_bound, size=(pop_size, DNA_size))

    # convert to readable string
    def DNA2product(self, DNA, n_moves, start_point):
        pop = (DNA - 0.5) / 4
        pop[:, 0], pop[:, n_moves] = start_point[0], start_point[1]
        lines_x = np.cumsum(pop[:, :n_moves], axis=1)
        lines_y = np.cumsum(pop[:, n_moves:], axis=1)
        return lines_x, lines_y

    def get_fitness(self, lines_x, lines_y, goal_point, obstacle_line):
        dist2goal = np.sqrt(
            (goal_point[0] - lines_x[:, -1]) ** 2 + (goal_point[1] - lines_y[:, -1]) ** 2)
        fitness = np.power(1 / (dist2goal + 1), 2)
        ##################################################
        points = (lines_x > obstacle_line[0, 0] -
                  0.25) & (lines_x < obstacle_line[1, 0] + 0.25)
        y_values = np.where(points, lines_y, np.zeros_like(lines_y) - 100)
        bad_lines = ((y_values > obstacle_line[0, 1]) & (
            y_values < obstacle_line[1, 1])).max(axis=1)
        fitness[bad_lines] = 1e-6
        ##################################################
        points = (lines_y > obstacle_line[0, 1] -
                  0.25) & (lines_y < obstacle_line[1, 1] + 0.25)
        x_values = np.where(points, lines_x, np.zeros_like(lines_x) - 100)
        bad_lines = ((x_values > obstacle_line[0, 0]) & (
            x_values < obstacle_line[1, 0])).max(axis=1)
        fitness[bad_lines] = 1e-6
        ##################################################
        return fitness

    # Added by Choi, T for EA lectures
    def get_sharing_fitness(self, lines_x, lines_y, goal_point, obstacle_line, min_dist=450):
        dist2goal = np.sqrt(
            (goal_point[0] - lines_x[:, -1]) ** 2 + (goal_point[1] - lines_y[:, -1]) ** 2)
        fitness = np.power(1 / (dist2goal + 1), 2)
        ##################################################
        points = (lines_x > obstacle_line[0, 0] -
                  0.25) & (lines_x < obstacle_line[1, 0] + 0.25)
        y_values = np.where(points, lines_y, np.zeros_like(lines_y) - 100)
        bad_lines = ((y_values > obstacle_line[0, 1]) & (
            y_values < obstacle_line[1, 1])).max(axis=1)
        fitness[bad_lines] = 1e-6
        ##################################################
        points = (lines_y > obstacle_line[0, 1] -
                  0.25) & (lines_y < obstacle_line[1, 1] + 0.25)
        x_values = np.where(points, lines_x, np.zeros_like(lines_x) - 100)
        bad_lines = ((x_values > obstacle_line[0, 0]) & (
            x_values < obstacle_line[1, 0])).max(axis=1)
        fitness[bad_lines] = 1e-6
        ##################################################

        for i in range(self.pop_size):
            if fitness[i] == 1e-6:
                continue

            denom = 1
            for j in range(self.pop_size):
                if fitness[j] == 1e-6:
                    continue

                dist = (self.pop[i] != self.pop[j]).sum()
                if dist < min_dist:
                    denom += 1 - dist / min_dist
            fitness[i] /= denom
        return fitness

    def select(self, fitness):
        idx = np.random.choice(np.arange(
            self.pop_size), size=self.pop_size, replace=True, p=fitness/fitness.sum())
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
            # mating and produce one child
            parent[cross_points] = pop[i_, cross_points]
        return parent

    def one_point_crossover(self, parent, pop):
        if np.random.rand() < CROSS_RATE:
            # select another individual from pop
            i_ = np.random.randint(0, self.pop_size, size=1)
            j_ = np.random.randint(1, self.DNA_size - 1, size=1)
            flag = True if np.random.randint(0, 2) < 0.5 else False
            cross_points = [flag] * self.DNA_size
            cross_points[int(j_):] = [not flag] * len(cross_points[int(j_):])
            # mating and produce one child
            parent[cross_points] = pop[i_, cross_points]
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
            # mating and produce one child
            parent[cross_points] = pop[i_, cross_points]
        return parent

    def mutate(self, child):
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                child[point] = np.random.randint(*self.DNA_bound)
        return child

    def evolve(self, fitness):
        pop = self.tournament_select(fitness)
        pop_copy = pop.copy()
        for parent in pop:  # for every parent
            child = self.two_point_crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop


class Line(object):
    def __init__(self, n_moves, goal_point, start_point, obstacle_line):
        self.n_moves = n_moves
        self.goal_point = goal_point
        self.start_point = start_point
        self.obstacle_line = obstacle_line

        plt.ion()

    def plotting(self, lines_x, lines_y):
        plt.cla()
        plt.scatter(*self.goal_point, s=200, c='r')
        plt.scatter(*self.start_point, s=100, c='b')
        plt.plot(self.obstacle_line[:, 0],
                 self.obstacle_line[:, 1], lw=3, c='k')
        plt.plot(lines_x.T, lines_y.T, c='k')
        plt.xlim((-5, 15))
        plt.ylim((-5, 15))
        plt.pause(0.01)


ga = GA(DNA_size=DNA_SIZE, DNA_bound=DIRECTION_BOUND,
        cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE)

env = Line(N_MOVES, GOAL_POINT, START_POINT, OBSTACLE_LINE)

for generation in range(N_GENERATIONS):
    lx, ly = ga.DNA2product(ga.pop, N_MOVES, START_POINT)
    # fitness = ga.get_fitness(lx, ly, GOAL_POINT, OBSTACLE_LINE)
    fitness = ga.get_sharing_fitness(lx, ly, GOAL_POINT, OBSTACLE_LINE)
    ga.evolve(fitness)
    print('Gen:', generation, '| best fit:', fitness.max())
    env.plotting(lx, ly)

plt.ioff()
plt.show()
