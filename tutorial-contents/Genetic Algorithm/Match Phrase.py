# Adding rank and tournament selections by Choi, T
# Adding one- and two-point crossovers by Choi, T

"""
Visualize Genetic Algorithm to match the target phrase.
Visit my tutorial website for more: https://morvanzhou.github.io/tutorials/
"""

import numpy as np

TARGET_PHRASE = 'You get it!'       # target DNA
POP_SIZE = 300                      # population size
CROSS_RATE = 0.4                    # mating probability (DNA crossover)
MUTATION_RATE = 0.01                # mutation probability
N_GENERATIONS = 1000

DNA_SIZE = len(TARGET_PHRASE)
# convert string to number
TARGET_ASCII = np.fromstring(TARGET_PHRASE, dtype=np.uint8)
ASCII_BOUND = [32, 126]


class GA(object):
    def __init__(self, DNA_size, DNA_bound, cross_rate, mutation_rate, pop_size):
        self.DNA_size = DNA_size
        DNA_bound[1] += 1
        self.DNA_bound = DNA_bound
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size

        # int8 for convert to ASCII
        self.pop = np.random.randint(
            *DNA_bound, size=(pop_size, DNA_size)).astype(np.int8)

    def translateDNA(self, DNA):    # convert to readable string
        return DNA.tostring().decode('ascii')

    def get_fitness(self):  # count how many character matches
        match_count = (self.pop == TARGET_ASCII).sum(axis=1)
        return match_count

    def select(self):
        # add a small amount to avoid all zero fitness
        fitness = self.get_fitness() + 1e-4
        idx = np.random.choice(np.arange(
            self.pop_size), size=self.pop_size, replace=True, p=fitness/fitness.sum())
        return self.pop[idx]

    # Added by Choi, T for EA lectures
    def rank_select(self):
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

        # add a small amount to avoid all zero fitness
        fitness = self.get_fitness() + 1e-4
        rank_fitness = rankdata(fitness)
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True,
                               p=list(map(lambda x: x / sum(rank_fitness), rank_fitness)))
        return self.pop[idx]

    # Added by Choi, T for EA lectures
    def tournament_select(self, tournament_size=2):
        # add a small amount to avoid all zero fitness
        fitness = self.get_fitness() + 1e-4
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
            cross_points = [flag] * DNA_SIZE
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
                child[point] = np.random.randint(
                    *self.DNA_bound)    # choose a random ASCII index
        return child

    def evolve(self):
        pop = self.tournament_select()
        pop_copy = pop.copy()
        for parent in pop:  # for every parent
            child = self.two_point_crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop


if __name__ == '__main__':
    ga = GA(DNA_size=DNA_SIZE, DNA_bound=ASCII_BOUND, cross_rate=CROSS_RATE,
            mutation_rate=MUTATION_RATE, pop_size=POP_SIZE)

    for generation in range(N_GENERATIONS):
        fitness = ga.get_fitness()
        best_DNA = ga.pop[np.argmax(fitness)]
        best_phrase = ga.translateDNA(best_DNA)
        print('Gen', generation, ': ', best_phrase)
        if best_phrase == TARGET_PHRASE:
            break
        ga.evolve()
