# Adding rank and tournament selections by Choi, T
# Adding one- and two-point crossovers by Choi, T
# Adding sharing method by Choi, T

"""
Visualize Genetic Algorithm to find a maximum point in a function.
Visit my tutorial website for more: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 10           # DNA length
POP_SIZE = 100          # population size
CROSS_RATE = 0.8        # mating probability (DNA crossover)
MUTATION_RATE = 0.003   # mutation probability
N_GENERATIONS = 200
X_BOUND = [0, 10]       # x upper and lower bounds


def F(x):
    return np.sin(10*x)*x + np.cos(2*x) * \
        x   # to find the maximum of this function


# find non-zero fitness for selection
def get_fitness(pred):
    return pred + 1e-3 - np.min(pred)


# Added by Choi, T for EA lectures
def get_sharing_fitness(pop, pred, min_dist=1.5):
    fitness = pred + 1e-3 - np.min(pred)
    for i in range(POP_SIZE):
        denom = 1
        for j in range(POP_SIZE):
            dist = (pop[i] != pop[j]).sum()
            if dist < min_dist:
                denom += 1 - dist / min_dist
        fitness[i] /= denom
    return fitness


# convert binary DNA to decimal and normalize it to a range(0, 5)
def translateDNA(pop):
    return pop.dot(2 ** np.arange(DNA_SIZE)
                   [::-1]) / float(2**DNA_SIZE-1) * X_BOUND[1]


def select(pop, fitness):   # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness/fitness.sum())
    return pop[idx]


# Added by Choi, T for EA lectures
def rank_select(pop, fitness):
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
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=list(map(lambda x: x / sum(rank_fitness), rank_fitness)))
    return pop[idx]


# Added by Choi, T for EA lectures
def tournament_select(pop, fitness, tournament_size=2):
    idx = []
    for _ in range(POP_SIZE):
        participants = np.random.choice(
            np.arange(POP_SIZE), size=tournament_size, replace=False)
        participants_fitness = list(np.array(fitness)[participants])
        winner = participants_fitness.index(max(participants_fitness))
        idx.append(participants[winner])
    return pop[idx]


def crossover(parent, pop):  # mating process (genes crossover)
    if np.random.rand() < CROSS_RATE:
        # select another individual from pop
        i_ = np.random.randint(0, POP_SIZE, size=1)
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(
            np.bool)    # choose crossover points
        # mating and produce one child
        parent[cross_points] = pop[i_, cross_points]
    return parent


def one_point_crossover(parent, pop):
    if np.random.rand() < CROSS_RATE:
        # select another individual from pop
        i_ = np.random.randint(0, POP_SIZE, size=1)
        j_ = np.random.randint(1, DNA_SIZE - 1, size=1)
        flag = True if np.random.randint(0, 2) < 0.5 else False
        cross_points = [flag] * DNA_SIZE
        cross_points[int(j_):] = [not flag] * len(cross_points[int(j_):])
        # mating and produce one child
        parent[cross_points] = pop[i_, cross_points]
    return parent


def two_point_crossover(parent, pop):
    if np.random.rand() < CROSS_RATE:
        # select another individual from pop
        i_ = np.random.randint(0, POP_SIZE, size=1)
        j_ = np.sort(np.random.choice(
            np.arange(DNA_SIZE) - 2, size=2, replace=False) + 1)
        flag = True if np.random.randint(0, 2) < 0.5 else False
        cross_points = [flag] * DNA_SIZE
        cross_points[int(j_[0]):int(j_[1])] = [not flag] * \
            len(cross_points[int(j_[0]):int(j_[1])])
        # mating and produce one child
        parent[cross_points] = pop[i_, cross_points]
    return parent


def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child


pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE)
                        )   # initialize the pop DNA

plt.ion()   # something about plotting
x = np.linspace(*X_BOUND, 200)
plt.plot(x, F(x))

for _ in range(N_GENERATIONS):
    # compute function value by extracting DNA
    F_values = F(translateDNA(pop))

    # something about plotting
    if 'sca' in globals():
        sca.remove()
    sca = plt.scatter(translateDNA(pop), F_values,
                      s=200, lw=0, c='red', alpha=0.5)
    plt.pause(0.05)

    # GA part (evolution)
    # fitness = get_fitness(F_values)
    fitness = get_sharing_fitness(pop, F_values)
    print("Most fitted DNA: ", pop[np.argmax(fitness), :])
    pop = tournament_select(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        child = two_point_crossover(parent, pop_copy)
        child = mutate(child)
        parent[:] = child   # parent is replaced by its child

plt.ioff()
plt.show()
