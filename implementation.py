'''This file contains a skeleton implementation for the practical assignment 
for the NACO 21/22 course. 

Your Genetic Algorithm should be callable like so:
    >>> problem = ioh.get_problem(...)
    >>> ga = GeneticAlgorithm(...)
    >>> ga(problem)

In order to ensure this, please inherit from the provided Algorithm interface,
in a similar fashion as the RandomSearch Example:
    >>> class GeneticAlgorithm(Algorithm):
    >>>     ...

Please be sure to use don't change this name (GeneticAlgorithm) for your implementation.

If you override the constructor in your algoritm code, please be sure to 
call super().__init__, in order to correctly setup the interface. 

Only use keyword arguments for your __init__ method. This is a requirement for the
test script to properly evaluate your algoritm.
'''

from hashlib import new
from math import sqrt
import ioh
import random
from algorithm import Algorithm

class RandomSearch(Algorithm):
    '''An example of Random Search.'''

    def __call__(self, problem: ioh.problem.Integer) -> None:
        self.y_best: float = float("inf")
        for iteration in range(self.max_iterations):
            # Generate a random bit string
            x: list[int] = [random.randint(0, 1) for _ in range(problem.meta_data.n_variables)]
            print(x)
            # Call the problem in order to get the y value    
            y: float = problem(x)
            # update the current state
            self.y_best = max(self.y_best, y)
            
            
class GeneticAlgorithm(Algorithm):
    '''A skeleton (minimal) implementation of your Genetic Algorithm.'''
    def __call__(self, problem: ioh.problem.Integer) -> None:
        self.problem = problem
        pop = 100 #initial population size
        pm = 0.008 #mutation probability
        pc = 0.7 #crossover probability
        s = 2 #subset for tournement selection
        k = 2 #binary/ discrete domain

        population = self.initiation(k, pop)
            
        for iteration in range(self.max_iterations):
            #check for best solution
            for candidate in population:
                print( problem(candidate))

            #select parents
            selected = self.tournement_selection(population, s)

            if len(selected) %2:
                selected.pop()
            next_gen = []
            
            for i in range(0, len(selected), 2):
                c1, c2 = selected[i], selected[i+1]
                children = self.two_point_crossover(c1, c2, pc)
                for child in children:
                    next_gen.append(self.bit_mutation(child, pm))
            population = next_gen

    #prduces initial population
    def initiation(self, k, pop):
        population =[]
        for i in range(pop):
            population.append([random.randint(0, k) for _ in range(self.problem.meta_data.n_variables)])
        return population

    #every int has pm chance to be changed in one of the other integers
    def int_mutation(self, candidate, pm):
        child = candidate
        for i in range(len(candidate)):
            if random.random() <- pm:
                option = [0,1,2]
                option.remove(candidate[i])
                child[i] = random.choice(option)
        return child

    #pm chance for every int to be changed in one of the other integers
    def flipint_mutation(self, candidate, pm):
        child = candidate
        if random.random() <= pm:
            for i in range(len(candidate)):
                option = [0,1,2]
                option.remove(candidate[i])
                child[i] = random.choice(option)
        return child

    #every bit has pm chance to be flipped
    def bit_mutation(self, candidate, pm):
        child = candidate
        for i in range(len(candidate)):
            if random.random() <= pm:
                child[i] = 1- candidate[i]
        return child

    #pm chance for every bit to be flipped
    def flipbit_mutation(self, candidate, pm):
        child = candidate
        if random.random() < pm:
            for i in range(len(candidate)):
                child[i] = 1 - candidate[i]
        return child
                
    #random point k is chosen and crossover is done over that point
    def two_point_crossover(self, candidate1, candidate2, pc):
        p1, p2 = candidate1, candidate2
        child1, child2 = candidate1, candidate2
        k = 2
        if random.random() <= pc:
            for i in range(k):
                pt = random.randrange(1, len(candidate1)-2)
                child1 = p1[:pt] + p2[pt:]
                child2 = p2[:pt] + p1[pt:]
                p1 = child1
                p2 = child2
        return [child1, child2]

    #every bit is chosen from either parent with equal chance
    def uniform_crossover(self, candidate1, candidate2, pc):
        child1, child2 = candidate1, candidate2
        if random.random() <= pc:
            for i in range(len(candidate1)):
                if random.random() <= 0.5:
                    child1[i] = candidate1[i]
                    child2[i] = candidate2[i]
                else:
                    child1[i] = candidate2[i]
                    child2[i] = candidate1[i]
        return [child1, child2]
    
    #selects the best candidates from a small subset of the population to fill a new population
    def tournement_selection(self, population, subset_size):
        winners = []
        for i in range(len(population)):
            p = random.randrange(len(population))
            champion = population[p]
            for j in range(subset_size -1):
                q = random.randrange(len(population))
                challenger = population[q]
                if self.problem(challenger) > self.problem(champion):
                    champion = challenger
            winners.append(champion)
        return winners

    #take a population and selects candidates based on chance 
    #depending on their fitness divided by the total fitness
    def roulette_selection(self, population) -> list:
        new_population = []
        total_fitness = 0
        for candidate in population:
            total_fitness += self.problem(candidate)
        
        while len(new_population) < len(population):
            for candidate in population:
                pi = self.problem(candidate)/ total_fitness
                if random.random() <= pi*5:
                    new_population.append(candidate)
        return new_population
        
            
def main():
    # Set a random seed in order to get reproducible results
    random.seed(42)

    # Instantiate the algoritm, you should replace this with your GA implementation 
    algorithm = GeneticAlgorithm()

    # Get a problem from the IOHexperimenter environment
    problem: ioh.problem.Integer = ioh.get_problem(1, 1, 5, "Integer")

    # Run the algoritm on the problem
    algorithm(problem)

    # Inspect the results
    print("Best solution found:")
    print("".join(map(str, problem.state.current_best.x)))
    print("With an objective value of:", problem.state.current_best.y)
    print()


if __name__ == '__main__':
    main()
