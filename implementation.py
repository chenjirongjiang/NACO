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
        self.y_best: float = float("inf")
        self.x_best: int = [random.randint(0, 1) for _ in range(problem.meta_data.n_variables)]
        self.evolution()


    #flips a bit in a bitstring with chance pm
    def swap_mutation(self, candidate, pm):
        for i in range(len(candidate)):
            if random.random() < pm:
                candidate[i] = 1- candidate[i]
        return candidate

    def cross_over(self, candidate1, candidate2, pc):
        p1, p2 = candidate1, candidate2
        k = random.randrange(len(candidate1)-1)
        if random.random() <= pc:
            for i in range(k):
                pt = random.randrange(len(candidate1)-1)
                p1 = candidate1[:pt] + candidate2[pt:]
                p2 = candidate2[:pt] + candidate1[pt:]
        return [p1, p2]

    #takes a candidate and returns its evaluation
    def fitness(self, candidate) -> float:
        return self.problem(candidate)
    
    def tournement_selection(self, population, subset_size):
        winners = []
        for candidate in population:
            champion = population[random.randrange(len(population))]
            for j in range(subset_size -1):
                challenger = population[random.randrange(len(population))]
                if self.fitness(challenger) > self.fitness(champion):
                    champion = challenger
                winners.append(champion)
        return winners


    #take a population and selects candidates based on chance 
    #depending on their fitness divided by the total fitness
    def roulette_selection(self, population) -> list:
        new_population = []
        total_fitness = 0
        for candidate in population:
            total_fitness += self.fitness(candidate)

        for candidate in population:
            pi = self.fitness(candidate)/ total_fitness
            if random.random() <= pi:
                new_population.append(candidate)
        return new_population


    #implementing all functions to find best candidate
    def evolution(self) -> None:        
        n = 100 #initial population size
        pm = 1/ n #mutation probability
        pc = 1 #crossover probability

        #generate a random population size n
        population = []
        
        for iteration in range(self.max_iterations):
            for i in range(n - len(population)):
                population.append([random.randint(0, 1) for _ in range(self.problem.meta_data.n_variables)])
            #check for best solution
            for candidate in population:
                if self.fitness(candidate) > self.y_best:
                    self.y_best = self.fitness(candidate)
                    self.x_best = candidate

            #select parents
            selected = self.tournement_selection(population,2)
            next_gen = []
            
            if len(selected) % 2:
                next_gen.append(selected.pop())
            
            for i in range(0, len(selected), 2):
                c1, c2 = selected[i], selected[i+1]
                for c in self.cross_over(c1, c2, pc):
                    next_gen.append(self.swap_mutation(c, pm))
            population = next_gen
        
            
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
