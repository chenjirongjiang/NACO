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
        self.y_best: float = float("-inf")
        for iteration in range(self.max_iterations):
            # Generate a random bit string
            x: list[int] = [random.randint(0, 1) for _ in range(problem.meta_data.n_variables)]
            # Call the problem in order to get the y value    
            y: float = problem(x)
            # update the current state
            self.y_best = max(self.y_best, y)
            
            
class GeneticAlgorithm(Algorithm):
    '''A skeleton (minimal) implementation of your Genetic Algorithm.'''

    def __call__(self, problem: ioh.problem.Integer) -> None:
        self.problem = problem
        self.y_best: float = float("-inf")
        self.x_best: int = [random.randint(0, 1) for _ in range(problem.meta_data.n_variables)]
        self.evolution()

    def generate_population(self, n):
        population = []
        for i in range(n):
            population.append([random.randint(0, 1) for _ in range(self.problem.meta_data.n_variables)])
        return population

    def swap_mutation(self, candidate, pm):
        for i in range(len(candidate)):
            if random.random() < pm:
                candidate[i] = 1- candidate[i]

    #takes a candidate and returns its evaluation
    def fitness(self, candidate) -> float:
        return self.problem(candidate)
    
    #take a population and selects candidates based on chance 
    #depending on their fitness divided by the total fitness
    def roulette_selection(self, population) -> list:
        new_population = []
        total_fitness = 0
        for candidate in population:
            total_fitness += self.fitness(candidate)

        for candidate in population:
            pi = self.fitness(candidate)/ total_fitness
            if random.random() < pi:
                new_population.append(candidate)

        return new_population

    def evolution(self) -> None:
        n = 100 #initial population size
        pm = 1/ n #mutation probability
        
        population = self.generate_population(n)
        for iteration in range(self.max_iterations):
            for candidate in population:
                self.swap_mutation(candidate, pm )
            population= self.roulette_selection(population)
            for i in self.generate_population(n-len(population)):
                population.append(i)

        
        """for candidate in population:
            if self.fitness(candidate) > self.y_best:
                self.y_best = self.fitness(candidate)
                self.x_best = candidate"""

    
            
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
