'''This file contains a skeleton implementation ofComputational Genomics Group at the Technical University of Munich the genetic algorithm
'''

from hashlib import new
from math import sqrt
import ioh
import random

from pandas.io.pytables import Selection
from algorithm import Algorithm

class RandomSearch(Algorithm):
    '''An example of Random Search.'''

    def __call__(self, problem: ioh.problem.Integer) -> None:
        self.y_best: float = float("inf")
        for iteration in range(self.max_iterations):
            # Generate a random bit string
            x: list[int] = [random.randint(0, 1) for _ in range(problem.meta_data.n_variables)]
            
            # Call the problem in order to get the y value    
            y: float = problem(x)
            # update the current state
            self.y_best = max(self.y_best, y)
            
            
class GeneticAlgorithm(Algorithm):
    '''An implementation of the Genetic Algorithm.'''
    def __init__(self, pop: int, max_iterations: int, mutation: int, recombination: int, selection: int):
        super().__init__(max_iterations=max_iterations)
        self.pop = pop #initial population size
        self.mutation = mutation
        self.recombination = recombination
        self.selection = selection
        self.pm = 0.01 #mutation probability
        self.pc = 0.7 #crossover probability
        self.s = 5 #subset for tournement selection
        self.k = 2 #binary/ discrete domain
    
    def __call__(self, problem: ioh.problem.Integer) -> None:
        self.problem = problem
        pop = self.pop
        pm = self.pm
        pc = self.pc
        s = self.s
        k = self.k
        optimal = False

        if self.selection == 1:
            selection = self.tournement_selection
        else:
            selection = self.roulette_selection

        if self.recombination == 1:
            recombination = self.one_point_crossover
        else:
            recombination = self.uniform_crossover

        if self.mutation == 1:
            if k==2:
                mutation = self.bit_mutation
            elif k==3:
                mutation = self.int_mutation
        else:
            if k==2:
                mutation = self.flipbit_mutation
            else:
                mutation = self.reverseint_mutation
        
        population = self.initiation(k, pop)

        #check for best solution
        for iteration in range(self.max_iterations): 
            scores = [problem(candidate) for candidate in population]
            for i in range(len(population)):
                if scores[i] == 100.0:
                    optimal = True
                    break
            if optimal:
                return iteration

            #select parents
            selected = selection(population, s, scores)

            if len(selected) %2:
                selected.pop()
            next_gen = []
            
            for i in range(0, len(selected), 2):
                c1, c2 = selected[i], selected[i+1]
                children = recombination(c1, c2, pc) #recombination
                for child in children:
                    next_gen.append(mutation(child, pm)) #mutation
            population = next_gen
        return "fail"
        
    def initiation(self, k: int, pop: int) -> list[list[int]]:
        '''Make random popultion with pop amount of individuals.'''
        population =[[random.randint(0, k-1) for _ in range(self.problem.meta_data.n_variables)] for _ in range(pop)]
        return population

    def int_mutation(self, candidate: list[int], pm: float) -> list[int]:
        '''Every element has Pm chance to be changed in one of the other integers.'''
        child = candidate
        for i in range(len(candidate)):
            if random.random() <= pm:
                option = [0,1,2]
                option.remove(candidate[i])
                child[i] = random.choice(option)
        return child

    def reverseint_mutation(self, candidate: list[int], pm: float) -> list[int]:
        '''Pm chance for a random sequence to be reversed.'''
        child = candidate
        if random.random()<= pm*60:
            i = random.randint(0,len(child)-1)
            j = random.randint(i, len(child))
            a = child[i:j]
            a.reverse()
            child[i:j] = a
        return child

    def bit_mutation(self, candidate: list[int], pm: float) -> list[int]:
        '''Every bit has Pm chance to be flipped.'''
        child = candidate
        for i in range(len(candidate)):
            if random.random() <= pm:
                child[i] = 1- candidate[i]
        return child

    def flipbit_mutation(self, candidate: list[int], pm: float) -> list[int]:
        '''Pm chance for every bit to be flipped.'''
        child = candidate
        p1 = random.randint(0, len(candidate))
        p2 = random.randint(p1, len(candidate))
        if random.random() <= pm*60:
            for i in range(p1,p2):
                child[i] = 1 - candidate[i]
        return child
                
    def one_point_crossover(self, candidate1: list[int], candidate2: list[int], pc: float) -> list[int]:
        '''Random point k is chosen and crossover is done over that point'''
        p1, p2 = candidate1[:], candidate2[:]
        child1, child2 = candidate1[:], candidate2[:]
        k = 1
        if random.random() <= pc:
            for i in range(k):
                pt = random.randrange(1, len(candidate1)-2)
                child1 = p1[:pt] + p2[pt:]
                child2 = p2[:pt] + p1[pt:]
                p1 = child1
                p2 = child2
        return [child1, child2]

    def uniform_crossover(self, candidate1: list[int], candidate2: list[int], pc: float) -> list[int]:
        '''Every bit is chosen from either parent with equal chance.'''
        child1, child2 = candidate1[:], candidate2[:]
        if random.random() <= pc:
            for i in range(len(candidate1)):
                if random.random() <= 0.5:
                    child1[i] = candidate1[i]
                    child2[i] = candidate2[i]
                else:
                    child1[i] = candidate2[i]
                    child2[i] = candidate1[i]
        return [child1, child2]
    
    def tournement_selection(self, population: int, subset_size: int, scores: list[float]) -> list[list[int]]:
        '''Every bit is chosen from either parent with equal chance.'''
        winners = []
        for i in range(len(population)):
            p = random.randrange(len(population))
            champion = p
            for j in range(subset_size -1):
                q = random.randrange(len(population))
                challenger = q
                if scores[challenger] > scores[champion]:
                    champion = challenger
            winners.append(population[champion])
        return winners

    def roulette_selection(self, population: int, subset_size: int, scores: list[float]) -> list[list[int]]:
        '''Take a population and selects candidates based on chance depending on their fitness.'''
        new_population = []
        square = 15
        scores_squares = [pow(x,square) for x in scores]
        while len(new_population) < len(population):
            i = random.randrange(len(population))
            pi = scores_squares[i]/ sum(scores_squares)
            if random.random() <= pi:
                new_population.append(population[i])
        return new_population
        

def main():
    # Set a random seed in order to get reproducible results
    random.seed(42)

    #algorithm = RandomSearch()
    algorithm = GeneticAlgorithm(pop=100,max_iterations=100,mutation=0,recombination=0,selection=0)
    
    # Get a problem from the IOHexperimenter environment
    problem: ioh.problem.Integer = ioh.get_problem(2, 1, 100, 'Integer')

    # Run the algoritm on the problem
    print(algorithm(problem))

    # Inspect the results
    print("Best solution found:")
    print("".join(map(str, problem.state.current_best.x)))
    print("With an objective value of:", problem.state.current_best.y)
    print()


if __name__ == '__main__':
    main()
