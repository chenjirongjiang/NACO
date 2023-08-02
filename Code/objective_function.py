import typing
import ioh
import pandas as pd
from algorithm import Algorithm
from implementation import GeneticAlgorithm, RandomSearch
import ast


class CellularAutomata:
    '''Cellular Automata instance with the following parameters:
        k: Domain size of cell values
        rule_number: Transition rule corresponding to the Wolfram code
        t: Amount of time steps
        c0: State of CA at timestep 0
        '''
    def __init__(self, k: int, rule_number: int, t: int, c0: typing.List[int]):
        self.k = k
        self.rule_number = rule_number
        self.t = t
        self.c0 = c0
        self.radius = 1
        
    
    def __call__(self):
        '''Evaluate for T timesteps. Return Ct for a given C0.'''
        state = self.c0
        rule_states = self.rule_list()
        for i in range(self.t):
            state = self.transition(state, self.k, rule_states )
        return state


    def neighbours(self, c0: list[int], cell_position: int) -> list:
        '''Determine neighbourhood of a cell in c0.'''
        neighbourhood = []
        for r in range(self.radius+1):
            if cell_position + r >= len(c0):
                neighbourhood.insert(0, c0[cell_position - r])
                neighbourhood.append(0)
            elif cell_position - r < 0:
                neighbourhood.insert(0, 0)
                neighbourhood.append(c0[cell_position + r])
            elif r == 0:
                neighbourhood.append(c0[cell_position])
            else:
                neighbourhood.insert(0, c0[cell_position - r])
                neighbourhood.append(c0[cell_position + r])
        return neighbourhood

    def rule_list(self) -> list[int]:
        '''Make a binary or ternary list of the transition rule.'''
        if self.k ==2:
            binary_rule = format(self.rule_number, "b")
            binary_list = [int(i) for i in binary_rule]
            for i in range(8-len(binary_list)):
                binary_list.insert(0, 0)
            return binary_list
        else:
            nums = []
            while self.rule_number:
                self.rule_number, r = divmod(self.rule_number, 3)
                nums.append(r)
            
            while len(nums) < 27:
                nums.append(0)
            nums.reverse()
            return nums
            
        

    def transition(self, c0: list[int], k: int, rule_states: list[int]) -> list[int]:
        '''Determine new state by searching in rule states with the index corresponding with the neighbourhood.'''
        if k == 2:
            states = [[1,1,1], [1,1,0], [1,0,1], [1,0,0], [0,1,1], [0,1,0], [0,0,1], [0,0,0]]
        elif k == 3:
            states = [[2, 2, 2], [2, 2, 1], [2, 2, 0], [2, 1, 2], [2, 1, 1], [2, 1, 0], [2, 0, 2], [2, 0, 1], [2, 0, 0], 
                      [1, 2, 2], [1, 2, 1], [1, 2, 0], [1, 1, 2], [1, 1, 1], [1, 1, 0], [1, 0, 2], [1, 0, 1], [1, 0, 0], 
                      [0, 2, 2], [0, 2, 1], [0, 2, 0], [0, 1, 2], [0, 1, 1], [0, 1, 0], [0, 0, 2], [0, 0, 1], [0, 0, 0]]

        rule_states = rule_states
        new_state = []
        for i in range(len(c0)):
            neighbourhood = self.neighbours(c0, i)
            index_state = states.index(neighbourhood)
            new_state.append(rule_states[index_state])

        return new_state



def mkfunc1(k, rule, t, ct):
    def objective_function_1(c0_prime: typing.List[int]) -> float:
        '''Objective function that compares the similarity between given end state and solution candidate end state.
        Adds 1 for every similarity and divides by total amount of points.'''
        ca = CellularAutomata(k, rule, t, c0_prime)
        ct_prime = ca()

        similarity = 0.0 

        #Add one score for each corresponding integer
        for i in range(len(ct)):
            if ct_prime[i] == ct[i]:
                similarity += 1
        
        similarity = similarity/len(ct) 
        return similarity*100 #make it a ratio out of 100 for easier comprehensibility

    return objective_function_1


def mkfunc2( rule, t, ct):
    def objective_function_2(c0_prime: typing.List[int]) -> float:
        '''Objective function that compares the similarity between given end state and solution candidate end state.
        Adds 1 for every similarity and consecutive alignment and divides by total amount of point.'''
        ca = CellularAutomata(2, rule, t, c0_prime)
        ct_prime = ca()

        similarity = 0.0 
        n = len(ct)

        #add one score for each corresponding integer and one for every consecutive alignment
        for i in range(n):
            for j in range(n-i):
                if ct_prime[i+j] ==ct[i+j]:
                    similarity += 1
                    continue
                else:
                    break
        
        k = (n*(n+1))/2 #highest possible value, sum of {n, n-1 ..., 1}
        similarity = similarity/ k 

        return similarity*100 #make it a ratio out of 100 for easier comprehensibility

    return objective_function_2

def experiment():
    '''Wrap objective function and starts experiment based on the given inputs.'''

    #get values from input csv
    columns = ["k", "rule #","T","CT"]
    df = pd.read_csv("ca_input.csv", usecols=columns)
    algorithm_use = "GeneticAlgorithm" #Also result foldername

    for test in range(len(df.values)):
        print(test)
        k = df.values[test][0]
        rule = df.values[test][1]
        t = df.values[test][2]
        ct = ast.literal_eval(df.values[test][3])
        
        #Variables for GA, because IOH Analyzer doesnt take results from a loop
        pop = 100
        max_iteration = 10
        mutation = 0
        recombination = 0
        selection = 0
        objective = 0

        if objective == 0:
            # Wrap objective_function as an ioh problem
            problem = ioh.problem.wrap_integer_problem(
                    mkfunc1(k,rule,t,ct),
                    f"problem_{test}",
                    60, 
                    ioh.OptimizationType.Maximization,
                    ioh.IntegerConstraint([0]*60, [1]*60)
            )
        else:
            problem = ioh.problem.wrap_integer_problem(
                    mkfunc2(rule=rule, t= t, ct = ct),
                    f"problem_{test}",
                    60, 
                    ioh.OptimizationType.Maximization,
                    ioh.IntegerConstraint([0]*60, [1]*60)
            )
        logger = ioh.logger.Analyzer(root=f"Data", folder_name=f"{algorithm_use}/problem_{test}", algorithm_name=f"{algorithm_use}", 
                algorithm_info=f"pop = {pop}, max_iteration = {max_iteration}, mutation = {mutation}, recombination = {recombination}, selection =  {selection}, objective = {objective}"
                , store_positions=True)
        problem.attach_logger(logger)

        if algorithm_use == "GeneticAlgorithm":
            algorithm = GeneticAlgorithm(pop=pop, max_iterations=max_iteration, mutation=mutation, recombination=recombination, selection=selection)
        elif algorithm_use == "RandomSearch":
            algorithm = RandomSearch()
        algorithm(problem)

if __name__ == '__main__':
    experiment()
