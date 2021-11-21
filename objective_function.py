import typing
import ioh

from implementation import GeneticAlgorithm, RandomSearch


class CellularAutomata:
    '''Skeleton CA, you should implement this.'''
    
    def __init__(self,k, rule_number: int, t: int, c0: typing.List[int]):
        self.k = k
        self.rule_number = rule_number
        self.t = t
        self.c0 = c0
        self.radius = 1
        
    
    def __call__(self):
        '''Evaluate for T timesteps. Return Ct for a given C0.'''
        state = self.c0

        for i in range(self.t):
            state = self.transition(state, self.k )
        
        return state


    def neighbours(self, c0, cell_position):
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

    def rule_list(self):
        binary_list = []
        binary_rule = format(self.rule_number, "b")
        for i in binary_rule:
            binary_list.append(i)
        if self.rule_number <256:
            for i in range(8-len(binary_list)):
                binary_list.insert(0, 0)
        else:
            for i in range(27-len(binary_list)):
                binary_list.insert(0, 0)
        return binary_list

    def transition(self, c0, k):
        if k == 2:
            states = [[1,1,1], [1,1,0], [1,0,1], [1,0,0], [0,1,1], [0,1,0], [0,0,1], [0,0,0]]
        elif k == 3:
            states = [[2, 2, 2], [2, 2, 1], [2, 2, 0], [2, 1, 2], [2, 1, 1], [2, 1, 0], [2, 0, 2], [2, 0, 1], [2, 0, 0], 
                      [1, 2, 2], [1, 2, 1], [1, 2, 0], [1, 1, 2], [1, 1, 1], [1, 1, 0], [1, 0, 2], [1, 0, 1], [1, 0, 0], 
                      [0, 2, 2], [0, 2, 1], [0, 2, 0], [0, 1, 2], [0, 1, 1], [0, 1, 0], [0, 0, 2], [0, 0, 1], [0, 0, 0]]
        rule_states = self.rule_list()
        new_state = []
        for i in range(len(c0)):
            neighbourhood = self.neighbours(c0, i, self.radius)
            index_state = states.index(neighbourhood)
            new_state.append(rule_states[index_state])
        return new_state

def objective_function(c0_prime: typing.List[int]) -> float:
    '''Skeleton objective function. You should implement a method
    which computes a similarity measure between c0_prime a suggested by your
    GA, with the true c0 state for the ct state given in the sup. material. '''
    
    ct, rule, t = None, None, None # Given by the sup. material 

    ca = CellularAutomata(rule)
    ct_prime = ca(c0_prime, t)
    similarity = 0.0 # You should implement this

    return similarity

        
def example():
    '''An example of wrapping a objective function in ioh and collecting data
    for inputting in the analyzer.'''
    
    algorithm = GeneticAlgorithm()

    # Wrap objective_function as an ioh problem
    problem = ioh.problem.wrap_integer_problem(
            objective_function,
            "objective_function_ca_1",
            60, 
            ioh.OptimizationType.Maximization,
            ioh.IntegerConstraint([0]*60, [1]*60)
    )
    # Attach a logger to the problem
    logger = ioh.logger.Analyzer(store_positions=True)
    problem.attach_logger(logger)

    # run your algoritm on the problem
    algorithm(problem)



if __name__ == '__main__':
    example()
