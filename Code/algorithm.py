import abc
import random
import functools
import inspect 
import ioh

class Algorithm(abc.ABC):
    '''This class should be used as a base class (interface) for the implementation as Genetic Algorithm.     
    '''
    
    def __init__(self, max_iterations:int = 500):
        self.max_iterations: int = max_iterations
    
    def __init_subclass__(cls, *args, **kwargs): 
        def check_init(init):        
            @functools.wraps(init)
            def wrapper(instance, *args, **kwargs):
                init(instance, *args, **kwargs)
                if not hasattr(instance, "max_iterations"):
                    raise AttributeError(
                       f"Required 'max_iterations' attribute is not defined on {instance}. "
                       f"Did you forget to call super().__init__ in the __init__ method?"
                    )
            
              
            return wrapper
        
        cls.__init__ = check_init(cls.__init__)        
        super().__init_subclass__(**kwargs)


    @abc.abstractmethod
    def __call__(self, problem: ioh.problem.Integer) -> None:
        pass


