from abc import ABCMeta, abstractmethod
import rbclassifier.optproblems
import numpy as np

class NotFeasibleForParameters(Exception):
    """SVM cannot separate points with this parameters"""

class Bound(object):
    __metaclass__ = ABCMeta

    """Class for lower and upper relevance bounds"""
    def __init__(self):
        pass
        
    @abstractmethod
    def solve(self):
        pass


class LowerBound(Bound):
    """Class for lower bounds """
    problem = rbclassifier.optproblems.MinProblem
    def __init__(self,acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y):
        super()
        self.acceptableStati = acceptableStati
        self.prob_instance = self.problem(acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y)

    def solve(self):
        status = self.prob_instance.solve().problem.status

        if status in self.acceptableStati:
            return self.prob_instance
        else:
            raise NotFeasibleForParameters
           


class UpperBound(Bound):
    """Class for Upper bounds """
    
        
    
    def __init__(self,acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y):
        super()
        problem1 = rbclassifier.optproblems.MaxProblem1
        problem2 = rbclassifier.optproblems.MaxProblem2
        self.acceptableStati = acceptableStati
        self.prob_instance1 = problem1(acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y)
        self.prob_instance2 = problem2(acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y)        
        
    def solve(self):
        status = [None,None]
        status[0] = self.prob_instance1.solve()
        status[1] = self.prob_instance2.solve()
        valid_problems = list(filter(lambda x: x.problem.status in self.acceptableStati, status))
        if len(valid_problems) == 0:
           raise NotFeasibleForParameters
        max_index = np.argmax([np.abs(x.problem.value) for x in valid_problems])
        best_problem = valid_problems[max_index]

        return best_problem
