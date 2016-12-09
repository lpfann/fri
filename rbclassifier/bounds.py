from abc import ABCMeta, abstractmethod
import rbclassifier.optproblems

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
    def __init__(self):
        super()

    def solve(self,acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y):
        prob_instance =self.problem(acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y)
        status  = prob_instance.solve(di)
        
        if status in acceptableStati:
            return prob_instance
        else:
            raise NotFeasibleForParameters
           


class UpperBound(Bound):
    """Class for Upper bounds """
    problem1 = rbclassifier.optproblems.MaxProblem1
    problem2 = rbclassifier.optproblems.MaxProblem2
        
    
    def __init__(self):
        super()        
        
    def solve(self,acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y):
        prob_instance1 = self.problem1(acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y)
        prob_instance2 = self.problem2(acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y)

        status[0] = prob_instance1.solve(di)
        status[1] = prob_instance2.solve(di)
        valid_problems = filter(lambda x: x.status in acceptableStati, status)
        if len(valid_problems) == 0:
            raise NotFeasibleForParameters
        max_index = np.argmax([np.abs(x.value) for x in valid_problems])
        best_problem = valid_problems[max_index]

        return best_problem
