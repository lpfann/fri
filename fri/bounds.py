from abc import ABCMeta, abstractmethod

import numpy as np
from cvxpy import OPTIMAL, OPTIMAL_INACCURATE


class NotFeasibleForParameters(Exception):
    """SVM cannot separate points with this parameters"""


class Bound(object):
    __metaclass__ = ABCMeta

    """Class for lower and upper relevance bounds"""

    def __init__(self, optim_dim, X, Y, initLoss, initL1):
        self.optim_dim = optim_dim
        self.X = X
        self.Y = Y
        self.initL1 = initL1
        self.initLoss = initLoss
        self.optim_dim = optim_dim
        self.acceptableStati = [OPTIMAL, OPTIMAL_INACCURATE]
        self.isUpperBound = None

    @abstractmethod
    def solve(self):
        pass
    def __repr__(self):
        return "{self.__class__.__name__}(optim_dim={self.optim_dim}, X.shape={self.X.shape}, Y.shape={self.Y.shape}, initL1={self.initL1}, initLoss={self.initLoss})".format(
            self=self)

class LowerBound(Bound):
    """Class for lower bounds """

    def __init__(self, problemClass=None, optim_dim=None, kwargs=None, initLoss=None, initL1=None, X=None, Y=None):
        # Init Super class, could be used for data manipulation
        super().__init__(optim_dim, X, Y)
        super().__init__(optim_dim, X, Y, initLoss, initL1)


        # Init problem instance usually defined in the main class
        self.prob_instance = problemClass.minProblem(di=optim_dim, kwargs=kwargs, X=self.X, Y=self.Y, initLoss=initLoss, initL1=initL1, parameters=problemClass._best_params)

        # Define bound type for easier indexing after result collection
        self.isUpperBound = False

    def solve(self):
        status = self.prob_instance.solve().problem.status

        if status in self.acceptableStati:
            return self
        else:
            print("DEBUG: Lower Bound - current_feature={} - Status={}".format(self.optim_dim, status))
            raise NotFeasibleForParameters


class UpperBound(Bound):
    """Class for Upper bounds """

    def __init__(self, problemClass=None, optim_dim=None, kwargs=None, initLoss=None, initL1=None, X=None, Y=None):
        super().__init__(optim_dim, X, Y, initLoss, initL1)

        self.prob_instance1 = problemClass.maxProblem1(di=optim_dim, kwargs=kwargs, X=self.X, Y=self.Y, initLoss=initLoss, initL1=initL1, parameters=problemClass._best_params)
        self.prob_instance2 = problemClass.maxProblem2(di=optim_dim, kwargs=kwargs, X=self.X, Y=self.Y, initLoss=initLoss, initL1=initL1, parameters=problemClass._best_params)

        self.isUpperBound = True

    def solve(self):
        status = [None, None]
        status[0] = self.prob_instance1.solve()
        status[1] = self.prob_instance2.solve()
        status = list(filter(lambda x: x is not None, status))

        valid_problems = list(filter(lambda x: x.problem.status in self.acceptableStati, status))
        if len(valid_problems) == 0:
            print("DEBUG: Upper Bound - current_feature={}".format(self.optim_dim))
            raise NotFeasibleForParameters

        max_index = np.argmax([np.abs(x.problem.value) for x in valid_problems])
        best_problem = valid_problems[max_index]

        self.prob_instance = best_problem
        return self


class ShadowLowerBound(LowerBound):
    """ Class for shadow lower bounds 
        Permute the data to get bounds for random data.
    """

    def __init__(self, problemClass=None, optim_dim=None, kwargs=None, initLoss=None, initL1=None, X=None, Y=None, random_state=None):
        # Permute dimension optim_dim and insert it at the first column
        if not random_state:
            X = np.append(np.random.permutation(X[:, optim_dim]).reshape((X.shape[0], 1)), X, axis=1)
        else:
            X = np.append(random_state.permutation(X[:, optim_dim]).reshape((X.shape[0], 1)), X, axis=1)
        # Optimize for the first (random permutated) column
        super().__init__(problemClass, 0, kwargs, initLoss, initL1, X, Y)
        self.isShadow = True
        self.optim_dim = optim_dim


class ShadowUpperBound(UpperBound):
    """ Class for shadow upper bounds 
        Permute the data to get bounds for random data.
    """

    def __init__(self, problemClass=None, optim_dim=None, kwargs=None, initLoss=None, initL1=None, X=None, Y=None, random_state=None):
        # Permute dimension optim_dim and insert it at the first column
        if not random_state:
            X = np.append(np.random.permutation(X[:, optim_dim]).reshape((X.shape[0], 1)), X, axis=1)
        else:
            X = np.append(random_state.permutation(X[:, optim_dim]).reshape((X.shape[0], 1)), X, axis=1)
        
        # Optimize for the first (random permutated) column
        super().__init__(problemClass, 0, kwargs, initLoss, initL1, X, Y)
        self.isShadow = True
        self.optim_dim = optim_dim
