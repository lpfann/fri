from abc import ABCMeta, abstractmethod

from cvxpy import OPTIMAL,OPTIMAL_INACCURATE

import fri.optproblems
import numpy as np

class NotFeasibleForParameters(Exception):
    """SVM cannot separate points with this parameters"""

class Bound(object):
    __metaclass__ = ABCMeta

    """Class for lower and upper relevance bounds"""
    def __init__(self,di,X,Y):
        self.X = X
        self.Y = Y
        self.di = di
        self.acceptableStati = [OPTIMAL, OPTIMAL_INACCURATE]

    @abstractmethod
    def solve(self):
        pass


class LowerBound(Bound):
    """Class for lower bounds """

    def __init__(self, di, d, n, kwargs, L1, svmloss, C, X, Y, regression=False, epsilon=None):
        super().__init__(di, X,Y)
        if regression:
            prob = fri.optproblems.MinProblemRegression
            self.prob_instance = prob(di=di, d=d, n=n, kwargs=kwargs, X=self.X, Y=self.Y, C=C, svmloss=svmloss, L1=L1,epsilon=epsilon)

        else:
            prob = fri.optproblems.MinProblemClassification
            self.prob_instance = prob(di=di, d=d, n=n, kwargs=kwargs, X=self.X, Y=self.Y, C=C, svmloss=svmloss, L1=L1)

        self.type = 0

    def solve(self):
        status = self.prob_instance.solve().problem.status

        if status in self.acceptableStati:
            return self
        else:
            raise NotFeasibleForParameters

class UpperBound(Bound):
    """Class for Upper bounds """

    def __init__(self, di, d, n, kwargs, L1, svmloss, C, X, Y, regression=False,epsilon=None):
        super().__init__(di, X,Y)
        if regression:
            prob1 = fri.optproblems.MaxProblem1Regression
            prob2 = fri.optproblems.MaxProblem2Regression
            self.prob_instance1 = prob1(di=di, d=d, n=n, kwargs=kwargs, X=self.X, Y=self.Y, C=C, svmloss=svmloss, L1=L1,epsilon=epsilon)
            self.prob_instance2 = prob2(di=di, d=d, n=n, kwargs=kwargs, X=self.X, Y=self.Y, C=C, svmloss=svmloss, L1=L1,epsilon=epsilon)
        else:
            prob1 = fri.optproblems.MaxProblem1
            prob2 = fri.optproblems.MaxProblem2
            self.prob_instance1 = prob1(di=di, d=d, n=n, kwargs=kwargs, X=self.X, Y=self.Y, C=C, svmloss=svmloss, L1=L1)
            self.prob_instance2 = prob2(di=di, d=d, n=n, kwargs=kwargs, X=self.X, Y=self.Y, C=C, svmloss=svmloss, L1=L1)

        self.type = 1
        
    def solve(self):
        status = [None,None]
        status[0] = self.prob_instance1.solve()
        status[1] = self.prob_instance2.solve()
        status =list(filter(lambda x: x is not None, status))

        valid_problems = list(filter(lambda x: x.problem.status in self.acceptableStati, status))
        if len(valid_problems) == 0:
           raise NotFeasibleForParameters
        max_index = np.argmax([np.abs(x.problem.value) for x in valid_problems])
        best_problem = valid_problems[max_index]
        self.prob_instance = best_problem
        return self


class ShadowLowerBound(LowerBound):
    """Class for lower bounds """
    def __init__(self, di, d, n, kwargs, L1, svmloss, C, X, Y, regression=False,epsilon=None,random_state=None):
        if not random_state:
            X = np.append(np.random.permutation(X[:, di]).reshape((n, 1)), X ,axis=1)
        else:
            X = np.append(random_state.permutation(X[:, di]).reshape((n, 1)), X ,axis=1)

        super().__init__(0, d + 1, n, kwargs, L1, svmloss, C, X, Y, regression=regression,epsilon=epsilon)
        self.isShadow = True
        self.di = di

class ShadowUpperBound(UpperBound):
    """Class for Upper bounds """
    
    
    def __init__(self, di, d, n, kwargs, L1, svmloss, C, X, Y, regression=False,epsilon=None,random_state=None):
        if not random_state:
            X = np.append(np.random.permutation(X[:, di]).reshape((n, 1)), X ,axis=1)
        else:
            X = np.append(random_state.permutation(X[:, di]).reshape((n, 1)), X ,axis=1)
            
        super().__init__(0, d + 1, n, kwargs, L1, svmloss, C, X, Y, regression=regression,epsilon=epsilon)
        self.isShadow = True
        self.di = di

