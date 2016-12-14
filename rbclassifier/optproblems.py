from abc import ABCMeta, abstractmethod
import cvxpy as cvx
import numpy as np

class BaseProblem(object):
    """Base class for all common optimization problems."""

    __metaclass__ = ABCMeta

    def __init__(self, acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y):
        # General data
        self.d = d
        self.n = n
        self.L1 = L1
        self.svmloss = svmloss
        self.C = C
        self.X = X
        self.Y = np.array([Y, ] * 1)
        self.M = 2 * L1
        # Dimension specific data
        self.di = di
        # Solver parameters
        self.acceptableStati = acceptableStati
        self.kwargs = kwargs
        # Solver Variables
        self.xp = cvx.Variable(d)       # x' , our opt. value
        self.omega = cvx.Variable(d)    # complete linear weight vector
        self.b = cvx.Variable()         # shift
        self.eps = cvx.Variable(n)      # slack variables

        self.problem = None

        self._constraints =  [  
                                # points still correctly classified with soft margin
                                cvx.mul_elemwise(self.Y.T, self.X * self.omega - self.b) >= 1 - self.eps,
                                self.eps >= 0,
                                # L1 reg. and allow slack
                                cvx.norm(self.omega, 1) + self.C * cvx.sum_squares(self.eps) <= self.L1 + self.C * self.svmloss
            ]

    def solve(self):

        self.problem = cvx.Problem(self._objective, self._constraints)
        self.problem.solve(**self.kwargs)
        return self



class MinProblem(BaseProblem):
    """Class for minimization."""

    def __init__(self, acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y):
        super().__init__(acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y)

        self._constraints.extend(
        [
            cvx.abs(self.omega) <= self.xp,
        ])

        self._objective = cvx.Minimize(self.xp[self.di])


class MaxProblemBase(BaseProblem):
    """Class for maximization."""


    def __init__(self, acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y):
        super().__init__(acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y)

        self._constraints.extend([
            cvx.abs(self.omega) <= self.xp,
            #self.xp[self.di] >= 0,
        ])


class MaxProblem1(MaxProblemBase):
    """Class for maximization."""

    def __init__(self, acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y):
        super().__init__(acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y)

        self._constraints.extend( [
           self.xp[self.di] <= self.omega[self.di],
            self.xp[self.di] <= -(self.omega[self.di]) + self.M
        ])

        self._objective = cvx.Maximize(self.xp[self.di])


class MaxProblem2(MaxProblemBase):
    """Class for maximization."""

    def __init__(self, acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y):
        super().__init__(acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y)

        self._constraints.extend( [  
            self.xp[self.di] <= -(self.omega[self.di]),
            self.xp[self.di] <= (self.omega[self.di]) + self.M
        ])

        self._objective = cvx.Maximize(self.xp[self.di])


