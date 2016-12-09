from abc import ABCMeta, abstractmethod
import cvxpy as cvx
import numpy as np

class BaseProblem(object):
    """Base class for all common optimization problems."""

    __metaclass__ = ABCMeta

    def __init__(self, acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y):
        self.acceptableStati = acceptableStati
        self.di = di
        self.d = d
        self.n = n
        self.kwargs = kwargs
        self.L1 = L1
        self.svmloss = svmloss
        self.C = C
        self.X = X
        self.Y = np.array([Y, ] * 1)
        self.M = 2 * L1
        self.xp = cvx.Variable(d)
        self.omega = cvx.Variable(d)
        self.omegai = cvx.Parameter(d)
        self.b = cvx.Variable()
        self.eps = cvx.Variable(n)
        self.problem = None

        self._constraints =  [  
                                # points still correctly classified with soft margin
                                cvx.mul_elemwise(self.Y.T, self.X * self.omega - self.b) >= 1 - self.eps,
                                self.eps >= 0,
                                # L1 reg. and allow slack
                                cvx.norm(self.omega, 1) + self.C * cvx.sum_squares(self.eps) <= self.L1 + self.C * self.svmloss
            ]

    def solve(self, di):
        self.di = di
        self.dim = np.zeros(self.d)
        self.dim[self.di] = 1
        self.omegai.value = self.dim

        self.problem = cvx.Problem(self._objective, self._constraints)
        self.problem.solve(**self.kwargs)
        return self.problem



class MinProblem(BaseProblem):
    """Class for minimization."""

    def __init__(self, acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y):
        super().__init__(acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y)

        self._constraints.extend(
        [
            cvx.abs(self.omega) <= self.xp,
        ])

        self._objective = cvx.Minimize(self.xp.T * self.omegai)


class MaxProblemBase(BaseProblem):
    """Class for maximization."""


    def __init__(self, acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y):
        super().__init__(acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y)

        self._constraints.extend([
            #cvx.abs(omega) <= xp,
            self.xp >= 0,
        ])


class MaxProblem1(MaxProblemBase):
    """Class for maximization."""

    def __init__(self, acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y):
        super().__init__(acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y)

        self._constraints.extend( [
           self.xp.T * self.omegai <= self.omega.T * self.omegai,
            self.xp.T * self.omegai <= -(self.omega.T * self.omegai) + self.M
        ])

        self._objective = cvx.Maximize(self.xp.T * self.omegai)


class MaxProblem2(MaxProblemBase):
    """Class for maximization."""

    def __init__(self, acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y):
        super().__init__(acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y)

        self._constraints.extend( [  
            self.xp.T * self.omegai <= -(self.omega.T * self.omegai),
            self.xp.T * self.omegai <= (self.omega.T * self.omegai) + self.M
        ])

        self._objective = cvx.Maximize(self.xp.T * self.omegai)


