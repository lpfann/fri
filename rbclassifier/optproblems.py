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
        self.problem.solve(**self.kwargs)
        status = self.problem.status

        if status in self.acceptableStati:
            return RelevanceBoundsClassifier.Opt_output(prob_min.value, omega.value.reshape(d), b.value)
        else:
            # return softMarginLPOptimizer.Opt_output(0, 0, 0)
            raise NotFeasibleForParameters

class MinProblem(BaseProblem):
    """Class for minimization."""

    def __init__(self, acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y):
        super(acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y)

        self._constraints = self._constraints +
        [
            cvx.abs(self.omega) <= self.xp,
        ]

        self.dim = np.zeros(self.d)
        self.dim[self.di] = 1
        self.omegai.value = self.dim

        self._objective = cvx.Minimize(self.xp.T * self.omegai)
        self.problem = cvx.Problem(self._objective, self._constraints)
        return self


class MaxProblemBase(BaseProblem):
    """Class for maximization."""


    def __init__(self, acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y):
        super(acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y)

        self._constraints = self._constraints + [
            #cvx.abs(omega) <= xp,
            self.xp >= 0,
        ]

        return self

class MaxProblem1(MaxProblemBase):
    """Class for maximization."""

    def __init__(self, acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y):
        super(acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y)

        self._constraints = self._constraints + [
           self.xp.T * self.omegai <= self.omega.T * self.omegai,
            self.xp.T * self.omegai <= -(self.omega.T * self.omegai) + self.M
        ]

        self._objective = cvx.Maximize(self.xp.T * self.omegai)
        self.problem = cvx.Problem(self._objective, self._constraints)


        return self

class MaxProblem2(MaxProblemBase):
    """Class for maximization."""

    def __init__(self, acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y):
        super(acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y)

        maxConst2 = [   
            xp.T * omegai <= -(omega.T * omegai),
            xp.T * omegai <= (omega.T * omegai) + M
        ]
        maxConst2.extend(constraints2[:])
        obj_max2 = cvx.Maximize(xp.T * omegai)
        prob_max2 = cvx.Problem(obj_max2, maxConst2)

        self.problem = prob_min

        return self

    def _opt_max(self, acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y):


        dim = np.zeros(d)
        dim[di] = 1
        omegai.value = dim
        valid = False
        prob_max1.solve(**kwargs)
        status = prob_max1.status
        opt_value = 0
        weights = None
        bias = None
        if status in acceptableStati:
            opt_value = np.abs(prob_max1.value)
            weights = omega.value.reshape(d)
            bias = b.value
            valid = True
        prob_max2.solve(**kwargs)
        status = prob_max2.status
        if status in acceptableStati and np.abs(prob_max2.value) > np.abs(opt_value):
            opt_value = np.abs(prob_max2.value)
            weights = omega.value.reshape(d)
            bias = b.value
            valid = True

        if not valid:
            # return softMarginLPOptimizer.Opt_output(0, 0, 0)
            raise NotFeasibleForParameters
        else:
            return RelevanceBoundsClassifier.Opt_output(opt_value, weights, bias)

        

        
        