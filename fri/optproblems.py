from abc import ABCMeta

import cvxpy as cvx
import numpy as np

#
# Minimal loss in constraints to mitigate numerical instabilities for solvers
#
MINLOSS = 0.01

class BaseProblem(object):
    def __init__(self, di=None, kwargs=None, X=None, Y=None,initLoss=None,initL1=None, parameters=None):
        # General data parameters
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.X = X
        self.Y = Y

        # InitModel Parameters
        self.initL1 = initL1
        self.initLoss = max(MINLOSS, initLoss)
        self.parameters = parameters
        # Dimension specific data
        self.di = di

        # Solver parameters
        self.kwargs = kwargs

        self.problem = None
        self._constraints = None
        self._objective = None

    def __str__(self):
        return '{self.__class__.__name__}(current_dim={self.di}, n={self.n}, d={self.d}, initL1={self.initL1}, initLoss={self.initLoss}, '\
                            'parameters={self.parameters}, kwargs={self.kwargs})'.format(self=self)

    def solve(self):
        self.problem = cvx.Problem(self._objective, self._constraints)
        try:
            self.problem.solve(**self.kwargs)
        except Exception as e:
            print(e)
            return None

        return self

# ****************************************************************************
# *                              Classification                              *
# ****************************************************************************

class BaseClassificationProblem(BaseProblem):
    """Base class for all common optimization problems."""

    __metaclass__ = ABCMeta

    def __init__(self, di=None, kwargs=None, X=None, Y=None, initLoss=1, initL1=1, parameters=None):
        super().__init__(di=di, kwargs=kwargs, X=X, Y=Y, initLoss=initLoss, initL1=initL1, parameters=parameters)
        # Solver parameters
        self.kwargs = kwargs

        # Solver Variables
        self.xp = cvx.Variable(nonneg=True)  # x' , our opt. value
        self.omega = cvx.Variable(self.d)  # complete linear weight vector
        self.b = cvx.Variable()  # shift

        point_distances = cvx.multiply(self.Y, self.X * self.omega + self.b)
        self.loss = cvx.sum(cvx.pos(1 - point_distances))
        self.weight_norm = cvx.norm(self.omega, 1)

        self._constraints = [
            self.weight_norm <= self.initL1
        ]

        self._constraints.extend(
            [
                self.loss <= self.initLoss
            ])


class MinProblemClassification(BaseClassificationProblem):
    """Class for minimization."""

    def __init__(self, di=None, kwargs=None, X=None, Y=None, initLoss=None, initL1=None, parameters=None):
        super().__init__(di=di, kwargs=kwargs, X=X, Y=Y, initLoss=initLoss, initL1=initL1, parameters=parameters)

        self._constraints.extend(
            [
                cvx.abs(self.omega[self.di]) <= self.xp
            ])

        self._objective = cvx.Minimize(self.xp)


class MaxProblem1(BaseClassificationProblem):
    """First Class for maximization.
        We take the maximum of both these max. classes to get the absolute value.
    """

    def __init__(self, di=None, kwargs=None, X=None, Y=None, initLoss=None, initL1=None, parameters=None):
        super().__init__(di=di, kwargs=kwargs, X=X, Y=Y, initLoss=initLoss, initL1=initL1, parameters=parameters)

        self._constraints.extend(
            [
                self.xp <= self.omega[self.di]
            ])

        self._objective = cvx.Maximize(self.xp)


class MaxProblem2(BaseClassificationProblem):
    """Second Class for maximization.
        We take the maximum of both these max. classes to get the absolute value.
    """

    def __init__(self, di=None, kwargs=None, X=None, Y=None, initLoss=None, initL1=None, parameters=None):
        super().__init__(di=di, kwargs=kwargs, X=X, Y=Y, initLoss=initLoss, initL1=initL1, parameters=parameters)

        self._constraints.extend(
            [
                self.xp <= -(self.omega[self.di])
            ])

        self._objective = cvx.Maximize(self.xp)


# ****************************************************************************
# *                                Regression                                *
# ****************************************************************************


class BaseRegressionProblem(BaseProblem):
    __metaclass__ = ABCMeta

    def __init__(self, di=None, kwargs=None, X=None, Y=None,  initLoss=None, initL1=None, parameters=None):
        super().__init__(di=di, kwargs=kwargs, X=X, Y=Y, initLoss=initLoss, initL1=initL1,parameters=parameters)

        # General data
        # self.Y = Y # other format then with classification
        self.Y = Y.reshape((-1, 1))

        self.epsilon = self.parameters["epsilon"]

        # Solver parameters
        self.kwargs = kwargs

        # Solver Variables
        self.xp = cvx.Variable()  # x' , our opt. value
        self.omega = cvx.Variable(shape=(self.d, 1))  # complete linear weight vector
        self.b = cvx.Variable()  # offset from origin
        self.slack = cvx.Variable(shape=(self.n, 1), nonneg=True)  # slack variables
        self.loss = cvx.sum(self.slack)
        self.weight_norm = cvx.norm(self.omega, 1)

        self._constraints = [
            self.weight_norm <= self.initL1,
            self.loss <= self.initLoss,
            cvx.abs(self.Y - (self.X * self.omega + self.b)) <= self.epsilon + self.slack
        ]


class MinProblemRegression(BaseRegressionProblem):
    """Class for minimization."""

    def __init__(self, di=None, kwargs=None, X=None, Y=None, initLoss=None, initL1=None, parameters=None):
        super().__init__(di=di, kwargs=kwargs, X=X, Y=Y, initLoss=initLoss, initL1=initL1, parameters=parameters)

        self._constraints.extend(
            [
                cvx.abs(self.omega[self.di]) <= self.xp,
            ])

        self._objective = cvx.Minimize(self.xp)


class MaxProblem1Regression(BaseRegressionProblem):
    """Class for maximization."""

    def __init__(self, di=None, kwargs=None, X=None, Y=None, epsilon=None, initLoss=None, initL1=None, parameters=None):
        super().__init__(di=di, kwargs=kwargs, X=X, Y=Y, initLoss=initLoss, initL1=initL1, parameters=parameters)

        self._constraints.extend(
            [
                self.xp <= self.omega[self.di]
            ])

        self._objective = cvx.Maximize(self.xp)


class MaxProblem2Regression(BaseRegressionProblem):
    """Class for maximization."""

    def __init__(self, di=None, kwargs=None, X=None, Y=None, epsilon=None, initLoss=None, initL1=None, parameters=None):
        super().__init__(di=di, kwargs=kwargs, X=X, Y=Y, initLoss=initLoss, initL1=initL1, parameters=parameters)

        self._constraints.extend(
            [
                self.xp <= -self.omega[self.di]
            ])

        self._objective = cvx.Maximize(self.xp)


# ****************************************************************************
# *                            Ordinal Regression                            *
# ****************************************************************************
# TODO: define problems as cvxpy problem with constraints and objective
class BaseOrdinalRegressionProblem(BaseProblem):
    __metaclass__ = ABCMeta
    def __init__(self, di=None, kwargs=None, X=None, Y=None,  initLoss=None, initL1=None, parameters=None):
        super().__init__(di=di, kwargs=kwargs, X=X, Y=Y, initLoss=initLoss, initL1=initL1,parameters=parameters)

        # Solver parameters
        self.kwargs = kwargs

        # Optimal parameters from initial gridsearch
        self.C = self.parameters["C"]
        self.w_opt = self.parameters["w"]
        self.chi_opt = self.parameters["chi"]
        self.xi_opt = self.parameters["xi"]

        #TODO: Set appropriate delta value
        self.delta = 0.1
        self.mu = np.linalg.norm(self.w_opt, ord=1) + self.C * np.sum(self.chi_opt + self.xi_opt)

        # Prepare the same Problem structure as in initial search
        (n, d) = X.shape
        n_bins = len(np.unique(Y))

        X_re = []
        y = np.array(Y)
        for i in range(n_bins):
            indices = np.where(y == i)
            X_re.append(X[indices])


        self.w = cvx.Variable(shape=(d, 1))
        self.b = cvx.Variable(shape=(n_bins - 1, 1))
        self.chi = []
        self.xi = []
        for i in range(n_bins):
            n_x = len(np.where(y == i)[0])
            self.chi.append(cvx.Variable(shape=(n_x, 1)))
            self.xi.append(cvx.Variable(shape=(n_x, 1)))


        self._constraints = []
        for i in range(n_bins - 1):
            self._constraints.append(X_re[i] * self.w - self.chi[i] <= self.b[i] - 1)

        for i in range(1, n_bins):
            self._constraints.append(X_re[i] * self.w + self.xi[i] >= self.b[i - 1] + 1)

        for i in range(n_bins - 2):
            self._constraints.append(self.b[i] <= self.b[i + 1])

        for i in range(n_bins):
            self._constraints.append(self.chi[i] >= 0)
            self._constraints.append(self.xi[i] >= 0)

        # Extend contstraints with regard to the initial problem
        self._constraints.append(cvx.norm(self.w,1) + self.C * cvx.sum(cvx.hstack(self.chi) + cvx.hstack(self.xi)) <= (1 + self.delta) * self.mu)
        self._constraints.append(self.w <= cvx.abs(self.w))
        self._constraints.append(-self.w <= cvx.abs(self.w))





class MinProblemOrdinalRegression(BaseOrdinalRegressionProblem):
    def __init__(self, di=None, kwargs=None, X=None, Y=None, initLoss=None, initL1=None, parameters=None):
        super().__init__(di=di, kwargs=kwargs, X=X, Y=Y, initLoss=initLoss, initL1=initL1, parameters=parameters)

        # Define problem specific objective
        self._objective = cvx.Minimize(cvx.abs(self.w[self.di]))



class MaxProblem1OrdinalRegression(BaseOrdinalRegressionProblem):
    def __init__(self, di=None, kwargs=None, X=None, Y=None, epsilon=None, initLoss=None, initL1=None, parameters=None):
        super().__init__(di=di, kwargs=kwargs, X=X, Y=Y, initLoss=initLoss, initL1=initL1, parameters=parameters)

        # Extend problem specific constraints
        self._constraints.append(cvx.abs(self.w[self.di]) <= self.w[self.di])

        # Define problem specific objective
        self._objective = cvx.Maximize(cvx.abs(self.w[self.di]))



class MaxProblem2OrdinalRegression(BaseOrdinalRegressionProblem):
    def __init__(self, di=None, kwargs=None, X=None, Y=None, epsilon=None, initLoss=None, initL1=None, parameters=None):
        super().__init__(di=di, kwargs=kwargs, X=X, Y=Y, initLoss=initLoss, initL1=initL1, parameters=parameters)

        # Extend problem specific constraints
        self._constraints.append(cvx.abs(self.w[self.di]) <= -self.w[self.di])

        # Define problem specific objective
        self._objective = cvx.Maximize(cvx.abs(self.w[self.di]))
