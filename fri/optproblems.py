from abc import ABCMeta

import cvxpy as cvx

#
# Minimal loss in constraints to mitigate numerical instabilities for solvers
#
MINLOSS = 0.01

class BaseProblem(object):
    def __init__(self, di=0, kwargs=None, X=None, Y=None,initLoss=None,initL1=None, parameters=None):
        # General data parameters
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.X = X
        self.Y = Y

        # InitModel Parameters
        self.initL1 = initL1
        self.initLoss = max(MINLOSS, initLoss)
        print(parameters)
        self.parameters = parameters
        # Dimension specific data
        self.di = di

        # Solver parameters
        self.kwargs = kwargs

        self.problem = None
        self._constraints = None
        self._objective = None

    def solve(self):
        self.problem = cvx.Problem(self._objective, self._constraints)
        try:
            self.problem.solve(**self.kwargs)
        except Exception as e:
            print(e)
            return None

        return self


class BaseClassificationProblem(BaseProblem):
    """Base class for all common optimization problems."""

    __metaclass__ = ABCMeta

    def __init__(self, di=0, kwargs=None, X=None, Y=None, initLoss=1, initL1=1, parameters=None):
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

    def __init__(self, di=0, kwargs=None, X=None, Y=None, initLoss=1, initL1=1, parameters=None):
        super().__init__(di=di, kwargs=kwargs, X=X, Y=Y, initLoss=initLoss, initL1=initL1, parameters=parameters)

        self._constraints.extend(
            [
                cvx.abs(self.omega[di]) <= self.xp
            ])

        self._objective = cvx.Minimize(self.xp)


class MaxProblem1(BaseClassificationProblem):
    """First Class for maximization.
        We take the maximum of both these max. classes to get the absolute value.
    """

    def __init__(self, di=0, kwargs=None, X=None, Y=None, initLoss=1, initL1=1, parameters=None):
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

    def __init__(self, di=0, kwargs=None, X=None, Y=None, initLoss=1, initL1=1, parameters=None):
        super().__init__(di=di, kwargs=kwargs, X=X, Y=Y, initLoss=initLoss, initL1=initL1, parameters=parameters)

        self._constraints.extend(
            [
                self.xp <= -(self.omega[self.di])
            ])

        self._objective = cvx.Maximize(self.xp)


'''
#############
            ##### REGRESSION
#############
'''


class BaseRegressionProblem(BaseProblem):
    __metaclass__ = ABCMeta

    def __init__(self, di=0, kwargs=None, X=None, Y=None,  initLoss=1, initL1=1, parameters=None):
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

    def __init__(self, di=0, kwargs=None, X=None, Y=None, initLoss=1, initL1=1, parameters=None):
        super().__init__(di=di, kwargs=kwargs, X=X, Y=Y, initLoss=initLoss, initL1=initL1, parameters=parameters)

        self._constraints.extend(
            [
                cvx.abs(self.omega[self.di]) <= self.xp,
            ])

        self._objective = cvx.Minimize(self.xp)


class MaxProblem1Regression(BaseRegressionProblem):
    """Class for maximization."""

    def __init__(self, di=0, kwargs=None, X=None, Y=None, epsilon=0.1, initLoss=1, initL1=1, parameters=None):
        super().__init__(di=di, kwargs=kwargs, X=X, Y=Y, initLoss=initLoss, initL1=initL1, parameters=parameters)

        self._constraints.extend(
            [
                self.xp <= self.omega[self.di]
            ])

        self._objective = cvx.Maximize(self.xp)


class MaxProblem2Regression(BaseRegressionProblem):
    """Class for maximization."""

    def __init__(self, di=0, kwargs=None, X=None, Y=None, epsilon=0.1, initLoss=1, initL1=1, parameters=None):
        super().__init__(di=di, kwargs=kwargs, X=X, Y=Y, initLoss=initLoss, initL1=initL1, parameters=parameters)

        self._constraints.extend(
            [
                self.xp <= -self.omega[self.di]
            ])

        self._objective = cvx.Maximize(self.xp)
