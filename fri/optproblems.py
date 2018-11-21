import abc

import cvxpy as cvx
import numpy as np
from cvxpy import SolverError

class ProblemType(object, metaclass=abc.ABCMeta):
    # Decorator class to add problem type specific constraints and variables to the BaseProblem
    @abc.abstractmethod
    def add_type_specific(self, baseProblem):
        pass
#
# Minimal loss in constraints to mitigate numerical instabilities for solvers
#
MINLOSS = 0.01

class BaseProblem(object):
    def __init__(self, problemType: ProblemType, di: int = None, kwargs: dict = None,
                 X: np.ndarray = None, Y: np.ndarray = None, initLoss: float = None, initL1: float = None,
                 parameters: dict = None,
                 presetModel: np.ndarray = None):
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
        self._constraints = []
        self.xp = cvx.Variable(nonneg=True, name="currentDimWeight")  # x' , our opt. value
        self.omega = cvx.Variable(self.d, name="Omega")  # complete linear weight vector
        if kwargs is None:
            self.kwargs = {}  # Fix (unpacking)error when manually calling bounds without keyword arguments to the solver
        else:
            self.kwargs = kwargs

        # Check if some model values have a pre fixed value
        self.presetModel = presetModel
        if presetModel is not None:
            if np.all(np.isnan(presetModel)):
                self.presetModel = None
            else:
                assert presetModel.shape == (self.d, 2)
                for dim in range(self.d):
                    # Skip current dimension
                    if dim == di:
                        continue
                    current_preset = presetModel[dim]

                    # Skip unset values
                    if all(np.isnan(current_preset)):
                        continue

                    # a weight bigger than the optimal model L1 makes no sense
                    assert abs(current_preset[0]) <= self.initL1
                    assert abs(current_preset[1]) <= self.initL1
                    # We add a pair of constraints depending on sign of known coefficient
                    # this makes it possible to solve this as a convex problem
                    if current_preset[0] >= 0:
                        self._constraints.extend([
                            self.omega[dim] >= current_preset[0],
                            self.omega[dim] <= current_preset[1],
                        ])
                    else:
                        self._constraints.extend([
                            self.omega[dim] <= current_preset[0],
                            self.omega[dim] >= current_preset[1],
                        ])
        self.problem = None
        self._objective = None

        # Add problem type specific constraints and variables to cvxpy
        problemType().add_type_specific(self)

    def __str__(self):
        return '{self.__class__.__name__}(current_dim={self.di}, n={self.n}, d={self.d}, initL1={self.initL1}, initLoss={self.initLoss}, ' \
               'parameters={self.parameters}, kwargs={self.kwargs}, presetModel={self.presetModel})'.format(self=self)

    def solve(self):
        self.problem = cvx.Problem(self._objective, self._constraints)
        try:
            self.problem.solve(**self.kwargs)
        except SolverError as e:
            kwargs = self.kwargs
            kwargs["solver"] = "SCS"
            self.problem.solve(**kwargs) 
        except Exception as e:
            return None
        return self


class MinProblem(BaseProblem):
    def __init__(self, problemType, di=None, kwargs=None, X=None, Y=None, initLoss=None, initL1=None, parameters=None, presetModel=None):
        super().__init__(problemType, di=di, kwargs=kwargs, X=X, Y=Y, initLoss=initLoss, initL1=initL1,
                         parameters=parameters, presetModel=presetModel)

        self._constraints.extend(
            [
                cvx.abs(self.omega[self.di]) <= self.xp
            ])

        self._objective = cvx.Minimize(self.xp)


class MaxProblem1(BaseProblem):
    def __init__(self, problemType, di=None, kwargs=None, X=None, Y=None, initLoss=None, initL1=None, parameters=None, presetModel=None):
        super().__init__(problemType, di=di, kwargs=kwargs, X=X, Y=Y, initLoss=initLoss, initL1=initL1,
                         parameters=parameters,presetModel=presetModel)

        self._constraints.extend(
            [
                self.xp <= self.omega[self.di]
            ])

        self._objective = cvx.Maximize(self.xp)


class MaxProblem2(BaseProblem):
    def __init__(self, problemType, di=None, kwargs=None, X=None, Y=None, initLoss=None, initL1=None, parameters=None, presetModel=None):
        super().__init__(problemType, di=di, kwargs=kwargs, X=X, Y=Y, initLoss=initLoss, initL1=initL1,
                         parameters=parameters,presetModel=presetModel)

        self._constraints.extend(
            [
                self.xp <= -(self.omega[self.di])
            ])

        self._objective = cvx.Maximize(self.xp)


class BaseClassificationProblem(ProblemType):
    def add_type_specific(self, baseProblem):
        # Problem Specific variables and constraints
        baseProblem.b = cvx.Variable(name="offset")  # shift
        point_distances = cvx.multiply(baseProblem.Y, baseProblem.X * baseProblem.omega + baseProblem.b)
        baseProblem.loss = cvx.sum(cvx.pos(1 - point_distances))
        baseProblem.weight_norm = cvx.norm(baseProblem.omega, 1)

        baseProblem._constraints.extend(
            [
                baseProblem.weight_norm <= baseProblem.initL1,
                baseProblem.loss <= baseProblem.initLoss
            ])


class BaseRegressionProblem(ProblemType):

    def add_type_specific(self, baseProblem):
        baseProblem.epsilon = baseProblem.parameters["epsilon"]
        baseProblem.b = cvx.Variable(name="offset")  # offset from origin
        baseProblem.slack = cvx.Variable(shape=(baseProblem.n), nonneg=True)  # slack variables
        baseProblem.loss = cvx.sum(baseProblem.slack)
        baseProblem.weight_norm = cvx.norm(baseProblem.omega, 1)

        baseProblem._constraints.extend([
            baseProblem.weight_norm <= baseProblem.initL1,
            baseProblem.loss <= baseProblem.initLoss,
            cvx.abs(baseProblem.Y - (
                    baseProblem.X * baseProblem.omega + baseProblem.b)) <= baseProblem.epsilon + baseProblem.slack
        ])


# ****************************************************************************
# *                            Ordinal Regression                            *
# ****************************************************************************
# TODO: define problems as cvxpy problem with constraints and objective
class BaseOrdinalRegressionProblem(ProblemType):
    def add_type_specific(self, baseProblem):
        # TODO: anpassen an baseProblem, wir erweitern Min und Maxprobleme um variablen (siehe oben)
        # anstatt self einfach die baseProblem nehmen um Werte zuzuweisen

        # Prepare the same Problem structure as in initial search
        (n, d) = baseProblem.X.shape
        n_bins = len(np.unique(baseProblem.Y))

        baseProblem.b = cvx.Variable(n_bins - 1)
        baseProblem.chi = cvx.Variable(n, nonneg=True)
        baseProblem.xi = cvx.Variable(n, nonneg=True)
        baseProblem.loss = cvx.sum(baseProblem.chi + baseProblem.xi)
        baseProblem.weight_norm = cvx.norm(baseProblem.omega, 1)

        for i in range(n_bins - 1):
            indices = np.where(baseProblem.Y == i)
            baseProblem._constraints.append(baseProblem.X[indices] * baseProblem.omega - baseProblem.chi[indices] <= baseProblem.b[i] - 1)

        for i in range(1, n_bins):
            indices = np.where(baseProblem.Y == i)
            baseProblem._constraints.append(baseProblem.X[indices] * baseProblem.omega + baseProblem.xi[indices] >= baseProblem.b[i - 1] + 1)

        for i in range(n_bins - 2):
            baseProblem._constraints.append(baseProblem.b[i] <= baseProblem.b[i + 1])

        baseProblem._constraints.append(cvx.norm(baseProblem.omega, 1) <= baseProblem.initL1)
        baseProblem._constraints.append(cvx.sum(baseProblem.chi + baseProblem.xi) <= baseProblem.initLoss)

