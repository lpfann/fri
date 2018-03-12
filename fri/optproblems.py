import abc

import cvxpy as cvx

#
# Minimal loss in constraints to mitigate numerical instabilities for solvers
#
MINLOSS = 0.01

class BaseProblem(object):
    def __init__(self, problemType, di=None, kwargs=None, X=None, Y=None, initLoss=None, initL1=None, parameters=None, presetModel=None):
        # General data parameters
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.X = X
        self.Y = Y

        # Solver Variables
        self.xp = cvx.Variable(nonneg=True)  # x' , our opt. value
        self.omega = cvx.Variable(self.d)  # complete linear weight vector

        # InitModel Parameters
        self.initL1 = initL1
        self.initLoss = max(MINLOSS, initLoss)
        self.parameters = parameters
        # Dimension specific data
        self.di = di
        # Solver parameters
        self._constraints = []
        self.omega = cvx.Variable(self.d)  # complete linear weight vector

        # Check if some model values have a pre fixed value
        if presetModel is not None:
            assert len(
                presetModel) == self.d  # only assert length, # TODO: add check for two dimensionality when allow ranges of constraints
            for dim in range(self.d):
                assert presetModel[dim] < self.initL1  # a weight bigger than the optimal L1 makes no sense
                # Skip current dimension
                if dim == di:
                    continue
                if presetModel[dim] >= 0:  # Negative elements are considered unset/free
                    self._constraints.extend([
                        cvx.abs(self.omega[dim]) <= presetModel[dim]
                    ])
        self.presetModel = presetModel



        # Solver parameters
        if kwargs is None:
            self.kwargs = {}  # Fix (unpacking)error when manually calling bounds without keyword arguments to the solver
        else:
            self.kwargs = kwargs

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
        except Exception as e:
            print(e)
            return None

        return self


class ProblemType(object, metaclass=abc.ABCMeta):
    # Decorator class to add problem type specific constraints and variables to the BaseProblem
    @abc.abstractmethod
    def add_type_specific(self, baseProblem):
        pass


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
        baseProblem.b = cvx.Variable()  # shift
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
        baseProblem.b = cvx.Variable()  # offset from origin
        baseProblem.slack = cvx.Variable(shape=(baseProblem.n), nonneg=True)  # slack variables
        baseProblem.loss = cvx.sum(baseProblem.slack)
        baseProblem.weight_norm = cvx.norm(baseProblem.omega, 1)

        baseProblem._constraints.extend([
            baseProblem.weight_norm <= baseProblem.initL1,
            baseProblem.loss <= baseProblem.initLoss,
            cvx.abs(baseProblem.Y - (
                    baseProblem.X * baseProblem.omega + baseProblem.b)) <= baseProblem.epsilon + baseProblem.slack
        ])
