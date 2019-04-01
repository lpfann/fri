from abc import ABCMeta, abstractmethod

import numpy as np
from cvxpy import OPTIMAL, OPTIMAL_INACCURATE

import fri.base
from fri.lupi_optproblems import MinProblem_lupi, MaxProblem1_lupi, MaxProblem2_lupi


class Bound_lupi(object):
    __metaclass__ = ABCMeta

    """Class for lower and upper relevance bounds"""

    def __init__(self, optim_dim, X, X_priv, Y, initLoss, initL1, initL1_priv, presetModel, problemClass, kwargs):
        self.optim_dim = optim_dim
        self.X = X
        self.X_priv = X_priv
        self.Y = Y
        self.initL1 = initL1
        self.initL1_priv = initL1_priv
        self.initLoss = initLoss
        self.optim_dim = optim_dim
        self.presetModel = presetModel
        self.acceptableStati = [OPTIMAL, OPTIMAL_INACCURATE]
        self.isUpperBound = None
        self.problemClass = problemClass
        self.kwargs = kwargs

    @abstractmethod
    def solve(self):
        pass

    def __repr__(self):
        return "{self.__class__.__name__}(optim_dim={self.optim_dim}, X.shape={self.X.shape}, Y.shape={self.Y.shape}," \
               " initL1={self.initL1}, initLoss={self.initLoss}, presetModel={self.presetModel})".format(
            self=self)

class LowerBound_lupi(Bound_lupi):
    """Class for lower bounds """

    def __init__(self, problemClass=None, optim_dim=None, kwargs=None, initLoss=None, initL1=None, initL1_priv=None, X=None, X_priv=None, Y=None,
                 presetModel=None):
        # Init Super class, could be used for data manipulation
        super().__init__(optim_dim, X, X_priv, Y, initLoss, initL1, initL1_priv, presetModel,problemClass,kwargs)


        # Init problem instance usually defined in the main class
        self.prob_instance = MinProblem_lupi(problemClass.problemType, di=optim_dim, kwargs=kwargs, X=self.X, X_priv=self.X_priv, Y=self.Y,
                                        initLoss=initLoss, initL1=initL1, initL1_priv=initL1_priv, parameters=problemClass._best_params,
                                        presetModel=presetModel)

        # Define bound type for easier indexing after result collection
        self.isUpperBound = False

    def solve(self):
        status = self.prob_instance.solve().problem.status

        if status in self.acceptableStati:
            return self
        else:
            raise fri.base.NotFeasibleForParameters(status, self)


class UpperBound_lupi(Bound_lupi):
    """Class for Upper bounds """

    def __init__(self, problemClass=None, optim_dim=None, kwargs=None, initLoss=None, initL1=None, initL1_priv=None, X=None, X_priv=None, Y=None,
                 presetModel=None):
        super().__init__(optim_dim, X, X_priv, Y, initLoss, initL1, initL1_priv, presetModel, problemClass, kwargs)

        self.prob_instance1 = MaxProblem1_lupi(problemClass.problemType, di=optim_dim, kwargs=kwargs, X=self.X, X_priv=self.X_priv, Y=self.Y,
                                          initLoss=initLoss, initL1=initL1, initL1_priv=initL1_priv, parameters=problemClass._best_params,
                                          presetModel=presetModel)
        self.prob_instance2 = MaxProblem2_lupi(problemClass.problemType, di=optim_dim, kwargs=kwargs, X=self.X, X_priv=self.X_priv, Y=self.Y,
                                          initLoss=initLoss, initL1=initL1, initL1_priv=initL1_priv, parameters=problemClass._best_params,
                                          presetModel=presetModel)

        self.isUpperBound = True

    def solve(self):
        status = [None, None]
        status[0] = self.prob_instance1.solve()
        status[1] = self.prob_instance2.solve()
        status = list(filter(lambda x: x is not None, status))

        valid_problems = list(filter(lambda x: x.problem.status in self.acceptableStati, status))
        if len(valid_problems) == 0:
            raise fri.base.NotFeasibleForParameters("Upper bound has no feasible problems.", self)

        max_index = np.argmax([np.abs(x.problem.value) for x in valid_problems])
        best_problem = valid_problems[max_index]

        self.prob_instance = best_problem
        return self


class ShadowLowerBound_lupi(LowerBound_lupi):
    """ Class for shadow lower bounds
        Permute the data to get bounds for random data.
    """

    def __init__(self, problemClass=None, optim_dim=None, kwargs=None, initLoss=None, initL1=None, initL1_priv=None, X=None, X_priv=None, Y=None,
                 sampleNum=None, presetModel=None):
        # Seed random state with permutation sample number given by parent
        random_state = np.random.RandomState(optim_dim + sampleNum)
        # Permute dimension optim_dim
        X_copy = np.copy(X)
        perm_dim = random_state.permutation(X_copy[:, optim_dim])

        X_copy[:, optim_dim] = perm_dim

        #TODO: check out what's supposed to happen with X_priv
        # Optimize for the first (random permutated) column
        super().__init__(problemClass=problemClass, optim_dim=optim_dim, kwargs=kwargs, initLoss=initLoss, initL1=initL1,
                         initL1_priv=initL1_priv, X=X_copy, X_priv=X_priv, Y=Y, presetModel=presetModel)
        self.isShadow = True

    def relax_problem(self,factor=0.01):
        L1 = self.initL1 * (1+factor)
        loss = self.initLoss * (1+factor)
        super().__init__(self.problemClass, self.optim_dim, self.kwargs, loss, L1, self.X, self.Y, presetModel=self.presetModel)

    def solve(self):
        status = self.prob_instance.solve().problem.status

        if status in self.acceptableStati:
            self.shadow_value = self.prob_instance.problem.value
            return self
        else:
            self.relax_problem()
            status = self.prob_instance.solve().problem.status

            if status in self.acceptableStati:
                self.shadow_value = self.prob_instance.problem.value
            else:
                self.shadow_value = 0

        return self

class ShadowUpperBound_lupi(UpperBound_lupi):
    """ Class for shadow upper bounds
        Permute the data to get bounds for random data.
    """

    def __init__(self, problemClass=None, optim_dim=None, kwargs=None, initLoss=None, initL1=None, initL1_priv=None, X=None, X_priv=None, Y=None,
                 sampleNum=None, presetModel=None):
        # Seed random state with permutation sample number given by parent
        random_state = np.random.RandomState(optim_dim + sampleNum)
        # Permute dimension optim_dim
        X_copy = np.copy(X)
        perm_dim = random_state.permutation(X_copy[:, optim_dim])

        X_copy[:, optim_dim] = perm_dim
        #TODO: check out what's supposed to happend with X_priv
        # Optimize for the first (random permutated) column
        super().__init__(problemClass=problemClass, optim_dim=optim_dim, kwargs=kwargs, initLoss=initLoss, initL1=initL1,
                         initL1_priv=initL1_priv, X=X_copy, X_priv=X_priv, Y=Y, presetModel=presetModel)
        self.isShadow = True

    def relax_problem(self,factor=0.01):
        L1 = self.initL1 * (1+factor)
        loss = self.initLoss * (1+factor)
        super().__init__(self.problemClass, self.optim_dim, self.kwargs, loss, L1, self.X, self.Y, presetModel=self.presetModel)

    def solve(self):
        status = [None, None]
        status[0] = self.prob_instance1.solve()
        status[1] = self.prob_instance2.solve()
        status = list(filter(lambda x: x is not None, status))

        valid_problems = list(filter(lambda x: x.problem.status in self.acceptableStati, status))
        if len(valid_problems) == 0:
            self.relax_problem()

            status = [None, None]
            status[0] = self.prob_instance1.solve()
            status[1] = self.prob_instance2.solve()
            status = list(filter(lambda x: x is not None, status))
            valid_problems = list(filter(lambda x: x.problem.status in self.acceptableStati, status))

            # Problem still infeasible, strongly relevant feature, retrun 0 value to denote need of this variable
            if len(valid_problems) == 0:
                self.shadow_value = 0
                return self

        max_index = np.argmax([np.abs(x.problem.value) for x in valid_problems])
        best_problem = valid_problems[max_index]

        self.prob_instance = best_problem
        self.shadow_value = self.prob_instance.problem.value
        return self
