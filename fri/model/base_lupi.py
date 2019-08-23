from abc import abstractmethod

from .base_cvxproblem import Relevance_CVXProblem


class LUPI_Relevance_CVXProblem(Relevance_CVXProblem):
    def __init__(
        self,
        current_feature: int,
        data: tuple,
        hyperparameters,
        best_model_constraints,
        preset_model=None,
        best_model_state=None,
        probeID=-1,
    ) -> None:
        super().__init__(
            current_feature,
            data,
            hyperparameters,
            best_model_constraints,
            preset_model,
            best_model_state,
            probeID,
        )

    def preprocessing_data(self, data, best_model_state):
        lupi_features = best_model_state["lupi_features"]
        X_combined, y = data
        X, X_priv = split_dataset(X_combined, lupi_features)
        self.X_priv = X_priv
        super().preprocessing_data((X, y), best_model_state)

        assert lupi_features == X_priv.shape[1]
        self.d_priv = lupi_features
        # LUPI model, we need to offset the index
        self.lupi_index = self.current_feature - self.d
        if self.lupi_index >= 0:
            self.isPriv = True
        else:
            self.isPriv = False

    def init_objective_UB(self, **kwargs):
        # We have two models basically with different indexes
        if self.isPriv:
            self._init_objective_UB_LUPI(**kwargs)
        else:
            # We call sibling class of our lupi class, which is the normal problem
            super().init_objective_UB(**kwargs)

    def init_objective_LB(self, **kwargs):
        # We have two models basically with different indexes
        if self.isPriv:
            self._init_objective_LB_LUPI(**kwargs)
        else:
            # We call sibling class of our lupi class, which is the normal problem
            super().init_objective_LB(**kwargs)

    @abstractmethod
    def _init_objective_LB_LUPI(self, **kwargs):
        pass

    @abstractmethod
    def _init_objective_UB_LUPI(self, **kwargs):
        pass


def split_dataset(X_combined, lupi_features):
    assert X_combined.shape[1] > lupi_features
    X = X_combined[:, :-lupi_features]
    X_priv = X_combined[:, -lupi_features:]
    return X, X_priv


def is_lupi_feature(di, data, best_model_state):
    lupi_features = best_model_state["lupi_features"]
    X_combined, _ = data
    d = X_combined.shape[1] - lupi_features
    lupi_index = di - d
    return lupi_index >= 0
