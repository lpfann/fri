import logging

# noinspection PyUnresolvedReferences
from fri._version import __version__

logging.basicConfig(level=logging.INFO)
from enum import Enum
import fri.model


class ProblemName(Enum):
    """
    Enum which contains usable models for which feature relevance intervals can be computed in :func:`~FRI`.
    """

    CLASSIFICATION = fri.model.Classification
    REGRESSION = fri.model.Regression
    ORDINALREGRESSION = fri.model.OrdinalRegression
    LUPI_CLASSIFICATION = fri.model.LUPI_Classification
    LUPI_REGRESSION = fri.model.LUPI_Regression
    LUPI_ORDREGRESSION = fri.model.LUPI_OrdinalRegression


NORMAL_MODELS = [
    ProblemName.CLASSIFICATION,
    ProblemName.REGRESSION,
    ProblemName.ORDINALREGRESSION,
]
LUPI_MODELS = [
    ProblemName.LUPI_CLASSIFICATION,
    ProblemName.LUPI_REGRESSION,
    ProblemName.LUPI_ORDREGRESSION,
]

from fri.toydata import (
    genRegressionData,
    genClassificationData,
    genOrdinalRegressionData,
    quick_generate,
    genLupiData,
)
from fri.main import FRIBase
from fri.plot import plot_intervals


class FRI(FRIBase):
    def __init__(
        self,
        problemName: object,
        random_state: object = None,
        n_jobs: object = 1,
        verbose: object = 0,
        n_param_search: object = 10,
        n_probe_features: object = 20,
        slack_regularization: object = 0.001,
        slack_loss: object = 0.001,
        normalize: object = True,
        **kwargs,
    ):
        """
        Main class to use `FRI` in programattic fashion following the scikit-learn paradigm.

        Parameters
        ----------
        problemName: `ProblemName` or str
            Type of Problem as enum value or explicit string (e.g. "classification")
        random_state: object or int
            Random state object or int
        n_jobs: int or None
            Number of threads or -1 for automatic.
        verbose: int
            Verbosity if > 0
        n_param_search: int
            Number of parameter samples in random search for hyperparameters.
        n_probe_features: int
            Number of probes to generate to improve feature selection.
        slack_regularization: float
            Allow deviation from optimal L1 norm.
        slack_loss: float
            Allow deviation of loss.
        normalize: boolean
            Normalize relevace bounds to range of [0,1] depending on L1 norm.

        """
        if isinstance(problemName, ProblemName):
            problemtype = problemName.value
        else:
            if problemName == "classification" or problemName == "class":
                problemtype = ProblemName.CLASSIFICATION
            elif problemName == "regression" or problemName == "reg":
                problemtype = ProblemName.REGRESSION
            elif problemName == "ordinalregression" or problemName == "ordreg":
                problemtype = ProblemName.ORDINALREGRESSION
            elif problemName == "lupi_classification" or problemName == "lupi_class":
                problemtype = ProblemName.LUPI_CLASSIFICATION

        if problemtype is None:
            names = [enum.name.lower() for enum in ProblemName]
            print(
                f"Parameter 'problemName' was not recognized or unset. Try one of {names}."
            )
        else:
            super().__init__(
                problemtype,
                random_state=random_state,
                n_jobs=n_jobs,
                verbose=verbose,
                n_param_search=n_param_search,
                n_probe_features=n_probe_features,
                w_l1_slack=slack_regularization,
                loss_slack=slack_loss,
                normalize=normalize,
                **kwargs,
            )


__all__ = [
    "genRegressionData",
    "genClassificationData",
    "genOrdinalRegressionData",
    "quick_generate",
    "plot_intervals",
    "ProblemName",
    "FRI",
    "LUPI_MODELS",
    "NORMAL_MODELS",
    "genLupiData",
]
