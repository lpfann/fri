import logging

# noinspection PyUnresolvedReferences
from fri._version import __version__

logging.basicConfig(level=logging.INFO)
from enum import Enum
import fri.model

import arfs_gen


class ProblemName(Enum):
    """
    Enum which contains usable models for which feature relevance intervals can be computed in :func:`~FRI`.
    Values of enums contains class of model and data generation method found in external library `arfs_gen`.
    """

    CLASSIFICATION = [fri.model.Classification, arfs_gen.ProblemName.CLASSIFICATION]
    REGRESSION = [fri.model.Regression, arfs_gen.ProblemName.REGRESSION]
    ORDINALREGRESSION = [
        fri.model.OrdinalRegression,
        arfs_gen.ProblemName.ORDINALREGRESSION,
    ]
    LUPI_CLASSIFICATION = [
        fri.model.LUPI_Classification,
        arfs_gen.ProblemName.LUPI_CLASSIFICATION,
    ]
    LUPI_REGRESSION = [fri.model.LUPI_Regression, arfs_gen.ProblemName.LUPI_REGRESSION]
    LUPI_ORDREGRESSION = [
        fri.model.LUPI_OrdinalRegression,
        arfs_gen.ProblemName.LUPI_ORDREGRESSION,
    ]


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
from arfs_gen import genRegressionData, genClassificationData, genOrdinalRegressionData


def quick_generate(problemtype, **kwargs):
    "Overwrite arfs_gen method to handle different format of problemtype in fri"
    return arfs_gen.quick_generate(problemtype.value[1], **kwargs)


def genLupiData(problemname, **kwargs):
    "Overwrite arfs_gen method to handle different format of problemtype in fri"
    return arfs_gen.genLupiData(problemname.value[1], **kwargs)


from fri.main import FRIBase
from fri.plot import plot_intervals


class FRI(FRIBase):
    def __init__(
        self,
        problemName: object,
        random_state: object = None,
        n_jobs: int = 1,
        verbose: int = 0,
        n_param_search: int = 10,
        n_probe_features: int = 20,
        w_l1_slack: float = 0.001,
        loss_slack: float = 0.001,
        normalize: bool = True,
        **kwargs,
    ):
        """
        Main class to use `FRI` in programatic fashion following the scikit-learn paradigm.

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
        w_l1_slack: float
            Allow deviation from optimal L1 norm.
        loss_slack: float
            Allow deviation of loss.
        normalize: boolean
            Normalize relevace bounds to range of [0,1] depending on L1 norm.

        """
        self.problemName = problemName

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
            problem_class = problemtype[0]
            super().__init__(
                problem_class,
                random_state=random_state,
                n_jobs=n_jobs,
                verbose=verbose,
                n_param_search=n_param_search,
                n_probe_features=n_probe_features,
                w_l1_slack=w_l1_slack,
                loss_slack=loss_slack,
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
