import numpy as np

from fri import ProblemName
from .gen_data import genRegressionData, genClassificationData, genOrdinalRegressionData
from .gen_lupi import genLupiData

__all__ = [
    "genRegressionData",
    "genClassificationData",
    "genOrdinalRegressionData",
    "genLupiData",
]


def quick_generate(problem: object, **kwargs) -> [np.ndarray, np.ndarray]:
    """
    Method to wrap individual data generation functions.
    Allows passing `problem` as a string such as "classification" or `ProblemName` object of the corresponding type.
    For possible kwargs see `genClassificationData' or `genLupiData`.

    Parameters
    ----------
    problem : str or `ProblemName`
        Type of data to generate (e.g. "classification" or `ProblemName.CLASSIFICATION`
    kwargs : **dict
        arguments to pass to the generation functions

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
    """
    if problem is "regression" or problem is ProblemName.REGRESSION:
        gen = genRegressionData
    elif problem is "classification" or problem is ProblemName.CLASSIFICATION:
        gen = genClassificationData
    elif problem is "ordreg" or problem is ProblemName.ORDINALREGRESSION:
        gen = genOrdinalRegressionData
    elif problem is "lupi_regression" or problem is ProblemName.LUPI_REGRESSION:
        gen = genLupiData
        kwargs["problemName"] = ProblemName.LUPI_REGRESSION
    elif problem is "lupi_classification" or problem is ProblemName.LUPI_CLASSIFICATION:
        gen = genLupiData
        kwargs["problemName"] = ProblemName.LUPI_CLASSIFICATION
    elif problem is "lupi_ordreg" or problem is ProblemName.LUPI_ORDREGRESSION:
        gen = genLupiData
        kwargs["problemName"] = ProblemName.LUPI_ORDREGRESSION
    else:
        raise ValueError(
            "Unknown problem type. Try 'regression', 'classification' or 'ordreg' and/or add 'lupi_' prefix"
        )
    return gen(**kwargs)
