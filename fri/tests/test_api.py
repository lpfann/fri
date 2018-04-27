from sklearn.utils.estimator_checks import check_parameters_default_constructible, check_no_fit_attributes_set_in_init

from fri import FRIRegression, FRIClassification


def test_sklearn_api():
    check_parameters_default_constructible("FRIClassification",FRIClassification)
    check_no_fit_attributes_set_in_init("FRIClassification",FRIClassification)
    check_parameters_default_constructible("FRIRegression",FRIRegression)
    check_no_fit_attributes_set_in_init("FRIRegression",FRIRegression)

