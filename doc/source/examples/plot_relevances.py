"""
========================================================
Print out relevances in command line
========================================================
`print_interval_with_class` allows to print out relevances and class for debug purposes.
"""

from toydata.gen_data import genRegressionData

X, y = genRegressionData(
    n_samples=100,
    n_features=6,
    n_strel=2,
    n_redundant=2,
    n_repeated=0,
    random_state=123,
)

from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)

from fri import FRI, ProblemName

fri_model = FRI(ProblemName.REGRESSION)
fri_model.fit(X_scaled, y)

output = fri_model.print_interval_with_class()
print(output)
