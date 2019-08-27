"""
========================================================
Quick plotting without explicit matplotlib
========================================================

An example plot of `FRI` output on classification data
"""

from fri import genClassificationData

X, y = genClassificationData(
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

fri_model = FRI(ProblemName.CLASSIFICATION)
fri_model.fit(X_scaled, y)

from fri.plot import plot_intervals

plot_intervals(fri_model)
