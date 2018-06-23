# Feature relevance intervals

[![Build Status](https://travis-ci.org/lpfann/fri.svg?branch=master)](https://travis-ci.org/lpfann/fri)
[![CircleCI](https://circleci.com/gh/lpfann/fri/tree/master.svg?style=svg)](https://circleci.com/gh/lpfann/fri/tree/master)
[![Coverage Status](https://coveralls.io/repos/github/lpfann/fri/badge.svg)](https://coveralls.io/github/lpfann/fri)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/lpfann/fri/master?filepath=notebooks)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1188750.svg)](https://doi.org/10.5281/zenodo.1188750)

This repo contains the python implementation of the all-relevant feature selection method described in the corresponding publications[1,2].

Try out the online demo notebook [here](https://mybinder.org/v2/gh/lpfann/fri/master?filepath=notebooks).

![Example output of method for biomedical dataset](/examples/example_plot.png?raw=true)

## Installation
The library needs various dependencies which should automatically be installed.
We highly recommend the [Anaconda](https://www.anaconda.com/download) Python distribution to provide all dependencies.
The library was written with Python 3 in mind and due to the foreseeable ending of Python 2 support, backwards compatibility is not planned.

If you just want to use the __stable__ version from PyPi use
```shell
$ pip install fri
```

To install the module in __development__ clone the repo and execute:
```shell
$ python setup.py install
```

## Testing
To test if the library was installed correctly you can use the `pytest` command to run all included tests.

```shell
$ pip install pytest
```
then run in the root directory:
```shell
$ pytest
```

## Usage
Examples and API descriptions can be found [here](https://lpfann.github.io/fri/).

In general, the library follows the sklearn API format.
The two important classes exposed to the user are
``` 
FRIClassification
```
and
```
FRIRegression
```
depending on your data type.


## Parameters ##

__C__ : float, optional
   > Set a fixed regularization parameter.
   > If None, value will automatically be determined using GridSearch.

__random_state__ :  int seed, RandomState instance, or None (default=None)
   >The seed of the pseudo random number generator to use when shuffling the data.

__shadow_features__ : boolean, default = True
   > Use shuffled contrast features for each real feature as a baseline
   > correction. 
   > Each feature gets shuffled independently and 
   > feature relevance bounds computed on these contrast features.
   > Leads to a more sparse output but still has some 
   > ploblems with very sparse binary features, which lead to a better 
   > "random" distribution. 
   > Increase __n_resampling__ when having problems with this.

__parallel__ : boolean, default = False
   > Uses multiprocessing with all available cores when enabled
   > to compute relevance bounds in parallel.

__n_resampling__ : int, default = 3
  > Number of contrast features which get computed per features.
  > Results are averaged to reduce problems on some sparse input features.

### Regression specific

__epsilon__ : float, optional
   > Controls size of epsilon tube around initial SVR Model.
   > By default, value is set using hyperparameter optimization.
   > 


## Attributes ##

**n_features_** : int
   > The number of selected features.

**allrel_prediction_** : array of shape [n_features]
   > The mask of selected features. Includes all relevant ones.

**ranking_** : array of shape [n_features]
  >  The feature ranking, such that ``ranking_[i]`` corresponds to the
  >  ranking position of the i-th feature. Selected (i.e., estimated
  >  best) features are assigned rank 1 and tentative features are assigned
  >  rank 2.


## Examples ##

```python
# ## Classification data
from fri import genClassificationData
X,y = genClassificationData(n_samples=100, n_features=6,n_strel=2, n_redundant=2,
                    n_repeated=0, flip_y=0)

# We created a binary classification set with 6 features of which 2 are strongly relevant and 2 weakly relevant.

# Scale Data
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)

# New object for Classification Data
from fri import FRIClassification
fri_model = FRIClassification()

# Fit to data
fri_model.fit(X_scaled,y)

# Print out feature relevance intervals
print(fri_model.interval_)

# ### Plot results
from fri import plot
plot.plotIntervals(fri_model.interval_)

# ### Print internal Parameters

print(fri_model.allrel_prediction_)

# Print out hyperparameter found by GridSearchCV
print(fri_model._hyper_C)
# Get weights for linear models used for each feature optimization

print(fri_model._omegas)

```



## References  
>[1] Göpfert C, Pfannschmidt L, Hammer B. Feature Relevance Bounds for Linear Classification. In: Proceedings of the ESANN. 25th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning; Accepted.
https://pub.uni-bielefeld.de/publication/2908201

>[2] Göpfert C, Pfannschmidt L, Göpfert JP, Hammer B. Interpretation of Linear Classifiers by Means of Feature Relevance Bounds. Neurocomputing. Accepted.
https://pub.uni-bielefeld.de/publication/2915273
