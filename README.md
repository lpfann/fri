# Feature relevance intervals

[![Build Status](https://travis-ci.org/lpfann/fri.svg?branch=master)](https://travis-ci.org/lpfann/fri)

This repo contains the python implementation of the all-relevant feature selection method described in the corresponding publication[1].

![Example output of method for biomedical dataset](/examples/example_plot.png?raw=true)

## Installation
Before installing the module you will need `numpy` and `scipy`.
We highly recommend the Anaconda Python distribution.
The library was written with Python 3 in mind, we did not test it under Python 2.

To install the module execute:
```shell
$ python setup.py install
```
or 
```shell
pip install fri
```

## Usage

A simple example can be found in the `examples` folder in form of a 'jupyter notebook'.

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
from fri.genData import genData
X,y = genData(n_samples=100, n_features=6,strRel=2, n_redundant=2,
                    n_repeated=0, flip_y=0)

# We created a binary classification set with 6 features of which 2 are strongly relevant and 2 weakly relevant.

# Scale Data
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)

# New object for Classification Data
from fri.fri import FRIClassification
fri = FRIClassification()

# Fit to data
fri.fit(X_scaled,y)

# Print out feature relevance intervals
print(fri.interval_)

# ### Plot results
from fri import plot
plot.plotIntervals(fri.interval_)

# ### Print internal Parameters

print(fri.allrel_prediction_)

# Print out hyperparameter found by GridSearchCV
print(fri._hyper_C)
# Get weights for linear models used for each feature optimization

print(fri._omegas)

```



## References  
>[1] Göpfert C, Pfannschmidt L, Hammer B. Feature Relevance Bounds for Linear Classification. In: Proceedings of the ESANN. 25th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning; Accepted.
https://pub.uni-bielefeld.de/publication/2908201
