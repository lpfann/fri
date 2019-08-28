# Quick start guide


## Installation
### Stable
Fri can be installed via the Python Package Index (PyPI).

If you have `pip` installed just execute the command

    pip install fri
   
to get the newest stable version.

The dependencies should be installed and checked automatically.
If you have problems installing please open issue at our [tracker](https://github.com/lpfann/fri/issues/new).

### Development
To install a bleeding edge dev version of `FRI` you can clone the GitHub repository using

    git clone git@github.com:lpfann/fri.git

and then check out the `dev` branch: `git checkout dev`.

We use [poetry](https://poetry.eustace.io/) for dependency management.

Run

    poetry install

in the cloned repository to install `fri` in a virtualenv.



To check if everything works as intented you can use `pytest` to run the unit tests.
Just run the command

    poetry run pytest

in the main project folder

## Using FRI
Now we showcase the workflow of using FRI on a simple classification problem.

### Data
To have something to work with, we need some data first.
`fri` includes a generation method for binary classification and regression data.

In our case we need some classification data.


```python
from fri import genClassificationData
```

We want to create a small set with a few features.

Because we want to showcase the all-relevant feature selection, we generate multiple strongly and weakly relevant features.


```python
n = 100
features = 6
strongly_relevant = 2
weakly_relevant = 2
```


```python
X,y = genClassificationData(n_samples=n,
                            n_features=features,
                            n_strel=strongly_relevant,
                            n_redundant=weakly_relevant,
                            random_state=123)
```

The method also prints out the parameters again.


```python
X.shape
```




    (100, 6)



We created a binary classification set with 6 features of which 2 are strongly relevant and 2 weakly relevant.

#### Preprocess
Because our method expects mean centered data we need to standardize it first.
This centers the values around 0 and deviation to the standard deviation


```python
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
```

### Model
Now we need to creata a Model. 

We use the `FRI` module.



```python
import fri
```

`fri` provides a convenience class `fri.FRI` to create a model.

`fri.FRI` needs the type of problem as a first argument of type `ProblemName`.

Depending on the Problem you want to analyze pick from one of the available models in `ProblemName`.


```python
list(fri.ProblemName)
```




    [<ProblemName.CLASSIFICATION: <class 'fri.model.classification.Classification'>>,
     <ProblemName.REGRESSION: <class 'fri.model.regression.Regression'>>,
     <ProblemName.ORDINALREGRESSION: <class 'fri.model.ordinal_regression.OrdinalRegression'>>,
     <ProblemName.LUPI_CLASSIFICATION: <class 'fri.model.lupi_classification.LUPI_Classification'>>,
     <ProblemName.LUPI_REGRESSION: <class 'fri.model.lupi_regression.LUPI_Regression'>>,
     <ProblemName.LUPI_ORDREGRESSION: <class 'fri.model.lupi_ordinal_regression.LUPI_OrdinalRegression'>>]



Because we have Classification data we use the `ProblemName.CLASSIFICATION` to instantiate our model.


```python
fri_model = fri.FRI(fri.ProblemName.CLASSIFICATION,slack_loss=0.2,slack_regularization=0.2)
```


```python
fri_model
```




    FRI(n_jobs=1, n_param_search=10, n_probe_features=20, normalize=True,
        problemName=None, random_state=RandomState(MT19937) at 0x7F30AE749C00,
        slack_loss=None, slack_regularization=None, verbose=0)



We used no parameters for creation so the defaults are active.

#### Fitting to data
Now we can just fit the model to the data using `scikit-learn` like commands.


```python
fri_model.fit(X_scaled,y)
```




    FRI(n_jobs=1, n_param_search=10, n_probe_features=20, normalize=True,
        problemName=None, random_state=RandomState(MT19937) at 0x7F30AE749C00,
        slack_loss=None, slack_regularization=None, verbose=0)



The resulting feature relevance bounds are saved in the `interval_` variable.


```python
fri_model.interval_
```




    array([[0.28158379, 0.42206863],
           [0.26824834, 0.41605723],
           [0.        , 0.48783056],
           [0.        , 0.44949121],
           [0.        , 0.04668038],
           [0.        , 0.0604022 ]])



If you want to print out the relevance class use the `print_interval_with_class()` function.


```python
print(fri_model.print_interval_with_class())
```

    ############## Relevance bounds ##############
    feature: [LB -- UB], relevance class
          0: [0.3 -- 0.4], Strong relevant
          1: [0.3 -- 0.4], Strong relevant
          2: [0.0 -- 0.5], Weak relevant
          3: [0.0 -- 0.4], Weak relevant
          4: [0.0 -- 0.0], Irrelevant
          5: [0.0 -- 0.1], Irrelevant
    


The bounds are grouped in 2d sublists for each feature.


To acess the relevance bounds for feature 2 we would use


```python
fri_model.interval_[2]
```




    array([0.        , 0.48783056])



The relevance classes are saved in the corresponding variable `relevance_classes_`:


```python
fri_model.relevance_classes_
```




    array([2, 2, 1, 1, 0, 0])



`2` denotes strongly relevant features, `1` weakly relevant and `0` irrelevant.

#### Plot results

The bounds in numerical form are useful for postprocesing.
If we want a human to look at it, we recommend the plot function `plot_relevance_bars`.

We can also color the bars according to `relevance_classes_`


```python
# Import plot function
from fri.plot import plot_relevance_bars
import matplotlib.pyplot as plt
%matplotlib inline
# Create new figure, where we can put an axis on
fig, ax = plt.subplots(1, 1,figsize=(6,3))
# plot the bars on the axis, colored according to fri
out = plot_relevance_bars(ax,fri_model.interval_,classes=fri_model.relevance_classes_)
```


![png](output_32_0.png)


### Setting constraints manually
Our model also allows to compute relevance bounds when the user sets a given range for the features.

We use a dictionary to encode our constraints.



```python
preset = {}
```

#### Example
As an example, let us constrain the third from our example to the minimum relevance bound.




```python
preset[2] = fri_model.interval_[2, 0]
```

We use the function `constrained_intervals`.

Note: we need to fit the model before we can use this function.
We already did that, so we are fine.


```python
const_ints = fri_model.constrained_intervals(preset=preset)
```


```python
const_ints
```




    array([[3.33537278e-01, 3.98346587e-01],
           [3.16282293e-01, 3.99927655e-01],
           [0.00000000e+00, 0.00000000e+00],
           [3.77824890e-01, 4.49491208e-01],
           [0.00000000e+00, 2.56795434e-02],
           [6.28844147e-11, 4.97955367e-02]])



Feature 3 is set to its minimum (at 0).

How does it look visually?


```python
fig, ax = plt.subplots(1, 1,figsize=(6,3))
out = plot_relevance_bars(ax, const_ints)
```


![png](output_41_0.png)


Feature 3 is reduced to its minimum (no contribution).

In turn, its correlated partner feature 4 had to take its maximum contribution.

### Print internal Parameters

If we want to take at internal parameters, we can use the `verbose` flag in the model creation.


```python
fri_model = fri.FRI(fri.ProblemName.CLASSIFICATION, verbose=True)
```


```python
fri_model.fit(X_scaled,y)
```

    Fitting 3 folds for each of 10 candidates, totalling 30 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done  30 out of  30 | elapsed:    0.3s finished
    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.


    ******************** Best found baseline model ********************
    Classification_SVM(C=6523.18940508926)
    score:  1.0
    'loss: -3.2162506599275367e-12'
    'w_l1: 28.743902865913274'
    'w: shape (6,)'
    'b: shape ()'
    'slack: shape (100,)'
    ******************************


    [Parallel(n_jobs=1)]: Done  18 out of  18 | elapsed:    0.2s finished
    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done  40 out of  40 | elapsed:    0.4s finished
    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.


    FS threshold: -0.3485850797626653-0.37121449904943193, Mean:0.011314709643383315, Std:0.014497743389929868, n_probes 4
    FS threshold: -0.09430000942634031-0.1585322164066591, Mean:0.0321161034901594, Std:0.014735763155338663, n_probes 7


    [Parallel(n_jobs=1)]: Done  20 out of  20 | elapsed:    0.3s finished





    FRI(n_jobs=1, n_param_search=10, n_probe_features=20, normalize=True,
        problemName=None, random_state=RandomState(MT19937) at 0x7F30AE749C00,
        slack_loss=None, slack_regularization=None, verbose=True)



This prints out the parameters of the baseline model

One can also see the best selected hyperparameter according to gridsearch and the training score of the model in `score`.


### Multiprocessing
To enable multiprocessing simply use the `n_jobs` parameter when init. the model.

It expects an integer parameter which defines the amount of processes used.
`n_jobs=-1` uses all available on the CPU.


```python
fri_model = fri.FRI(fri.ProblemName.CLASSIFICATION, n_jobs=-1, verbose=1)
```


```python
fri_model.fit(X_scaled,y)
```

    Fitting 3 folds for each of 10 candidates, totalling 30 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  15 out of  30 | elapsed:    0.1s remaining:    0.1s
    [Parallel(n_jobs=-1)]: Done  30 out of  30 | elapsed:    0.2s finished
    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   3 out of  18 | elapsed:    0.1s remaining:    0.4s
    [Parallel(n_jobs=-1)]: Done  18 out of  18 | elapsed:    0.1s finished
    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.


    ******************** Best found baseline model ********************
    Classification_SVM(C=0.3735574324157715)
    score:  0.9499248120300752
    'loss: 11.49412313137918'
    'w_l1: 4.827039706578124'
    'w: shape (6,)'
    'b: shape ()'
    'slack: shape (100,)'
    ******************************


    [Parallel(n_jobs=-1)]: Done  40 out of  40 | elapsed:    0.6s finished
    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.


    FS threshold: -0.003062137274245256-0.0037609499356378584, Mean:0.00034940633069630115, Std:0.00057365780239467, n_probes 11
    FS threshold: -0.2619964838392958-0.31312919443178205, Mean:0.025566355296243105, Std:0.03838315602969965, n_probes 8


    [Parallel(n_jobs=-1)]: Done  20 out of  20 | elapsed:    0.5s finished





    FRI(n_jobs=-1, n_param_search=10, n_probe_features=20, normalize=True,
        problemName=None, random_state=RandomState(MT19937) at 0x7F30AE749C00,
        slack_loss=None, slack_regularization=None, verbose=1)




```python

```
