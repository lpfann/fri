# Feature relevance intervals

[![Build Status](https://travis-ci.org/lpfann/fri.svg?branch=master)](https://travis-ci.org/lpfann/fri)
[![Coverage Status](https://coveralls.io/repos/github/lpfann/fri/badge.svg)](https://coveralls.io/github/lpfann/fri)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1188749.svg)](https://doi.org/10.5281/zenodo.1188749)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lpfann/fri/blob/master/doc/source/notebooks/Guide.ipynb)

This repository contains the Python implementation of the Feature Relevance Intervals method (FRI)[1,2].

Check out our online documentation [here](https://lpfann.github.io/fri/).
There you can find a quick start guide and more background information.
You can also run the guide directly [in Colab](https://colab.research.google.com/github/lpfann/fri/blob/master/doc/source/notebooks/Guide.ipynb).

![Example output of method for biomedical dataset](doc/source/examples/example_plot.png?raw=true)

## Installation
The library needs various dependencies which should automatically be installed.
We highly recommend the [Anaconda](https://www.anaconda.com/download) Python distribution to provide all dependencies.
The library was written with Python 3 in mind and due to the foreseeable ending of Python 2 support, backwards compatibility is not planned.

If you just want to use the __stable__ version from PyPI use
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



## References  
>[1] Göpfert C, Pfannschmidt L, Hammer B. Feature Relevance Bounds for Linear Classification. In: Proceedings of the ESANN. 25th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning;
https://pub.uni-bielefeld.de/publication/2908201

>[2] Göpfert C, Pfannschmidt L, Göpfert JP, Hammer B. Interpretation of Linear Classifiers by Means of Feature Relevance Bounds. Neurocomputing.
https://pub.uni-bielefeld.de/publication/2915273


>[3] Lukas Pfannschmidt, Jonathan Jakob, Michael Biehl, Peter Tino, Barbara Hammer: Feature Relevance Bounds for Ordinal Regression
. Proceedings of the ESANN. 27th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning; Accepted.
https://pub.uni-bielefeld.de/record/2933893

