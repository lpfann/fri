# Feature Relevance Intervals - FRI

[![Build Status](https://travis-ci.org/lpfann/fri.svg?branch=master)](https://travis-ci.org/lpfann/fri)
[![Coverage Status](https://coveralls.io/repos/github/lpfann/fri/badge.svg)](https://coveralls.io/github/lpfann/fri)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1188749.svg)](https://doi.org/10.5281/zenodo.1188749)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lpfann/fri/blob/master/doc/source/notebooks/Guide.ipynb)

![Example output of method for biomedical dataset](doc/source/examples/example_plot.png?raw=true)
This repository contains the Python implementation of the Feature Relevance Intervals method (FRI) also called Feature Relevance Bounds.

## Documentation
Check out our online documentation at [fri.lpfann.me](https://fri.lpfann.me).
There you can find a quick start guide and more background information.

You can also run the guide directly without setup online [here](https://colab.research.google.com/github/lpfann/fri/blob/master/doc/source/notebooks/Guide.ipynb).


## Installation

If you just want to use the __stable__ version from `PyPI` use
```shell
$ pip install fri
```

To install the __development__ version clone the repo and execute:
```shell
$ pip install -e .
```
## Usage
Please refer to the [documentation](https://fri.lpfann.me) for advice.

### Testing
To test if the library was installed correctly you can use the `pytest` command to run all included tests.

```shell
$ pip install pytest
```
then simply run
```shell
$ pytest
```


## References  

>[1] Göpfert C, Pfannschmidt L, Hammer B. Feature Relevance Bounds for Linear Classification. In: Proceedings of the ESANN. 25th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning;
https://pub.uni-bielefeld.de/publication/2908201

>[2] Göpfert C, Pfannschmidt L, Göpfert JP, Hammer B. Interpretation of Linear Classifiers by Means of Feature Relevance Bounds. Neurocomputing.
https://pub.uni-bielefeld.de/publication/2915273


>[3] Lukas Pfannschmidt, Jonathan Jakob, Michael Biehl, Peter Tino, Barbara Hammer: Feature Relevance Bounds for Ordinal Regression
. Proceedings of the ESANN. 27th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning; Accepted.
https://pub.uni-bielefeld.de/record/2933893

>[4] Pfannschmidt L, Göpfert C, Neumann U, Heider D, Hammer B: FRI - Feature Relevance Intervals for Interpretable and Interactive Data Exploration. Presented at the 16th IEEE International Conference on Computational Intelligence in Bioinformatics and Computational Biology, Certosa di Pontignano, Siena - Tuscany, Italy. https://ieeexplore.ieee.org/document/8791489

