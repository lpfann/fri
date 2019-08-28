Welcome to the documentation of FRI!
=====================================

FRI is a Python 3 package for analytical feature selection purposes.
Check out the :ref:`Quick Start Guide </notebooks/Guide.ipynb>` for a practical introduction.
For theoretical explanations look at the `Background`_ or the `References`_ for the long version.

The source code can be found at `GitHub <https://github.com/lpfann/fri>`_.
If you have any problems or suggestions open an issue!


Contents
---------

.. toctree::
  :maxdepth: 3
  :titlesonly:

  notebooks/Guide.ipynb
  auto_examples/index
  api/user


Background
----------
Feature selection is the task of finding relevant features used in a machine learning model.
Often used for this task are models which produce a sparse subset of all input features by permitting the use of additional features (e.g. Lasso with L1 regularization).
But these models are often tuned to filter out redundancies in the input set and produce only an unstable solution especially in the presence of higher dimensional data.

FRI calculates relevance bound values for all input features.
These bounds give rise to intervals which we named 'feature relevance intervals' (FRI).
A given interval symbolizes the allowed contribution each feature has, when it is allowed to be maximized and minimized independently from the others.
This allows us to approximate the global solution instead of relying on the local solutions of the alternatives.

.. figure:: relevancebars.png

  Example plot showing relevance intervals for datasets with 6 features.

With these we can classify features into three classes:
  * **Strongly relevant**: features which are crucial for model performance
  * **Weakly relevant**: features which are important but can be substituted by another weakly relevant feature
  * **Irrelevant**: features which have no association with the target variable

.. figure:: relevancebars_classes_edit.png

  Relevance intervals colored according to their feature classs. Red denotes strongly relevant features, green are weakly relevant and blue are irrelevant.

Installation
------------
FRI is available on PyPI and can be installed via ``pip``::

  pip install fri

All dependencies should be installed automatically if not already present.

License
-------
FRI is licensed under the `MIT License`_ 

.. _MIT License: https://github.com/lpfann/fri/blob/master/LICENSE

References
----------
  1. Göpfert C, Pfannschmidt L, Hammer B. Feature Relevance Bounds for Linear Classification. In: Proceedings of the ESANN. 25th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning;  https://pub.uni-bielefeld.de/publication/2908201
  2. Göpfert C, Pfannschmidt L, Göpfert JP, Hammer B. Interpretation of Linear Classifiers by Means of Feature Relevance Bounds. Neurocomputing. https://pub.uni-bielefeld.de/publication/2915273
  3. Pfannschmidt L., Jakob J., Biehl M., Tino P., Hammer B.: Feature Relevance Bounds for Ordinal Regression . Proceedings of the ESANN. 27th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning; https://pub.uni-bielefeld.de/record/2933893





Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

