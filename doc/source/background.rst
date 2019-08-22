Background
----------
.. note::
    We presented `FRI` at the CIBCB conference. Check out the :download:`slides <https://lpfann.me/talk/cibc19/talk.pdf>` for a short primer into how it works.

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


.. include:: references.rst