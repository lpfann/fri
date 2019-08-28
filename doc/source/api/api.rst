Scikit-learn compatible
=======================
Before using `FRI` you most likely know which problem you are going to analyse.
Depending on that use the corresponding value from `ProblemName`.
In the case of a classification problem we would use `ProblemName.CLASSIFICATION`.
If we have a problem with privileged information (`LUPI`) present we would use `ProblemName.LUPI_CLASSIFICATION`.


.. autoclass:: fri.ProblemName
    :members:
    :undoc-members:

.. autoclass:: fri.FRI
    :members:
    :inherited-members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: fri.main.FRIBase
    :members:
    :inherited-members:
    :undoc-members:
    :show-inheritance: