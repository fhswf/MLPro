.. _Howto BF 5:
`Howto 05 - (ML) Hyperparameters setup <https://github.com/fhswf/MLPro/blob/main/examples/bf/Howto%2005%20-%20(ML)%20Hyperparameters%20setup.py>`_
================
Ver. 1.0.1 (2021-12-10)

This module demonstrates how to set-up hyperparameters using available HyperParamTuple, 
HyperParamSpace, and HyperParam classes.

Prerequisites
`````````````````

Please install the following packages to run this examples properly:
    - :ref:`MLPro <Installation>`
    - `NumPy <https://pypi.org/project/numpy/>`_
    - `Matplotlib <https://pypi.org/project/matplotlib/>`_
    
Results
`````````````````

.. code-block:: bash

    Variable with ID b7ff579e-3446-4056-92d1-1e1dad99e47a = 100.00
    Variable with ID 6709ae31-811b-4fd7-b9ec-0b937068972e = 0.04
    Variable with ID 63c1bb9e-1785-4f62-ac66-ead2b19f08e1 = 0.00
    Variable with ID 168c1ae0-db10-4e9d-9903-2ac8bf5b2875 = 100000.00
    Variable with ID a6e219cf-2fbd-493d-a362-99e7356ada15 = 100.00
    Variable with ID 7a859b01-464a-472d-acdc-c6fcb733c6e6 = 256.00

    A new value for variable ID ids_[0]
    Variable with ID ids_[0] = 50.00

The variable myParameter will be created as a Hyperparameter.


Example Code
`````````````````

.. literalinclude:: ../../../../../../examples/bf/Howto 05 - (ML) Hyperparameters setup.py
    :language: python

  