.. _Howto BF MATH 010:
Howto BF-MATH-010: Normalizers
==============================

Prerequisites
^^^^^^^^^^^^^

Please install following packages to run this howto

    - `Numpy <https://www.numpy.org>`_



Executable code
^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../../../../../src/mlpro/bf/examples/howto_bf_math_010_normalizers.py
	:language: python



Results
^^^^^^^

The results will be available as follows

.. code-block:: bashh

    01. Parameters updated for the Z transformer


    02. Normalized value(Z transformer):
     [[ 1.04497494 -0.38178705  1.89666962 -1.91464488]
     [ 0.38490458  0.7682956  -1.63116063  1.15923813]
     [-0.20498109  1.85268962  0.09981211 -1.37339696]
     [-0.0100236   0.93645002 -0.10715926 -0.30260282]
     [-1.44410306 -0.24405349  0.20363054 -0.24671404]
     [ 1.93982982 -0.72489859 -0.51710897 -0.17999681]
     [ 0.01727045  0.35396823 -1.29807649 -0.08428727]
     [-0.0907917  -2.0231527   1.30171013  1.0389026 ]
     [-1.69055718 -0.21120917 -0.09837462  0.84888074]
     [ 0.05347684 -0.32630247  0.15005757  1.05462132]]


    03. Denormalized value (Z transformer):
     [[ 45.     -7.     65.    -87.   ]
     [ 21.3    47.1   -41.02   89.   ]
     [  0.12   98.11   11.    -56.01 ]
     [  7.12   55.01    4.78    5.3  ]
     [-44.371  -0.521  14.12    8.5  ]
     [ 77.13  -23.14   -7.54   12.32 ]
     [  8.1    27.61  -31.01   17.8  ]
     [  4.22  -84.21   47.12   82.11 ]
     [-53.22    1.024   5.044  71.23 ]
     [  9.4    -4.39   12.51   83.01 ]]


    04. Parameters updated for the Z transformer



    05. Normalized Data (Z transformer): [[ 0.11994467 -1.47066196  1.74588644 -0.56725513]]



    06. Normalized Data (validation Z transformer):  [[ 0.11994467 -1.47066196  1.74588644 -0.56725513]]



    07. Normalization parameters updated for z-transformer based on replaced data

    08. Normalized Data (Z transformer): [[ 0.16683367 -1.45314853  1.7459814  -0.48625054]]



    09. Normalized Data (validation Z transformer):  [[ 0.16683367 -1.45314853  1.7459814  -0.48625054]]


    10. Parameters updated for the MinMax Normalizer


    11. Normalized value (MinMax Normalizer):
     [1.         0.78947368]


    12. Denormalized value (MinMax Normalizer):
     [19.  8.]


    13. Boundaries updated (MinMax Normalizer)


    14. Parameters updated for MinMax normalizer


    15. Renormalized value (MinMax Normalizer):
     [2.60655738 9.86666667]


    16. Normalized value (Validation renormalization):
     [-0.58667025  0.98222222]



Cross Reference
^^^^^^^^^^^^^^^

    - :ref:`API Reference: Normalizers <target_ap_bf_math_norm>`