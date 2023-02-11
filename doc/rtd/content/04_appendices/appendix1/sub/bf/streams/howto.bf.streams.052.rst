.. _Howto BF STREAMS 052:
Howto BF-STREAMS-052: Accessing Data from Scikit-Learn
======================================================

.. automodule:: mlpro.bf.examples.howto_bf_streams_052_accessing_data_from_scikitlearn


**Prerequisites**


Please install the following packages to run this examples properly:
    - `Scikit-Learn <https://pypi.org/project/sklearn/>`_
    - `Numpy <https://pypi.org/project/numpy/>`_

**Executable code**

.. literalinclude:: ../../../../../../../../src/mlpro/bf/examples/howto_bf_streams_052_accessing_data_from_scikitlearn.py
	:language: python


**Results**

.. code-block:: bashh

    2023-02-11  22:51:43.599295  I  Wrapper "Stream Provider Scikit-learn": Instantiated
    2023-02-11  22:51:43.897836  I  Wrapper "Stream Provider Scikit-learn": Wrapped package scikit-learn installed in version 1.0.2
    2023-02-11  22:51:43.897836  I  Wrapper "Stream Provider Scikit-learn": Getting list of streams...
    2023-02-11  22:51:43.897836  I  Wrapper "Stream Provider Scikit-learn": Number of streams found: 10
    2023-02-11  22:51:43.897836  I  Wrapper "Stream Provider Scikit-learn": Name of requested stream: iris
    2023-02-11  22:51:43.897836  I  Wrapper "Stream Provider Scikit-learn": Getting list of streams...
    2023-02-11  22:51:43.897836  I  Wrapper "Stream Provider Scikit-learn": Number of streams found: 20
    2023-02-11  22:51:43.897836  I  Stream "Sklearn stream "iris"": Ready to access in mode 0
    2023-02-11  22:51:43.897836  I  Wrapper "Stream Provider Scikit-learn": Number of features in the stream: 4


    2023-02-11  22:51:43.897836  I  Stream "Sklearn stream "iris"": Reset
    2023-02-11  22:51:43.897836  W  Stream "Sklearn stream "iris"": Fetching first 10 stream instances...
    2023-02-11  22:51:43.897836  I  Stream "Sklearn stream "iris"": Instance 0:
       Data: [5.1 3.5 1.4 0.2] ...
       Label: [0]
    2023-02-11  22:51:43.913475  I  Stream "Sklearn stream "iris"": Instance 1:
       Data: [4.9 3.  1.4 0.2] ...
       Label: [0]
    2023-02-11  22:51:43.913475  I  Stream "Sklearn stream "iris"": Instance 2:
       Data: [4.7 3.2 1.3 0.2] ...
       Label: [0]
    2023-02-11  22:51:43.913475  I  Stream "Sklearn stream "iris"": Instance 3:
       Data: [4.6 3.1 1.5 0.2] ...
       Label: [0]
    2023-02-11  22:51:43.913475  I  Stream "Sklearn stream "iris"": Instance 4:
       Data: [5.  3.6 1.4 0.2] ...
       Label: [0]
    2023-02-11  22:51:43.913475  I  Stream "Sklearn stream "iris"": Instance 5:
       Data: [5.4 3.9 1.7 0.4] ...
       Label: [0]
    2023-02-11  22:51:43.913475  I  Stream "Sklearn stream "iris"": Instance 6:
       Data: [4.6 3.4 1.4 0.3] ...
       Label: [0]
    2023-02-11  22:51:43.913475  I  Stream "Sklearn stream "iris"": Instance 7:
       Data: [5.  3.4 1.5 0.2] ...
       Label: [0]
    2023-02-11  22:51:43.913475  I  Stream "Sklearn stream "iris"": Instance 8:
       Data: [4.4 2.9 1.4 0.2] ...
       Label: [0]
    2023-02-11  22:51:43.913475  I  Stream "Sklearn stream "iris"": Instance 9:
       Data: [4.9 3.1 1.5 0.1] ...
       Label: [0]
    2023-02-11  22:51:43.913475  I  Stream "Sklearn stream "iris"": Reset
    2023-02-11  22:51:43.913475  W  Stream "Sklearn stream "iris"": Fetching all 150 instances...
    2023-02-11  22:51:43.913475  I  Stream "Sklearn stream "iris"": Reset
    2023-02-11  22:51:43.913475  I  Stream "Sklearn stream "iris"": Instance 0:
       Data: [5.1 3.5 1.4 0.2] ...
       Label: [0]
    2023-02-11  22:51:43.913475  I  Stream "Sklearn stream "iris"": Instance 1:
       Data: [4.9 3.  1.4 0.2] ...
       Label: [0]
    2023-02-11  22:51:43.913475  I  Stream "Sklearn stream "iris"": Instance 2:
       Data: [4.7 3.2 1.3 0.2] ...
       Label: [0]
    2023-02-11  22:51:43.913475  I  Stream "Sklearn stream "iris"": Instance 3:
       Data: [4.6 3.1 1.5 0.2] ...
       Label: [0]
    2023-02-11  22:51:43.913475  I  Stream "Sklearn stream "iris"": Instance 4:
       Data: [5.  3.6 1.4 0.2] ...
       Label: [0]
    2023-02-11  22:51:43.913475  I  Stream "Sklearn stream "iris"": Instance 5:
       Data: [5.4 3.9 1.7 0.4] ...
       Label: [0]
    2023-02-11  22:51:43.913475  I  Stream "Sklearn stream "iris"": Instance 6:
       Data: [4.6 3.4 1.4 0.3] ...
       Label: [0]
    2023-02-11  22:51:43.913475  I  Stream "Sklearn stream "iris"": Instance 7:
       Data: [5.  3.4 1.5 0.2] ...
       Label: [0]
    2023-02-11  22:51:43.913475  I  Stream "Sklearn stream "iris"": Instance 8:
       Data: [4.4 2.9 1.4 0.2] ...
       Label: [0]
    2023-02-11  22:51:43.913475  I  Stream "Sklearn stream "iris"": Instance 9:
       Data: [4.9 3.1 1.5 0.1] ...
       Label: [0]
    2023-02-11  22:51:43.913475  W  Stream "Sklearn stream "iris"": Rest of the 150 instances dark...
    2023-02-11  22:51:43.913475  W  Stream "Sklearn stream "iris"": Done in 0.0  seconds (throughput = 140000000 instances/sec)


**Cross Reference**

+ :ref:`API Reference: Streams <target_api_bf_streams>`