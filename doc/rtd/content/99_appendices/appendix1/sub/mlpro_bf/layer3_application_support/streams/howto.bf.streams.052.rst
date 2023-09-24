.. _Howto BF STREAMS 052:
Howto BF-STREAMS-052: Accessing Data from Scikit-Learn
======================================================

Prerequisites
^^^^^^^^^^^^^

Please install the following packages to run this examples properly:

    - `Scikit-Learn <https://pypi.org/project/sklearn/>`_
    - `Numpy <https://pypi.org/project/numpy/>`_



Executable code
^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../../../../../../src/mlpro/bf/examples/howto_bf_streams_052_accessing_data_from_scikitlearn.py
	:language: python



Results
^^^^^^^

.. code-block:: bashh

    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Wrapper "Stream Provider Scikit-learn": Instantiated
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Wrapper "Stream Provider Scikit-learn": Wrapped package scikit-learn installed in version 1.0.2
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Wrapper "Stream Provider Scikit-learn": Getting list of streams...
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Wrapper "Stream Provider Scikit-learn": Number of streams found: 10
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Wrapper "Stream Provider Scikit-learn": Name of requested stream: iris
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Wrapper "Stream Provider Scikit-learn": Getting list of streams...
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Wrapper "Stream Provider Scikit-learn": Number of streams found: 20
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Sklearn stream "iris"": Ready to access in mode 0
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Wrapper "Stream Provider Scikit-learn": Number of features in the stream: 4


    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Sklearn stream "iris"": Reset
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream "Sklearn stream "iris"": Fetching first 10 stream instances...
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Sklearn stream "iris"": Instance 0:
       Data: [5.1 3.5 1.4 0.2] ...
       Label: [0]
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Sklearn stream "iris"": Instance 1:
       Data: [4.9 3.  1.4 0.2] ...
       Label: [0]
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Sklearn stream "iris"": Instance 2:
       Data: [4.7 3.2 1.3 0.2] ...
       Label: [0]
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Sklearn stream "iris"": Instance 3:
       Data: [4.6 3.1 1.5 0.2] ...
       Label: [0]
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Sklearn stream "iris"": Instance 4:
       Data: [5.  3.6 1.4 0.2] ...
       Label: [0]
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Sklearn stream "iris"": Instance 5:
       Data: [5.4 3.9 1.7 0.4] ...
       Label: [0]
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Sklearn stream "iris"": Instance 6:
       Data: [4.6 3.4 1.4 0.3] ...
       Label: [0]
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Sklearn stream "iris"": Instance 7:
       Data: [5.  3.4 1.5 0.2] ...
       Label: [0]
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Sklearn stream "iris"": Instance 8:
       Data: [4.4 2.9 1.4 0.2] ...
       Label: [0]
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Sklearn stream "iris"": Instance 9:
       Data: [4.9 3.1 1.5 0.1] ...
       Label: [0]
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Sklearn stream "iris"": Reset
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream "Sklearn stream "iris"": Fetching all 150 instances...
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Sklearn stream "iris"": Reset
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Sklearn stream "iris"": Instance 0:
       Data: [5.1 3.5 1.4 0.2] ...
       Label: [0]
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Sklearn stream "iris"": Instance 1:
       Data: [4.9 3.  1.4 0.2] ...
       Label: [0]
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Sklearn stream "iris"": Instance 2:
       Data: [4.7 3.2 1.3 0.2] ...
       Label: [0]
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Sklearn stream "iris"": Instance 3:
       Data: [4.6 3.1 1.5 0.2] ...
       Label: [0]
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Sklearn stream "iris"": Instance 4:
       Data: [5.  3.6 1.4 0.2] ...
       Label: [0]
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Sklearn stream "iris"": Instance 5:
       Data: [5.4 3.9 1.7 0.4] ...
       Label: [0]
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Sklearn stream "iris"": Instance 6:
       Data: [4.6 3.4 1.4 0.3] ...
       Label: [0]
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Sklearn stream "iris"": Instance 7:
       Data: [5.  3.4 1.5 0.2] ...
       Label: [0]
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Sklearn stream "iris"": Instance 8:
       Data: [4.4 2.9 1.4 0.2] ...
       Label: [0]
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Sklearn stream "iris"": Instance 9:
       Data: [4.9 3.1 1.5 0.1] ...
       Label: [0]
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream "Sklearn stream "iris"": Rest of the 150 instances dark...
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream "Sklearn stream "iris"": Done in 0.0  seconds (throughput = 140000000 instances/sec)



Cross Reference
^^^^^^^^^^^^^^^

   - :ref:`API Reference: Streams <target_ap_bf_streams>`