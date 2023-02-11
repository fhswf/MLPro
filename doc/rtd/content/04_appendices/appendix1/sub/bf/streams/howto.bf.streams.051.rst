.. _Howto BF STREAMS 051:
Howto BF-STREAMS-051: Accessing Data from OpenML
================================================

.. 2022-11-21/DA commented due to problems with openml 
    .. automodule:: mlpro.bf.examples.howto_bf_streams_051_accessing_data_from_openml


**Prerequisites**


Please install the following packages to run this examples properly:
    - `OpenML <https://pypi.org/project/openml/>`_
    - `Numpy <https://pypi.org/project/numpy/>`_


**Executable code**

..
    .. literalinclude:: ../../../../../../src/mlpro/bf/examples/howto_bf_streams_051_accessing_data_from_openml.py
	    :language: python

**Results**

.. code-block:: bashh

    2023-02-11  22:49:39.518522  I  Wrapper "OpenML": Instantiated
    2023-02-11  22:49:39.722690  I  Wrapper "OpenML": Wrapped package openml installed in version 0.12.2
    2023-02-11  22:49:39.722690  I  Wrapper "OpenML": Getting list of streams...
    2023-02-11  22:49:42.410072  I  Wrapper "OpenML": Number of streams found: 4992
    2023-02-11  22:49:42.410072  I  Wrapper "OpenML": Name of requested stream: BNG(autos,nominal,1000000)
    2023-02-11  22:49:42.410072  I  Wrapper "OpenML": Getting list of streams...
    2023-02-11  22:49:42.410072  I  Wrapper "OpenML": Number of streams found: 4992
    2023-02-11  22:49:42.410072  I  Wrapped OpenML stream "BNG(autos,nominal,1000000)": Ready to access in mode 0
    2023-02-11  22:49:42.912151  I  Wrapper "OpenML": Number of features in the stream: 25
    2023-02-11  22:49:42.912151  I  Wrapped OpenML stream "BNG(autos,nominal,1000000)": Reset
    2023-02-11  22:49:42.912151  W  Wrapped OpenML stream "BNG(autos,nominal,1000000)": Fetching first 10 stream instances...
    2023-02-11  22:49:42.912151  I  Wrapped OpenML stream "BNG(autos,nominal,1000000)": Instance 0:
       Data: [0. 6. 1. 1. 0. 4. 2. 0. 2. 1. 1. 0. 1. 2.] ...
       Label: [5]
    2023-02-11  22:49:42.927745  I  Wrapped OpenML stream "BNG(autos,nominal,1000000)": Instance 1:
       Data: [ 0. 11.  1.  0.  0.  0.  1.  0.  0.  2.  1.  0.  0.  3.] ...
       Label: [4]
    2023-02-11  22:49:42.927745  I  Wrapped OpenML stream "BNG(autos,nominal,1000000)": Instance 2:
       Data: [ 0. 12.  1.  0.  1.  2.  1.  0.  0.  1.  1.  1.  0.  3.] ...
       Label: [3]
    2023-02-11  22:49:42.927745  I  Wrapped OpenML stream "BNG(autos,nominal,1000000)": Instance 3:
       Data: [0. 1. 1. 0. 0. 2. 1. 0. 2. 0. 0. 2. 0. 3.] ...
       Label: [3]
    2023-02-11  22:49:42.927745  I  Wrapped OpenML stream "BNG(autos,nominal,1000000)": Instance 4:
       Data: [0. 7. 1. 0. 0. 2. 2. 0. 2. 0. 0. 2. 2. 0.] ...
       Label: [3]
    2023-02-11  22:49:42.927745  I  Wrapped OpenML stream "BNG(autos,nominal,1000000)": Instance 5:
       Data: [0. 8. 0. 1. 0. 2. 1. 0. 1. 1. 1. 1. 1. 3.] ...
       Label: [3]
    2023-02-11  22:49:42.927745  I  Wrapped OpenML stream "BNG(autos,nominal,1000000)": Instance 6:
       Data: [ 0. 20.  0.  0.  1.  3.  1.  0.  0.  1.  1.  1.  0.  2.] ...
       Label: [5]
    2023-02-11  22:49:42.927745  I  Wrapped OpenML stream "BNG(autos,nominal,1000000)": Instance 7:
       Data: [ 0. 18.  1.  0.  0.  3.  1.  0.  0.  1.  1.  1.  0.  3.] ...
       Label: [3]
    2023-02-11  22:49:42.927745  I  Wrapped OpenML stream "BNG(autos,nominal,1000000)": Instance 8:
       Data: [0. 4. 1. 0. 0. 2. 1. 0. 1. 1. 1. 0. 1. 3.] ...
       Label: [2]
    2023-02-11  22:49:42.927745  I  Wrapped OpenML stream "BNG(autos,nominal,1000000)": Instance 9:
       Data: [ 0. 12.  1.  0.  0.  2.  1.  0.  0.  1.  1.  0.  2.  3.] ...
       Label: [4]
    2023-02-11  22:49:42.927745  I  Wrapped OpenML stream "BNG(autos,nominal,1000000)": Reset
    2023-02-11  22:49:42.927745  W  Wrapped OpenML stream "BNG(autos,nominal,1000000)": Fetching all 1000000.0 instances...
    2023-02-11  22:49:42.927745  I  Wrapped OpenML stream "BNG(autos,nominal,1000000)": Reset
    2023-02-11  22:49:42.927745  I  Wrapped OpenML stream "BNG(autos,nominal,1000000)": Instance 0:
       Data: [0. 6. 1. 1. 0. 4. 2. 0. 2. 1. 1. 0. 1. 2.] ...
       Label: [5]
    2023-02-11  22:49:42.927745  I  Wrapped OpenML stream "BNG(autos,nominal,1000000)": Instance 1:
       Data: [ 0. 11.  1.  0.  0.  0.  1.  0.  0.  2.  1.  0.  0.  3.] ...
       Label: [4]
    2023-02-11  22:49:42.927745  I  Wrapped OpenML stream "BNG(autos,nominal,1000000)": Instance 2:
       Data: [ 0. 12.  1.  0.  1.  2.  1.  0.  0.  1.  1.  1.  0.  3.] ...
       Label: [3]
    2023-02-11  22:49:42.927745  I  Wrapped OpenML stream "BNG(autos,nominal,1000000)": Instance 3:
       Data: [0. 1. 1. 0. 0. 2. 1. 0. 2. 0. 0. 2. 0. 3.] ...
       Label: [3]
    2023-02-11  22:49:42.927745  I  Wrapped OpenML stream "BNG(autos,nominal,1000000)": Instance 4:
       Data: [0. 7. 1. 0. 0. 2. 2. 0. 2. 0. 0. 2. 2. 0.] ...
       Label: [3]
    2023-02-11  22:49:42.927745  I  Wrapped OpenML stream "BNG(autos,nominal,1000000)": Instance 5:
       Data: [0. 8. 0. 1. 0. 2. 1. 0. 1. 1. 1. 1. 1. 3.] ...
       Label: [3]
    2023-02-11  22:49:42.927745  I  Wrapped OpenML stream "BNG(autos,nominal,1000000)": Instance 6:
       Data: [ 0. 20.  0.  0.  1.  3.  1.  0.  0.  1.  1.  1.  0.  2.] ...
       Label: [5]
    2023-02-11  22:49:42.927745  I  Wrapped OpenML stream "BNG(autos,nominal,1000000)": Instance 7:
       Data: [ 0. 18.  1.  0.  0.  3.  1.  0.  0.  1.  1.  1.  0.  3.] ...
       Label: [3]
    2023-02-11  22:49:42.927745  I  Wrapped OpenML stream "BNG(autos,nominal,1000000)": Instance 8:
       Data: [0. 4. 1. 0. 0. 2. 1. 0. 1. 1. 1. 0. 1. 3.] ...
       Label: [2]
    2023-02-11  22:49:42.927745  I  Wrapped OpenML stream "BNG(autos,nominal,1000000)": Instance 9:
       Data: [ 0. 12.  1.  0.  0.  2.  1.  0.  0.  1.  1.  0.  2.  3.] ...
       Label: [4]
    2023-02-11  22:49:42.927745  W  Wrapped OpenML stream "BNG(autos,nominal,1000000)": Rest of the 1000000.0 instances dark...
    2023-02-11  22:49:59.477853  W  Wrapped OpenML stream "BNG(autos,nominal,1000000)": Done in 16.55  seconds (throughput = 60422 instances/sec)


**Cross Reference**

+ :ref:`API Reference <target_api_bf_streams>`