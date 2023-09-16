.. _Howto_BF_ZZ_999:
Howto BF-STREAMS-001: Accessing Native Data From MLPro
================================================

.. automodule:: mlpro.bf.examples.howto_bf_streams_001_accessing_native_data_from_mlpro



**Executable code**

.. literalinclude:: ../../../../../../../../../src/mlpro/bf/examples/howto_bf_streams_001_accessing_native_data_from_mlpro.py
	:language: python



**Results**

.. code-block:: bashh

    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream Provider "MLPro": Instantiated
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Random 10D x 1000": Instantiated
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Double Spiral 2D x 721": Instantiated
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Static Clouds 2D": Instantiated
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Static Clouds 3D": Instantiated
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream Provider "MLPro": Getting list of streams...
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream Provider "MLPro": Number of streams found: 4
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream "Random 10D x 1000": Features: 10 , Labels: 2 , Instances: 1000
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream "Double Spiral 2D x 721": Features: 2 , Labels: 0 , Instances: 721
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream "Static Clouds 2D": Features: 2 , Labels: 0 , Instances: 1000
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream "Static Clouds 3D": Features: 3 , Labels: 0 , Instances: 2000

    Press ENTER to iterate all streams dark...

    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream Provider "MLPro": Getting list of streams...
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream Provider "MLPro": Number of streams found: 4
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream "Random 10D x 1000": Number of instances: 1000
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream "Random 10D x 1000": Done in 0.031  seconds (throughput = 31982 instances/sec)
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream "Double Spiral 2D x 721": Number of instances: 721
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream "Double Spiral 2D x 721": Done in 0.0  seconds (throughput = 721000000 instances/sec)
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream "Static Clouds 2D": Number of instances: 1000
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream "Static Clouds 2D": Done in 0.016  seconds (throughput = 64012 instances/sec)
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream "Static Clouds 3D": Number of instances: 2000
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream "Static Clouds 3D": Done in 0.038  seconds (throughput = 52959 instances/sec)


**Cross Reference**


+ :ref:`API Reference: Streams <target_ap_bf_streams>`
