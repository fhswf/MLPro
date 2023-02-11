.. _Howto_BF_ZZ_999:
Howto BF-ZZ-999: My high-sophisticated ml sample
================================================

.. automodule:: mlpro.bf.examples.howto_bf_zz_999_desciption



**Executable code**

.. literalinclude:: ../../../../../../../../src/mlpro/bf/examples/howto_bf_zz_999_description.py
	:language: python



**Results**

.. code-block:: bashh

    2023-02-11  22:40:18.898725  I  Stream Provider "MLPro": Instantiated
    2023-02-11  22:40:18.898725  I  Stream "Random 10D x 1000": Instantiated
    2023-02-11  22:40:18.898725  I  Stream "Double Spiral 2D x 721": Instantiated
    2023-02-11  22:40:18.898725  I  Stream "Static Clouds 2D": Instantiated
    2023-02-11  22:40:18.898725  I  Stream "Static Clouds 3D": Instantiated
    2023-02-11  22:40:18.898725  I  Stream Provider "MLPro": Getting list of streams...
    2023-02-11  22:40:18.898725  I  Stream Provider "MLPro": Number of streams found: 4
    2023-02-11  22:40:18.898725  W  Stream "Random 10D x 1000": Features: 10 , Labels: 2 , Instances: 1000
    2023-02-11  22:40:18.898725  W  Stream "Double Spiral 2D x 721": Features: 2 , Labels: 0 , Instances: 721
    2023-02-11  22:40:18.898725  W  Stream "Static Clouds 2D": Features: 2 , Labels: 0 , Instances: 1000
    2023-02-11  22:40:18.898725  W  Stream "Static Clouds 3D": Features: 3 , Labels: 0 , Instances: 2000

    Press ENTER to iterate all streams dark...

    2023-02-11  22:53:21.191034  I  Stream Provider "MLPro": Getting list of streams...
    2023-02-11  22:53:21.191034  I  Stream Provider "MLPro": Number of streams found: 4
    2023-02-11  22:53:21.191034  W  Stream "Random 10D x 1000": Number of instances: 1000
    2023-02-11  22:53:21.222301  W  Stream "Random 10D x 1000": Done in 0.031  seconds (throughput = 31982 instances/sec)
    2023-02-11  22:53:21.222301  W  Stream "Double Spiral 2D x 721": Number of instances: 721
    2023-02-11  22:53:21.222301  W  Stream "Double Spiral 2D x 721": Done in 0.0  seconds (throughput = 721000000 instances/sec)
    2023-02-11  22:53:21.222301  W  Stream "Static Clouds 2D": Number of instances: 1000
    2023-02-11  22:53:21.237922  W  Stream "Static Clouds 2D": Done in 0.016  seconds (throughput = 64012 instances/sec)
    2023-02-11  22:53:21.237922  W  Stream "Static Clouds 3D": Number of instances: 2000
    2023-02-11  22:53:21.275686  W  Stream "Static Clouds 3D": Done in 0.038  seconds (throughput = 52959 instances/sec)


**Cross Reference**


+ :ref:`API Reference: Streams <target_api_bf_streams>`
