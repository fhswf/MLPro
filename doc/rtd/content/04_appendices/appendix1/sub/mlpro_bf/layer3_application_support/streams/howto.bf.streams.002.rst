.. _Howto_BF_STREAMS_002:
Howto BF-STREAMS-002: Accessing Data From CSV Files
=============================================================

.. automodule:: mlpro.bf.examples.howto_bf_streams_002_accessing_data_from_csv_files



**Executable code**

.. literalinclude:: ../../../../../../../../../src/mlpro/bf/examples/howto_bf_streams_002_accessing_data_from_csv_files.py
	:language: python



**Results**

.. code-block:: bashh

    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream Provider "MLPro": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Random 10D x 1000": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Double Spiral 2D x 721": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Static Clouds 2D": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Static Clouds 3D": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "CSV Format to MLPro Stream": Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream "CSV Format to MLPro Stream": Features: 3 , Labels: 1 , Instances: 100000 

    Press ENTER to iterate all streams dark...

    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream "CSV Format to MLPro Stream": Number of instances: 100000 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream "CSV Format to MLPro Stream": Done in 0.318  seconds (throughput = 314131 instances/sec) 


**Cross Reference**


+ :ref:`API Reference: Data from CSV files <ap2_csv_files>`
