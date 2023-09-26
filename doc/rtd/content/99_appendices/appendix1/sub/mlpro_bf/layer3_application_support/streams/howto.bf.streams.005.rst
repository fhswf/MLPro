.. _Howto_BF_STREAMS_005:
Howto BF-STREAMS-005: Streams Sampler
=====================================

**Executable code**


.. literalinclude:: ../../../../../../../../../test/howtos/bf/howto_bf_streams_005_sampler.py
	:language: python



**Results**

.. code-block:: bashh

    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "": Instantiated 

    Press ENTER to iterate all streams dark...

    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream CSV File "data_storage.csv": Reset
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream CSV File "data_storage.csv": Features: 3 , Labels: 1 , Instances: 100000 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream CSV File "data_storage.csv": Reset
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream CSV File "data_storage.csv": Number of instances being sampled: 15298 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream CSV File "data_storage.csv": Done in 0.658  seconds (throughput = 152048 instances/sec)
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream CSV File "data_storage.csv": Reset
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream CSV File "data_storage.csv": Number of instances being sampled: 24884 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream CSV File "data_storage.csv": Reset
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream CSV File "data_storage.csv": Number of instances being sampled: 100 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream CSV File "data_storage.csv": Reset
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream CSV File "data_storage.csv": Number of instances being sampled: 786 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream CSV File "data_storage.csv": Reset
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream CSV File "data_storage.csv": Number of instances being sampled: 5000

    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Random 10D x 1000": Instantiated 

    Press ENTER to iterate all streams dark...

    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Random 10D x 1000": Reset
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream "Random 10D x 1000": Features: 10 , Labels: 2 , Instances: 1000
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Random 10D x 1000": Reset
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream "Random 10D x 1000": Number of instances being sampled: 150
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream "Random 10D x 1000": Done in 0.008  seconds (throughput = 133174 instances/sec)
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Random 10D x 1000": Reset
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream "Random 10D x 1000": Number of instances being sampled: 239
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Random 10D x 1000": Reset
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream "Random 10D x 1000": Number of instances being sampled: 100 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Random 10D x 1000": Reset
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream "Random 10D x 1000": Number of instances being sampled: 333
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Stream "Random 10D x 1000": Reset
    YYYY-MM-DD  HH:MM:SS.SSSSSS  W  Stream "Random 10D x 1000": Number of instances being sampled: 50



**Cross Reference**

    - :ref:`API Reference: Random Samplers <ap2_samplers>`
    - :ref:`API Reference: Data from CSV files <ap2_csv_files>`
