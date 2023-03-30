Data Management
===============

Data management in a framework is extremely important, which mostly refers to the organization, storage, and retrieval of data within the framework.
In MLPro, our team also provides such functionalities as saving data, loading data, storing data, creating a buffer, and plotting data.
This involves defining a data model that describes the structure and relationships between data elements, implementing mechanisms for storing and retrieving data, and managing data consistency and integrity.
A well-designed data management system is essential for the efficient and effective processing of data within the framework.

On this page, the data management within the MLPro framework is explained.
However, an explanation of the plotting functionality is provided in the :ref:`next subchapter <DataPlotting>` of this documentation.
The related data management classes can be accessed as follows:

.. code-block:: python

    from mlpro.bf.data import * 

In general, there are two main functionalities of data management in MLPro:

    1) **Data Storing**
        The second possibility is to store a bunch of data in MLPro's **DataStoring** class with three different layers, as follows:

            - **Layer 1 - Data Names** : the labels or the feature names of the data.

            - **Layer 2 - Frames** : the frames can be added to each label or feature name. If none, then the frame id can be set to '0' all the time.

            - **Layer 3 - Values** : the values can be added to each frame with a specific label or feature name.

        Therefore, to add value to the data storage, the users can use ``DataStoring.memorize(p_variable, p_frame_id, p_value)``, in which ``p_variable, p_frame_id, p_value`` refer to the feature name, the frame id, and the added value respectively.

        For better understanding : :ref:`Howto BF-003: Store and plot data <Howto BF 003>`
    
    2) **Buffering**
        The other data handling functionality is buffering by MLPro's **Buffer** class.
        The buffer is an important component in machine learning and online learning areas, where a number of data have to be stored, updated frequently, and also used for data sampling purposes.
        The following shows what the users can do with the buffer and how to define the buffer:

            - Redefining ``Buffer._gen_sample_ind(self, p_num: int)`` to set up a method of sampling data from the buffer or optionally simply using **BufferRnd** class with random sampling functionality.

            - Instantiating a buffer with a defined maximum buffer size.

            - Storing the target values with specific feature names using **BufferElement**.

            - Adding the buffer element to the buffer.

            - Optional functionalities:

                - Checking whether the buffer is full.

                - Obtaining all data from the buffer.

                - Sampling from the buffer.

                - Clearing the buffer. 
        
        For better understanding : :ref:`Howto BF-004: Buffers <Howto BF 004>`
    
**Cross Reference**
    + :ref:`Howto BF-003: Store and plot data <Howto BF 003>`
    + :ref:`Howto BF-004: Buffers <Howto BF 004>`
    + :ref:`API Reference <target_api_bf_data>`
