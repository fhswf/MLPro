.. _Howto BF 3:
`Howto 03 - (Math) Spaces, subspaces and elements <https://github.com/fhswf/MLPro/blob/main/examples/bf/Howto%2003%20-%20(Math)%20Spaces%2C%20subspaces%20and%20elements.py>`_
================
Ver. 1.0.3 (2022-02-25)

This module demonstrates how to create a space and subspaces and to spawn elements.

Prerequisites
`````````````````

Please install the following packages to run this examples properly:
    - :ref:`MLPro <Installation>`
    - `NumPy <https://pypi.org/project/numpy/>`_

Results
`````````````````
.. code-block:: bash

    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Demo Spaces & Elements: Instantiated 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Demo Spaces & Elements: 6-dimensional Euclidian space created 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Demo Spaces & Elements: 12-dimensional Euclidian space created 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Demo Spaces & Elements: Subspace 1 - Number of dimensions and short name of second dimension: 3 Vel 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Demo Spaces & Elements: Subspace 2 - Number of dimensions and short name of third dimension: 3 AAcc 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Demo Spaces & Elements: Subspace 3 - Number of dimensions and short name of second dimension: 2 Ang 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Demo Spaces & Elements: New element created with dim ids / values: ['b0ad58b1-baa5-4bea-82eb-f198d3577e59', '417f3504-f431-4d5a-86e5-0c53a7362345', '8bfd718f-75a0-4eae-9504-563f5cc5749a', 'c9ed3e50-1096-41a9-b4fd-bc7f38b1d804', '0e96e1b2-040f-42d0-8a8b-f1c080222ea5', '359c7fd9-88f0-468a-8cfb-4cca7224fce7']  /  [0, 0, 0, 0, 0, 0] 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Demo Spaces & Elements: Element changed to  [4.77, -8.22, 0, 0, 0, 0] 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Demo Spaces & Elements: New element e1 = [0, 0, 0, 0, 0, 0] 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Demo Spaces & Elements: New element e2 = [0, 0, 0, 0, 0, 1] 
    YYYY-MM-DD  HH:MM:SS.SSSSSS  I  Demo Spaces & Elements: Euclidian distance between e1 and e2 = 1.0 
    
The variable demo is created with MathDemo as its type.

Example Code
`````````````````

.. literalinclude:: ../../../../../../examples/bf/Howto 03 - (Math) Spaces, subspaces and elements.py
    :language: python

