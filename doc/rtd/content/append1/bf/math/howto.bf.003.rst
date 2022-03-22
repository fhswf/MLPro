.. _Howto BF 3:
`Howto 03 - (Math) Spaces, subspaces and elements <https://github.com/fhswf/MLPro/blob/main/examples/bf/Howto%2003%20-%20(Math)%20Spaces%2C%20subspaces%20and%20elements.py>`_
================
Ver. 1.0.2 (2021-12-03)

This module demonstrates how to create a space and subspaces and to spawn elements. The results section below shows implementation of Euclidean spaces, subspaces, adding and modifying dimensions in the Euclidean spaces with the help of MLPro. This module also shows how to add, remove and modify elements in the space along with performing methods such as Euclidean distance on the elements in with the help of MLPro. Run the script provided in the example code section of this document to get the results as below.

Results
`````````````````
This is an example result to understand the Math module in MLPro

::

	YYYY-MM-DD  HH:MM:SS.ssss  I  Demo Spaces & Elements: Instantiated 
	YYYY-MM-DD  HH:MM:SS.ssss  I  Demo Spaces & Elements: 6-dimensional Euclidian space created 
	YYYY-MM-DD  HH:MM:SS.ssss  I  Demo Spaces & Elements: 12-dimensional Euclidian space created 
	YYYY-MM-DD  HH:MM:SS.ssss  I  Demo Spaces & Elements: Subspace 1 - Number of dimensions and short name of second dimension: 3 Vel 
	YYYY-MM-DD  HH:MM:SS.ssss  I  Demo Spaces & Elements: Subspace 2 - Number of dimensions and short name of third dimension: 3 AAcc 
	YYYY-MM-DD  HH:MM:SS.ssss  I  Demo Spaces & Elements: Subspace 3 - Number of dimensions and short name of second dimension: 2 Ang 
	YYYY-MM-DD  HH:MM:SS.ssss  I  Demo Spaces & Elements: New element created with dim ids / values: [0, 1, 2, 3, 4, 5]  /  [0, 0, 0, 0, 0, 0] 
	YYYY-MM-DD  HH:MM:SS.ssss  I  Demo Spaces & Elements: Element changed to  [4.77, -8.22, 0, 0, 0, 0] 
	YYYY-MM-DD  HH:MM:SS.ssss  I  Demo Spaces & Elements: New element e1 = [0, 0, 0, 0, 0, 0] 
	YYYY-MM-DD  HH:MM:SS.ssss  I  Demo Spaces & Elements: New element e2 = [0, 0, 0, 0, 0, 1] 
	YYYY-MM-DD  HH:MM:SS.ssss  I  Demo Spaces & Elements: Euclidian distance between e1 and e2 = 1.0


Prerequisites
`````````````````

Please install the following packages to run this examples properly:
    - :ref:`MLPro <Installation>`
    - `NumPy <https://pypi.org/project/numpy/>`_

Example Code
`````````````````
Run this code in order to get results shown above

.. literalinclude:: ../../../../../../examples/bf/Howto 03 - (Math) Spaces, subspaces and elements.py
    :language: python

