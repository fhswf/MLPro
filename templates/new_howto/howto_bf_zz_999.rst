.. _Howto BF ZZ 999:
Howto BF-ZZ-999: My high-sophisticated ml sample
================================================

.. automodule:: mlpro.bf.examples.howto_bf_mt_001_parallel_algorithms



Executable code
---------------
.. literalinclude:: ../../../../../../../../src/mlpro/bf/examples/howto_bf_mt_001_parallel_algorithms.py
	:language: python



Results
-------

The howto example logs details of the three runs and in particular the speed factors of multithreading and 
multiprocessing in comparison to the serial/synchronous execution. On a PC with an AMD Ryzen 7 CPU (8/16 cores)
running Linux, the system monitor shows an approx. 5x speedup with multithreading and an approx. 18x speedup with multiprocessing.

.. image:: images/howto.bf.mt.001/howto_bf_mt_001_parallel_algorithms.pngTitle with hyperlink to source
-----------------
Ver. 0.0.0 (YYYY-MM-DD) 

This module description ...

Prerequisites
`````````````````

Please install the following packages to run this examples properly:
    - :ref:`MLPro <Installation>`
  ..
    - `NumPy <https://pypi.org/project/numpy/>`_
  ..
    - `Matplotlib <https://pypi.org/project/matplotlib/>`_
  ..
    - `OpenAI Gym <https://pypi.org/project/gym/>`_
  ..
    - `PyTorch <https://pypi.org/project/torch/>`_
  ..
    - `Stable Baselines3 <https://pypi.org/project/stable-baselines3/>`_
  ..
    - `PettingZoo <https://pypi.org/project/PettingZoo/>`_
  ..
    - `Optuna <https://pypi.org/project/optuna/>`_
  ..
    - `Hyperopt <https://pypi.org/project/hyperopt/>`_
  ..
    - `ROS <http://wiki.ros.org/noetic/Installation>`_
    
Results
`````````````````
Descriptions, plots, images, screenshots of expected results.

Example Code
`````````````````
.. literalinclude:: ../../../template.py
    :language: python

