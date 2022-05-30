.. _Howto RL 018:
Howto RL-018: Train a wrapped Stable Baselines 3 policy on MLPro's native MultiGeo environment
====================================================================================================

.. automodule:: mlpro.rl.examples.howto_rl_018_train_wrapped_sb3_policy_on_multigeo_environment



Prerequisites
-------------

Please install the following packages to run this examples properly:
    - `Stable-Baselines3 <https://pypi.org/project/stable-baselines3/>`_
    - :ref:`RL Environment Multi Geometry Robot <multigeorobot>`
    - `NumPy <https://pypi.org/project/numpy/>`_
    - `Matplotlib <https://pypi.org/project/matplotlib/>`_
    - `OpenAI Gym <https://pypi.org/project/gym/>`_
    - `Pytorch <https://pypi.org/project/torch/>`_
    


Executable code
---------------
.. literalinclude:: ../../../../../src/mlpro/rl/examples/howto_rl_018_train_wrapped_sb3_policy_on_multigeo_environment.py
	:language: python



Results
-------

.. image:: images/multigeosim.gif

The Gazebo GUI should be the first thing that shows up. 
The Multi Geometry robot will move depending on the given action and the training is run. 
When the training is done, the logged rewards will be plotted using the matplotlib library.

The plotted figure is is not reproducible due to the simulator's nature of simulating real
world scenario. Although seeds can be set for the random generator, the sampling cannot be 
done at the exact same time during different runs.