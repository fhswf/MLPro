.. _Howto RL 010:
Howto RL-010: Train a wrapped Stable Baslines 3 policy on MLPro's native UR5 environment
========================================================================================

.. automodule:: mlpro.rl.examples.howto_rl_010_train_ur5_environment_with_wrapped_sb3_policy



Prerequisites
-------------

Please install the following packages to run this examples properly:
    - `Stable-Baselines3 <https://pypi.org/project/stable-baselines3/>`_
    - :ref:`RL Environment UR5 Joint Control <ur5jointcontrol>`
    - `NumPy <https://pypi.org/project/numpy/>`_
    - `Matplotlib <https://pypi.org/project/matplotlib/>`_
    - `OpenAI Gym <https://pypi.org/project/gym/>`_
    - `Pytorch <https://pypi.org/project/torch/>`_



Executable code
---------------
.. literalinclude:: ../../../../../src/mlpro/rl/examples/howto_rl_010_train_ur5_environment_with_wrapped_sb3_policy.py
	:language: python



Results
-------

.. image:: images/ur5simulation.gif

The Gazebo GUI should be the first thing that shows up. 
The UR5 robot will move depending on the given action and the training is run. 
When the training is done, the logged rewards will be plotted using the matplotlib library.

The plotted figure is not reproducible due to the simulator's nature of simulating real
world scenario. Although seeds can be set for the random generator, the sampling cannot be 
done at the exact same time during different runs. For a more reproducible results, 
:ref:`Howto RL-012 <Howto RL 012>` is more appropriate.
