.. _RobotHMI:
Robot Manipulator on Homogeneous Matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: mlpro.rl.pool.envs.robotinhtm

.. image:: images/3dmanipulator.png
    :width: 400

This environment represents the robot manipulator in term of mathematical equations.
The mathematical equations are based on rigid body transformation. In this case, the Homogeneous
Transformation Matrix (HTM) is used for the structure. HTM is a matrix that contains both the translation
rotation of a point with respect to some plane.

.. math::

    H=\begin{bmatrix}
    \mathbf{Rot}& \mathbf{Trans}\\ 
    \mathbf{0} & 1
    \end{bmatrix}
    =
    \underbrace{\begin{bmatrix}
    \mathbf{I} & \mathbf{Trans}\\ 
    \mathbf{0} & 1
    \end{bmatrix}}_{translation}
    \underbrace{\begin{bmatrix}
    \mathbf{Rot} & \mathbf{0}\\ 
    \mathbf{0} & 1
    \end{bmatrix}}_{rotation}
    
    
This robotinhtm environment can be imported via:

.. code-block:: python

    from mlpro.rl.pool.envs.robotinhtm import RobotHTM


**Prerequisites**

    - `NumPy <https://pypi.org/project/numpy/>`_
    - `PyTorch <https://pypi.org/project/torch/>`_


**General information**

+------------------------------------+-------------------------------------------------------+
|         Parameter                  |                         Value                         |
+====================================+=======================================================+
| Agents                             | 1                                                     |
+------------------------------------+-------------------------------------------------------+
| Native Source                      | MLPro                                                 |
+------------------------------------+-------------------------------------------------------+
| Action Space Dimension             | [4,]                                                  |
+------------------------------------+-------------------------------------------------------+
| Action Space Base Set              | Real number                                           |
+------------------------------------+-------------------------------------------------------+
| Action Space Boundaries            | [-pi,pi]                                              |
+------------------------------------+-------------------------------------------------------+
| State Space Dimension              | [6,]                                                  |
+------------------------------------+-------------------------------------------------------+
| State Space Base Set               | Real number                                           |
+------------------------------------+-------------------------------------------------------+
| State Space Boundaries             | [-inf,inf]                                            |
+------------------------------------+-------------------------------------------------------+
| Reward Structure                   | Overall reward                                        |
+------------------------------------+-------------------------------------------------------+
  
  
**Action space**

By default, there are 4 action in this environment. The action space represents the angular velocity of
each joint of the robot manipulator.
  
  
**State space**

The state space consists of end-effector positions (x,y,z) of the robot manipulator and target positions (x,y,z).
  

**Reward structure**

By default, the reward structures are shown in the following equation:

.. math::

    reward=-1*\frac{distError}{initDist}-stepReward

    
**Cross reference**
  + `Howto RL-ENV-002: SB3 policy on RobotHTM environment <https://mlpro-int-sb3.readthedocs.io/en/latest/content/01_example_pool/02_howtos_env/howto_rl_env_001_train_agent_with_SB3_policy_on_robothtm_environment.html>`_
  + `Howto RL-MB-001: MBRL on RobotHTM environment <https://mlpro-int-sb3.readthedocs.io/en/latest/content/01_example_pool/04_howtos_mb/howto_rl_mb_002_robothtm_environment.html>`_
  + :ref:`API reference <target_pool_rl_env_robot_manipulator>`

  
**Citation**

If you apply this environment in your research or work, please :ref:`cite <target_publications>` us and the `original paper <https://ieeexplore.ieee.org/document/10002834>`_.