.. _coldec:
2D Collision Avoidance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: mlpro.rl.pool.envs.collisionavoidance_2D


-- overview of 2D collision avoidance environment, what are the objectives

-- gif

.. image:: images/2Dcollisionavoidance_1.gif
    :width: 650px
    :align: center

-- reward function is not defined

-- target positions can be dynamics

-- number of points can be defined

-- collision can alos 

-- it always start with linear path


The 2D collision avoidance environment can be imported via:

.. code-block:: python

    from mlpro.rl.pool.envs.collisionavoidance_2D import DynamicTrajectoryPlanner


**Prerequisites**
Please install below package to use the MLPro's BGLP environment

 - `NumPy <https://pypi.org/project/numpy/>`_



**General Information**

+------------------------------------+------------------------------------------------------------------------+
|         Parameter                  |                         Value                                          |
+====================================+========================================================================+
| Agents                             | 1                                                                      |
+------------------------------------+------------------------------------------------------------------------+
| Native Source                      | MLPro                                                                  |
+------------------------------------+------------------------------------------------------------------------+
| Action Space Dimension             | [p_num_point-2,2], default: [3,2]                                      |
+------------------------------------+------------------------------------------------------------------------+
| Action Space Base Set              | Real numbers                                                           |
+------------------------------------+------------------------------------------------------------------------+
| Action Space Boundaries            | p_action_boundaries, default: [-0.02,0.02]                             |
+------------------------------------+------------------------------------------------------------------------+
| State Space Dimension              | [p_num_point,2], default: [5,2]                                        |
+------------------------------------+------------------------------------------------------------------------+
| State Space Base Set               | Real numbers                                                           |
+------------------------------------+------------------------------------------------------------------------+
| State Space Boundaries             | x-axis: p_xlimit, default: [-4,4]. y-axis: p_ylimit, default: [-4,4].  |
+------------------------------------+------------------------------------------------------------------------+
| Reward Structure                   | Individual reward for each agent                                       |
+------------------------------------+------------------------------------------------------------------------+
    

**Action Space**

-- to be edit

In this environment, we consider 5 actuators to be controlled. 
Thus, there are 5 agents and 5 joint actions because each agent requires an action.
Every action is normalized within a range between 0 and 1, except for Agent 3.
0 means the minimum possible action and 1 means the maximum possible action.
For Agent 3, the vibratory conveyor has a different character than other actuators, which mostly perform in a continuous manner.
The vibratory conveyor can only be either fully switched-on or switched-off. Therefore the base set of action for Agent 3 is an integer (0/1).
0 means off and 1 means on.

+-------+-------------------+--------+-------------------+--------------+
| Agent | Actuator          | Station| Parameter         | Boundaries   |
+=======+===================+========+===================+==============+
|   1   | Conveyor Belt     | A      | rpm               | 450 ... 1800 |
+-------+-------------------+--------+-------------------+--------------+
|   2   | Vacuum Pump       | B      | on-duration (sec) | 0 ... 4.575  |
+-------+-------------------+--------+-------------------+--------------+
|   3   | Vibratory Conveyor| B      | on/off            | 0/1          |
+-------+-------------------+--------+-------------------+--------------+
|   4   | Vacuum Pump       | C      | on-duration (sec) | 0 ... 9.5    |
+-------+-------------------+--------+-------------------+--------------+
|   5   | Rotary Feeder     | C      | rpm               | 450 ... 1450 |
+-------+-------------------+--------+-------------------+--------------+
  
  
**State Space**

-- to be edit

The state information in the BGLP is the fill levels of the reservoirs.
Each agent is always placed in between two reservoirs, e.g. between a silo and a hopper or vice versa.
Therefore, each agent has two state information, which is shared with their neighbours.
Every state is normalized within a range between 0 and 1.
0 means the minimum fill-level and 1 means the maximum fill-level.

+------+----------+--------+--------+---------------+
| Agent| State No.| Element| Station| Boundaries    |
+======+==========+========+========+===============+
|      | 1        | Silo   | A      | 0 ... 17.42 L |
+ 1    +----------+--------+--------+---------------+
|      | 2        |        |        |               |
+------+----------+ Hopper + A      + 0 ... 9.1 L   +
|      | 1        |        |        |               |
+ 2    +----------+--------+--------+---------------+
|      | 2        |        |        |               |
+------+----------+ Silo   + B      + 0 ... 17.42 L +
|      | 1        |        |        |               |
+ 3    +----------+--------+--------+---------------+
|      | 2        |        |        |               |
+------+----------+ Hopper + B      + 0 ... 9.1 L   +
|      | 1        |        |        |               |
+ 4    +----------+--------+--------+---------------+
|      | 2        |        |        |               |
+------+----------+ Silo   + C      + 0 ... 17.42 L +
|      | 1        |        |        |               |
+ 5    +----------+--------+--------+---------------+
|      | 2        | Hopper | C      | 0 ... 9.1 L   |
+------+----------+--------+--------+---------------+
  
  
**Reward Structure**

-- to be edit

The reward structure is implemented according to `this paper <https://doi.org/10.1016/j.compchemeng.2021.107382>`_.
You can also find the source code of the reward structure, `here <https://github.com/fhswf/MLPro/blob/13b7b8a82d90b626f40ea7c268706e43889b9e00/src/mlpro/rl/pool/envs/bglp.py#L971-L982>`_.
The given reward is an individual scalar reward for each agent. To be noted, this reward function is more suitable for a continuous production scenario.

If you would like to implement a customized reward function, you can follow these lines of codes:

.. code-block:: python

    class MyBGLP(BGLP):
    
        def calc_reward(self):
        
            # Each agent has an individual reward
            if self.reward_type == Reward.C_TYPE_EVERY_AGENT:
                for actnum in range(len(self.acts)):
                    acts = self.acts[actnum]
                    self.reward[actnum] = 0
                return self.reward[:]
                
            # Overall reward
            elif self.reward_type == Reward.C_TYPE_OVERALL:
                self.overall_reward = 0
                return self.overall_reward
 

**Cross Reference**

    + :ref:`API Reference <target_pool_rl_env_2dcol>`


**Citation**

-- to be edit

If you apply this environment in your research or work, please :ref:`cite <target_publications>` us and the `original paper <https://doi.org/10.1016/j.compchemeng.2021.107382>`_.