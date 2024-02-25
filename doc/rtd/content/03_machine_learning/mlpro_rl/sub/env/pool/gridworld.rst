.. _GridWorld:
Grid World
^^^^^^^^^^^^

.. automodule:: mlpro.rl.pool.envs.gridworld

Grid World is a very simple environment and suits to someone who just starts to understand Reinforcement Learning or Markov Decision Process.

In this Grid World environment, by default, the agent will be placed in a 2 dimensional grid world with the size of 8x8, tasked to reach 
the goal through position increment actions. The user can customize the dimension of the grid and decide 
the maximum number of steps. The agent is represented by number 1 and the goal is represented by number 2, where number 3 means that the agent is reaching the goal.
In the latest version of Grid World, we provided the possibilities to set your own or random initial and/or goal positions.
Moreover, there are two possible types of actions, such as continuous actions which can reached the goal in one-shot and discrete actions (only for 2-D grid world).
The discrete actions consists of 'up', 'right', 'down', and 'left' (or 'north', 'east', 'south', and 'west') respectively.
Here is the example of the grid world environment, by default and with random initial and goal states:

.. code-block:: bash

   [[0, 0, 0, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, 0, 0, 1],
   [0, 0, 0, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, 0, 0, 0],
   [0, 0, 0, 2, 0, 0, 0, 0],
   [0, 0, 0, 0, 0, 0, 0, 0]]
   
At the moment, we have not incorporated any obstacles or walls, which will be added in the near future.
The current implementation shows that if an action lead to a state outside the boundaries, then the state is back to the previous state.
    

This Grid World environment can be imported via:

.. code-block:: python

    from mlpro.rl.pool.envs.gridworld import GridWorld
    
**Prerequisites**

    - `NumPy <https://pypi.org/project/numpy/>`_


**General Information**

+------------------------------------+-----------------------------------------------------------------+
|         Parameter                  |                         Value                                   |
+====================================+=================================================================+
| Number of agent                    | 1                                                               |
+------------------------------------+-----------------------------------------------------------------+
| Native Source                      | MLPro                                                           |
+------------------------------------+-----------------------------------------------------------------+
| Action Space Dimension             | Depends on the grid size, e.g. (8, 8), (8, 8, 8), etc.          |
+------------------------------------+-----------------------------------------------------------------+
| Action Space Base Set              | (Type 1) Real number                                            |
+                                    +-----------------------------------------------------------------+
|                                    | (Type 2) Integer number                                         |
+------------------------------------+-----------------------------------------------------------------+
| Action Space Boundaries            | (Type 1) Depends on grid_size                                   |
+                                    +-----------------------------------------------------------------+
|                                    | (Type 2) 0 to 3                                                 |
+------------------------------------+-----------------------------------------------------------------+
| State Space Dimension              | Depends on the grid size                                        |
+------------------------------------+-----------------------------------------------------------------+
| State Space Base Set               | Integer number                                                  |
+------------------------------------+-----------------------------------------------------------------+
| State Space Boundaries             | 0 to 3                                                          |
+------------------------------------+-----------------------------------------------------------------+
| Reward Structure                   | Overall reward                                                  |
+------------------------------------+-----------------------------------------------------------------+
 
**Action Space**

There are two types of actions that can be selected in the beginning of the training, such as continuous actions ('C_ACTION_TYPE_CONT') and discrete actions ('C_ACTION_TYPE_DISC_2D').
At the moment, the discrete action is limited to 2-dimensional grid world.

For continuous action, the action directly affects the location of the agent. The action is 
interpreted as increments towards the current location value. The dimension depends on the grid_size
parameter. By default, there is a possibility to reach the target in one shot.

For discrete action, there are four possible actions that represented by number 0 to 3, as follows:
Number '0' means 'up' or 'north'.
Number '1' means 'right' or 'east'.
Number '2' means 'down' or 'south'.
Number '3' means 'left' or 'west'.

**State Space**

The state space is initialized from the grid_size parameter, which can be set up to however many dimension 
as needed. For example, the agent can be placed in a two dimensional world with a n x m size, three dimensional world with a n x m x p, or even more,
for instance by setting 
:code:`grid_size = (n,m)`
or
:code:`grid_size = (n,m,p)`.

Additionally, the initial and goal position can be randomized or predefined.
  
**Reward Structure**

The default reward function is really simple and straight forward, where the reward is 1, if the agent reaches the goal.
The reward is 1 minus the euclidean distance between goal states and current states, if the agent has not reached the goal yet.

.. code-block:: python
    
    reward = Reward(self.C_REWARD_TYPE)
    rew = 1
    euclidean_distance = np.linalg.norm(self.goal_pos-self.agent_pos)
    if euclidean_distance !=0:
        rew = 1/euclidean_distance
    if self.num_step >= self.max_step:
        rew -= self.max_step
    
    reward.set_overall_reward(rew.item())

  
**Cross Reference**
    + :ref:`Howto RL-MB-002: MBRL with MPC on Grid World Environment <Howto MB RL 002>`
    + :ref:`API Reference <target_pool_rl_env_grid_world>`


**Citation**

If you apply this environment in your research or work, please :ref:`cite <target_publications>` us.