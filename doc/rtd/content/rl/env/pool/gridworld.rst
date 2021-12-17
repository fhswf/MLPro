`Grid World <https://github.com/fhswf/MLPro/blob/main/src/mlpro/rl/pool/envs/gridworld.py>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    This Grid World environment can be installed via:

    .. code-block:: python
    
        import mlpro.rl.pool.envs.gridworld
    
    - **3rd Party Dependencies**
    
        - NumPy
    
    - **Overview**
    
    .. code-block:: bash
    
       [[0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 2, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0]]
        
        
    By default, the agent will be placed in a 2 dimensional grid world with the size of 8x8, tasked to reach 
    the goal through position increment actions. The user can customize the dimension of the grid and decide 
    the maximum number of steps. The agent is represented by number 1 and the goal is represented by number 2.
    
    
      
    - **General information**
    
    +------------------------------------+-------------------------------------------------------+
    |         Parameter                  |                         Value                         |
    +====================================+=======================================================+
    | Agents                             | 1                                                     |
    +------------------------------------+-------------------------------------------------------+
    | Native Source                      | MLPro                                                 |
    +------------------------------------+-------------------------------------------------------+
    | Action Space Dimension             | Depends on grid_size                                  |
    +------------------------------------+-------------------------------------------------------+
    | Action Space Base Set              | Real number                                           |
    +------------------------------------+-------------------------------------------------------+
    | Action Space Boundaries            | Depends on grid_size                                  |
    +------------------------------------+-------------------------------------------------------+
    | State Space Dimension              | Depends on grid_size                                  |
    +------------------------------------+-------------------------------------------------------+
    | State Space Base Set               | Integer number                                        |
    +------------------------------------+-------------------------------------------------------+
    | State Space Boundaries             | Depends on grid_size                                  |
    +------------------------------------+-------------------------------------------------------+
    | Reward Structure                   | Overall reward                                        |
    +------------------------------------+-------------------------------------------------------+
      
    - **Action space**
    
    The action directly affects the location of the agent. The action is 
    interpreted as increments towards the current location value. The dimension depends on the grid_size
    parameter.
      
    - **State space**
    
    The state space is initialized from the grid_size parameter, which can be set up to however many dimension 
    as needed. For example, the agent can be placed in a three dimensional world with a 4x4x4 size by setting 
    :code:`grid_size = (4,4,4)`
      
    - **Reward structure**
    
    .. code-block:: python
        
        reward = Reward(self.C_REWARD_TYPE)
        rew = 1
        euclidean_distance = np.linalg.norm(self.goal_pos-self.agent_pos)
        if euclidean_distance !=0:
            rew = 1/euclidean_distance
        if self.num_step >= self.max_step:
            rew -= self.max_step
        
        reward.set_overall_reward(rew.item())
      
    - **Version structure**
    
        + Version 1.0.0 : Initial version release in MLPro v. 0.0.0
        
    