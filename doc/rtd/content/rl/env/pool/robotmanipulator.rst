`Robot Manipulator on Homogeneous Matrix <https://github.com/fhswf/MLPro/blob/main/src/mlpro/rl/pool/envs/robotinhtm.py>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
    
        import mlpro.rl.pool.envs.robotinhtm
    
    - **Prerequisites**
        - PyTorch
        - NumPy
    
    
    - **General information**
    
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
      
    - **Action space**
    
    By default, there are 4 action in this environment. The action space represents the angular velocity of
    each joint of the robot manipulator.
      
    - **State space**
    
    The state space consists of end-effector positions (x,y,z) of the robot manipulator and target positions (x,y,z).
      
    - **Reward structure**
    
    By default, the reward structures are shown in the following equation:

    .. math::

        reward=-1*\frac{distError}{initDist}-stepReward
      
    - **Version structure**
    
        + Version 0.0.0 : Initial version release in MLPro v. 0.0.0???
        
    If you apply this environment in your research or work, please kindly cite the following related paper:
    
    .. code-block:: bibtex

     @article{NoName2021
      }