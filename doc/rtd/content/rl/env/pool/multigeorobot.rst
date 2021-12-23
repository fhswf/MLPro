`Multi Geometry Robot <https://github.com/fhswf/MLPro/blob/main/src/mlpro/rl/pool/envs/multigeorobot.py>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    This Multi Geometry Robot environment can be installed via:

    .. code-block:: python
    
        import mlpro.rl.pool.envs.multigeorobot
    
    - **3rd Party Dependencies**
        The environment has been tested in Ubuntu 20.04 running ROS Noetic. 
        
        The installation steps are as follow:
            1. Install `Ubuntu 20.04 <https://releases.ubuntu.com/20.04/>`_
            2. Install `ROS <http://wiki.ros.org/noetic/Installation/Ubuntu>`_
            3. Install `Moveit <https://moveit.ros.org/install/>`_
            4. Install Dependencies:
                .. code-block:: bash
                    
                    sudo apt-get install ros-$ROS_DISTRO-gripper-action-controller ros-$ROS_DISTRO-joint-trajectory-controller
                    sudo apt install ros-$ROS_DISTRO-moveit-resources-prbt-moveit-config
                    sudo apt install ros-$ROS_DISTRO-pilz-industrial-motion-planner
                    sudo apt install python3-pip
                    pip3 install catkin_tools gym empy defusedxml pymodbus numpy netifaces pycryptodomex
                    
            5. Build the Environment:
                .. code-block:: bash
            
                    cd MLPro/src/mlpro/rl/pool/envs/multigeorobot/src
                    cd .. && catkin_make
            
            6. Source the package:
                .. code-block:: bash
                
                    echo "source MLPro/src/mlpro/rl/pool/envs/multigeorobot/devel/setup.bash" >> ~/.bashrc
                    source ~/.bashrc
            7. Change the ros_ws_abspath parameter in:
                .. code-block:: bash
                
                    MLPro/src/mlpro/rl/pool/envs/multigeorobot/src/multi_geo_robot_rl/multi_geo_robot_training/config/multi_geo_robot.yaml
                
    - **Overview**
    
      
    - **General information**

    Building a robot model in URDF format can be a painful work, if you do not know what you are doing. Especially when the model is complex.
    Multi Geometry Robot environment provides an easy way to build your own robot with a predefined configuration. 
    This way, the user do not have to take care of the robot model. All the configurations are automatically built. The robot is also attached with a robotiq
    gripper.
    The robot can be configured in:
        .. code-block:: bash
        
            MLPro/src/mlpro/rl/pool/envs/multigeorobot/src/multi_geo_robot_rl/multi_geo_robot_training/config/multi_geo_robot.yaml

    Below are the parameters on above mentioned file that can be configured:

    +------------------------------------+-------------------------------------------------------------------+----------------------------+
    |         Parameter                  |                 Description                                       |  Example value             |
    +====================================+===================================================================+============================+
    | robot_type                         | Type of the robot, "2D" or "3D"                                   |      "3D"                  |
    +------------------------------------+-------------------------------------------------------------------+----------------------------+
    | arm_num                            | Number of arm, positive integer value                             |      3                     |
    +------------------------------------+-------------------------------------------------------------------+----------------------------+
    | arm_joint_seq                      | Arm Joint Sequence per arm, 0=Fixed Joint, 1=Revolute Joint       | [[1,1,0],[0,1,0],[0,1,0]]  |
    +------------------------------------+-------------------------------------------------------------------+----------------------------+
    | arm_length                         | Length of each Arm, list of positive floating value               | [0.2, 0.2, 0.2]            |
    +------------------------------------+-------------------------------------------------------------------+----------------------------+
    | arm_mass                           | Mass of each arm, list of positive floating value                 | [9, 2, 2]                  |
    +------------------------------------+-------------------------------------------------------------------+----------------------------+
    | adapter_mass                       | Mass of Adapter (connection between arm), positive floating value |        5                   |
    +------------------------------------+-------------------------------------------------------------------+----------------------------+
    | eef_dummy                          | Display dummy ball on the end effector point, boolean value       | False                      |
    +------------------------------------+-------------------------------------------------------------------+----------------------------+
    | max_iterations                     | Max cycle per episode, positive integer value                     |   20                       |
    +------------------------------------+-------------------------------------------------------------------+----------------------------+
    | task_and_robot_environment_name    | The name of the environment task registered on gym                |  'MultiGeoRobotReach-v0'   |
    +------------------------------------+-------------------------------------------------------------------+----------------------------+

    For the joint sequence explanation is shown on the figure below.
    
        .. image:: images/jointseqmultigeo.png
            :align: center
            :width: 400

    The configuration will automatically build the robot model based on the above configuration. Below are the general parameters information
    regarding the Multi Geometry Robot environment.

    
    +------------------------------------+-------------------------------------------------------+
    |         Parameter                  |                         Value                         |
    +====================================+=======================================================+
    | Agents                             | 1                                                     |
    +------------------------------------+-------------------------------------------------------+
    | Native Source                      | MLPro                                                 |
    +------------------------------------+-------------------------------------------------------+
    | Action Space Dimension             | [predefined by the configuration,]                    |
    +------------------------------------+-------------------------------------------------------+
    | Action Space Base Set              | Real number                                           |
    +------------------------------------+-------------------------------------------------------+
    | Action Space Boundaries            | [-0.1, 0.1]                                           |
    +------------------------------------+-------------------------------------------------------+
    | State Space Dimension              | [6,]                                                  |
    +------------------------------------+-------------------------------------------------------+
    | State Space Base Set               | Real number                                           |
    +------------------------------------+-------------------------------------------------------+
    | State Space Boundaries             | [-2.0, 2.0]                                           |
    +------------------------------------+-------------------------------------------------------+
    | Reward Structure                   | Overall reward                                        |
    +------------------------------------+-------------------------------------------------------+
      
    - **Action space**
    
    The action of the agent directly affects the joint angles (rad) of the robot. The action is 
    interpreted as increments towards the current value. The number of action depends on above configuration.
    
    - **State space**
    
    The state space consists of position information of the end effector (Red Ball) and 
    the target location (Blue Ball). 
      
    +--------------------+---------------------------------------------+-----------------------+
    | Element            | Parameter                                   | Boundaries            |
    +====================+=============================================+=======================+
    | PositionX          | m                                           | [-2.0, 2.0]           |
    +--------------------+---------------------------------------------+-----------------------+
    | PositionY          | m                                           | [-2.0, 2.0]           |
    +--------------------+---------------------------------------------+-----------------------+
    | PositionZ          | m                                           | [-2.0, 2.0]           |
    +--------------------+---------------------------------------------+-----------------------+
    | Targetx            | m                                           | [-2.0, 2.0]           |
    +--------------------+---------------------------------------------+-----------------------+
    | Targety            | m                                           | [-2.0, 2.0]           |
    +--------------------+---------------------------------------------+-----------------------+
    | Targetz            | m                                           | [-2.0, 2.0]           |
    +--------------------+---------------------------------------------+-----------------------+
    
    - **Reward structure**
    
    .. code-block:: python
        
        distance = np.linalg.norm(np.array(observations[:3]) - np.array(observations[3:]))
        ratio = distance/self.init_distance
        reward = -np.ones(1)*ratio
        reward = reward - 10e-3

        if done:
            reward += self.reached_goal_reward
      
    - **Version structure**
    
        + Version 1.4.0