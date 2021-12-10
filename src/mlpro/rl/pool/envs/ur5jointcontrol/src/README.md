**Installation Steps**
1. Install ubuntu 20.04

2. Install ROS: 
	- http://wiki.ros.org/noetic/Installation/Ubuntu
    
3. Install Moveit:
	- https://moveit.ros.org/install/
    
4. Install dependencies:
    - sudo apt-get install ros-$ROS_DISTRO-ur-client-library ros-$ROS_DISTRO-joint-trajectory-controller
    - sudo apt install ros-noetic-moveit-resources-prbt-moveit-config
    - sudo apt install ros-noetic-pilz-industrial-motion-planner
    - sudo apt install python3-pip
    - pip3 install catkin_tools gym pymodbus
    
5. Build the Environment:
    - cd MLPro/src/mlpro/rl/pool/envs/ur5jointcontrol/src
    - git submodule update --init
    - cd .. && catkin_make
    
6. Source the package:
    - echo "source MLPro/src/mlpro/rl/pool/envs/ur5jointcontrol/devel/setup.bash" >> ~/.bashrc
    - source ~/.bashrc
    
7. change ros_ws_abspath in:  
        .../ur5jointcontrol/src/reinforcement_learning/ur5_lab/config/ur5_simple_task_param.yaml

