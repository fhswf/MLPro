## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.pool.envs.ur5jointcontrol
## -- Module  : urjointcontrol.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-13  0.0.0     WB       Creation
## -- 2021-09-13  1.0.0     WB       Released first version
## -- 2021-09-13  1.0.1     WB       Instantiated without WrEnvGym
## -- 2021-09-23  1.0.2     WB       Increased C_LATENCY
## -- 2021-10-05  1.0.3     SY       Update following new attributes done and broken in State
## -- 2021-12-03  1.0.4     DA       Refactoring
## -- 2021-12-20  1.0.5     DA       Replaced 'done' by 'success'
## -- 2021-12-20  1.0.6     WB       Update 'success' and 'broken' rule
## -- 2021-12-21  1.0.7     DA       Class UR5JointControl: renamed method reset() to _reset()
## -- 2022-01-20  1.0.8     MRD      Use the gym wrapper to wrap the ur5 environment
## -- 2022-03-04  1.1.0     WB       Adds the ability to control gripper
## -- 2022-06-06  2.0.0     MRD      Add ability to self build the ros workspace
## --                       MRD      Add the connection to the real robot, wrapped in Gym environment
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.0.0 (2022-06-06)

This module provides an environment with multivariate state and action spaces 
based on the Gym-based environment 'UR5RandomTargetTask-v0'. 
"""

import sys
import platform
import subprocess
import time
import os
import mlpro
from mlpro.bf.various import Log
from mlpro.wrappers.openai_gym import WrEnvGYM2MLPro
from mlpro.rl.models import *
import numpy as np



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class UR5JointControl(Environment):
    """
    This environment multivariate space and action spaces by duplicating the
    Gym-based environment 'UR5RandomTargetTask-v0'. 
    """

    C_NAME = 'UR5JointControl'
    C_LATENCY = timedelta(0, 5, 0)
    C_INFINITY = np.finfo(np.float32).max
    C_REAL_MODE_ROS = 0
    C_REAL_MODE_SOCKET = 1

    ## -------------------------------------------------------------------------------------------------
    def __init__(self, p_seed=0, p_build=False, p_sim=True, p_real_method=C_REAL_MODE_ROS, p_robot_ip="", p_reverse_ip="", 
        p_reverse_port=50001, p_visualize=False, p_logging=True):
        """
        Parameters:
            p_logging       Boolean switch for logging
        """

        if p_real_method == self.C_REAL_MODE_ROS:
            # Use ROS as the simulation and real
            if p_build:
                self._build()

            try:
                import rospy
                from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
                from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
            except (ModuleNotFoundError, ImportError) as e:
                logger = Log()
                logger.C_TYPE = "Log"
                logger.C_NAME = "Pre-check ROS"
                logger.log(Log.C_LOG_TYPE_E, "ROS or UR5 Workspace is not installed")
                logger.log(Log.C_LOG_TYPE_W, "Please use the guideline here:")
                logger.log(Log.C_LOG_TYPE_W, "https://mlpro.readthedocs.io/en/latest/content/rl/env/pool/ur5jointcontrol.html")
            else:
                roscore = subprocess.Popen('roscore')
                rospy.init_node('ur5_lab_training_start', anonymous=True, log_level=rospy.WARN)

                LoadYamlFileParamsTest(rospackage_name="ur5_lab",
                                    rel_path_from_package_to_file="config",
                                    yaml_file_name="ur5_lab_task_param.yaml")

                ros_ws_path = mlpro.rl.pool.envs.ur5jointcontrol.__file__.replace("/__init__.py", "")
                rospy.set_param('ros_ws_path', ros_ws_path)
                rospy.set_param('sim', p_sim)

                # Init OpenAI_ROS ENV
                if p_sim:
                    environment = rospy.get_param('/ur5_lab/simulation_environment')
                    rospy.set_param('visualize', p_visualize)
                else:
                    environment = rospy.get_param('/ur5_lab/real_environment')
                    rospy.set_param('robot_ip', p_robot_ip)
                    rospy.set_param('reverse_ip', p_reverse_ip)
                    rospy.set_param('reverse_port', p_reverse_port)

                max_step_episode = rospy.get_param('/ur5_lab/max_iterations')

                self._gym_env = StartOpenAI_ROS_Environment(environment, max_step_episode)
                self._gym_env.seed(p_seed)

                self.C_NAME = 'Env "' + self._gym_env.spec.id + '"'

                if p_sim:
                    super().__init__(p_mode=Mode.C_MODE_SIM, p_logging=p_logging)
                else:
                    super().__init__(p_mode=Mode.C_MODE_REAL, p_logging=p_logging)

                self._state_space = WrEnvGYM2MLPro.recognize_space(self._gym_env.observation_space)
                self._action_space = WrEnvGYM2MLPro.recognize_space(self._gym_env.action_space)

        elif not p_sim:
            # Socket for the real robot
            # Check the library
            # try:
            #     import rospy
            #     from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
            #     from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
            # except (ModuleNotFoundError, ImportError) as e:
            pass

    ## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None):

        # 1 Reset Gym environment and determine initial state
        try:
            observation = self._gym_env.reset(seed=p_seed)
        except:
           self._gym_env.seed(p_seed)
           observation = self._gym_env.reset() 
        obs = DataObject(observation)

        # 2 Create state object from Gym observation
        state = State(self._state_space)
        state.set_values(obs.get_data())
        self._set_state(state)

    ## -------------------------------------------------------------------------------------------------
    def simulate_reaction(self, p_state: State, p_action: Action) -> State:

        # 1 Convert action to Gym syntax
        action_sorted = p_action.get_sorted_values()
        dtype = self._gym_env.action_space.dtype

        if (dtype == np.int32) or (dtype == np.int64):
            action_sorted = action_sorted.round(0)

        if action_sorted.size == 1:
            action_gym = action_sorted.astype(self._gym_env.action_space.dtype)[0]
        else:
            action_gym = action_sorted.astype(self._gym_env.action_space.dtype)

        # 2 Process step of Gym environment
        try:
            observation, reward_gym, done, info = self._gym_env.step(action_gym)
        except:
            observation, reward_gym, done, info = self._gym_env.step(np.atleast_1d(action_gym))

        obs = DataObject(observation)

        # 3 Create state object from Gym observation
        state = State(self._state_space, p_terminal=done)
        state.set_values(obs.get_data())

        # 4 Create reward object
        self._last_reward = Reward(Reward.C_TYPE_OVERALL)
        self._last_reward.set_overall_reward(reward_gym)

        # 5 Return next state
        return state

    ## -------------------------------------------------------------------------------------------------
    def compute_reward(self, p_state_old: State = None, p_state_new: State = None) -> Reward:
        if (p_state_old is not None) or (p_state_new is not None):
            raise NotImplementedError

        return self._last_reward

    ## -------------------------------------------------------------------------------------------------
    def compute_broken(self, p_state: State) -> bool:
        return self.get_broken()

    ## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure=None):
        self._gym_env.render()

    ## -------------------------------------------------------------------------------------------------
    def update_plot(self):
        self._gym_env.render()

    ## -------------------------------------------------------------------------------------------------
    def get_cycle_limit(self):
        return self._gym_env._max_episode_steps

    ## -------------------------------------------------------------------------------------------------
    def compute_success(self, p_state: State) -> bool:
        obs = p_state.get_values()
        close = np.allclose(a=obs[:3],
                            b=obs[3:],
                            atol=0.1)

        if close:
            self._state.set_terminal(True)

        return close

    ## -------------------------------------------------------------------------------------------------
    def _build(self):
        logger = Log()
        logger.C_TYPE = "Log"
        logger.C_NAME = "Pre-check ROS"

        # Check OS
        logger.log(Log.C_LOG_TYPE_I, "Checking Operating System ......")
        if platform.system() != "Linux":
            logger.log(Log.C_LOG_TYPE_E, "Operating System is not supported!")
            logger.log(Log.C_LOG_TYPE_E, "Please use Linux")
            logger.log(Log.C_LOG_TYPE_E, "Exiting....")
            sys.exit()
        else:
            logger.log(Log.C_LOG_TYPE_S, "Operating System is supported")

        # Check if ROS is installed
        process = subprocess.run("which roscore", shell=True, stdout=subprocess.PIPE)
        output = process.stdout
        logger.log(Log.C_LOG_TYPE_I, "Checking ROS Installation ......")
        if output==bytes():
            logger.log(Log.C_LOG_TYPE_E, "ROS is not installed!")
            logger.log(Log.C_LOG_TYPE_E, "Please install ROS")
            logger.log(Log.C_LOG_TYPE_E, "Exiting....")
            sys.exit()
        else:
            logger.log(Log.C_LOG_TYPE_S, "ROS is installed")

        import rospkg

        # Check if UR5 Workspace is installed
        installed = False
        rospack = rospkg.RosPack()
        try:
            rospack.get_path("ur5_lab")
        except rospkg.common.ResourceNotFound:
            logger.log(Log.C_LOG_TYPE_E, "UR5 Workspace is not installed!")
            logger.log(Log.C_LOG_TYPE_W, "If you have ran this script, please CTRL+C and restart terminal")
        else:
            installed = True

        if not installed:
            logger.log(Log.C_LOG_TYPE_W, "Building ROS Workspace in 10 Seconds")
            for sec in range(10):
                time.sleep(1)
                logger.log(Log.C_LOG_TYPE_W, str(9-sec)+"...")

            ros_workspace = os.path.dirname(mlpro.__file__)+"/rl/pool/envs/ur5jointcontrol"
            command = "cd " + ros_workspace + " && catkin_make"
            try:
                process = subprocess.check_output(command, shell=True)
            except subprocess.CalledProcessError as e:
                logger.log(Log.C_LOG_TYPE_E, "Build Failed")
                sys.exit()

            logger.log(Log.C_LOG_TYPE_S, "Successfully Built")
            command = "echo 'source "+ros_workspace+"/devel/setup.bash"+"' >> ~/.bashrc"
            process = subprocess.run(command, shell=True)
            logger.log(Log.C_LOG_TYPE_W, "Please restart your terminal and run the Howto script again")
            sys.exit()
        else:
            logger.log(Log.C_LOG_TYPE_S, "UR5 Workspace is installed")