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
## -- 2022-06-07  2.0.1     MRD      Define _export_action and _import_action for each communication type
## -------------------------------------------------------------------------------------------------

"""
Ver. 2.0.1 (2022-06-07)

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
from mlpro.wrappers.openai_gym import *
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
    C_LATENCY = timedelta(0, 0, 0)
    C_INFINITY = np.finfo(np.float32).max
    C_COM_MODE_ROS = 0
    C_COM_MODE_PLAIN = 1

    ## -------------------------------------------------------------------------------------------------
    def __init__(self, p_seed=0, p_build=True, p_real=False, p_com_method=C_COM_MODE_ROS, p_robot_ip="", p_reverse_ip="", 
        p_reverse_port=50001, p_visualize=False, p_logging=True):
        """
        Parameters:
            p_logging       Boolean switch for logging
        """

        # Use ROS as the simulation and real
        self._com_method = p_com_method
        if self._com_method == self.C_COM_MODE_ROS:
            if not p_real:
                super().__init__(p_mode=Mode.C_MODE_SIM, p_logging=p_logging)
            else:
                super().__init__(p_mode=Mode.C_MODE_REAL, p_logging=p_logging)
                self._real_ros_state = None
                self._export_action = self._export_action_ros
                self._import_state = self._import_state_ros
            self._compute_reward = self._compute_reward_ros
            self._reset = self._reset_ros

            if p_build:
                self._build()

            try:
                import rospy
                from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
                from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
            except (ModuleNotFoundError, ImportError) as e:
                self.log(Log.C_LOG_TYPE_E, "ROS or UR5 Workspace is not installed")
                self.log(Log.C_LOG_TYPE_W, "Please use the guideline here:")
                self.log(Log.C_LOG_TYPE_W, "https://mlpro.readthedocs.io/en/latest/content/rl/env/pool/ur5jointcontrol.html")
            else:
                roscore = subprocess.Popen('roscore')
                rospy.init_node('ur5_lab_training_start', anonymous=True, log_level=rospy.WARN)

                LoadYamlFileParamsTest(rospackage_name="ur5_lab",
                                    rel_path_from_package_to_file="config",
                                    yaml_file_name="ur5_lab_task_param.yaml")

                ros_ws_path = mlpro.rl.pool.envs.ur5jointcontrol.__file__.replace("/__init__.py", "")
                rospy.set_param('ros_ws_path', ros_ws_path)
                rospy.set_param('sim', not p_real)

                # Init OpenAI_ROS ENV
                if not p_real:
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

                self._state_space = WrEnvGYM2MLPro.recognize_space(self._gym_env.observation_space)
                self._action_space = WrEnvGYM2MLPro.recognize_space(gym.spaces.Box(low=-0.1, high=0.1, shape=(6,)))
        
        # Use plain as the real
        elif p_real:
            # Check the library
            try:
                from fhswf_at_ia.hc.pool.hardware.ur5_base import UR5Base
            except (ModuleNotFoundError, ImportError) as e:
                self.log(Log.C_LOG_TYPE_E, "Hardware Control Library is not found!")

            super().__init__(p_mode=Mode.C_MODE_REAL, p_logging=p_logging)

            self._export_action = self._export_action_plain
            self._import_state = self._import_state_plain
            self._compute_reward = self._compute_reward_plain
            self._reset = self._reset_plain

            # Initialize communication with the real
            self.ur5     = UR5Base(p_logging=True)

            self.ur5.set_connection_param(p_pc_ip  = p_reverse_ip,
                        p_pc_port = 31001,
                        p_robot_ip = p_robot_ip,
                        p_robot_port = 30002,
                        p_timeout = 10.0)

            # Connect to Robot
            try:
                con = self.ur5.connect()
            except:
                self.log(Log.C_LOG_TYPE_E, "Failed during establishing connection")
                raise ConnectionError
            else:
                if con:
                    self.log(Log.C_LOG_TYPE_S, "Connected to the robot")
                else:
                    self.log(Log.C_LOG_TYPE_E, "Cannot connect to the robot")
                    raise ConnectionError

            # Create state space from controller
            self._state_space = self.ur5.get_sensor_space()

            # Create action space from controller
            self._action_space = WrEnvGYM2MLPro.recognize_space(gym.spaces.Box(low=-0.1, high=0.1, shape=(6,)))

            # Set Maximum Step
            self._max_step_episode = 10

            # Set reset position
            self._reset_pos_plain = [0.0, -1.0, -1.15, -1.30, 1.57, 1.57]

            # Set goal position
            self._goal_pos_plain = [0.29796, -0.14046, 0.10748]

            # Initial Design
            self.init_distance = None
        else:
            raise NotImplementedError

    ## -------------------------------------------------------------------------------------------------
    @staticmethod
    def setup_spaces():
        return None, None


    ## -------------------------------------------------------------------------------------------------
    def _export_action_ros(self, p_action: Action) -> bool:
        try:
            self._real_ros_state = self.simulate_reaction(None, p_action)  
        except Exception as e:
            self.log(Log.C_LOG_TYPE_E, e)
            return False
        else:
            return True

    ## -------------------------------------------------------------------------------------------------
    def _export_action_plain(self, p_action: Action) -> bool:
        action_sorted = p_action.get_sorted_values()
        try:
            current_joint = np.array(self.ur5.get_joints())
            next_joint = current_joint + action_sorted
            next_joint = np.clip(next_joint, -math.pi, math.pi)
            self.ur5.move_joints(next_joint.tolist())
        except Exception as e:
            self.log(Log.C_LOG_TYPE_E, e)
            return False
        else:
            return True

    ## -------------------------------------------------------------------------------------------------
    def _import_state_ros(self) -> bool:
        try:
            self._set_state(self._real_ros_state)  
        except Exception as e:
            self.log(Log.C_LOG_TYPE_E, e)
            return False
        else:
            return True

    ## -------------------------------------------------------------------------------------------------
    def _import_state_plain(self) -> bool:
        try:
            observation = self.ur5.get_tcp()
        except Exception as e:
            self.log(Log.C_LOG_TYPE_E, e)
            return False
        else:
            observation = observation[:3]
            observation.extend(self._goal_pos_plain)
            obs = DataObject(observation)
            state = State(self._state_space)
            state.set_values(obs.get_data())
            self._set_state(state)
            return True
    
    def _reset_plain(self, p_seed=None):
        # 1 Reset Gym environment and determine initial state
        try:
            self.ur5.move_joints(self._reset_pos_plain)
        except Exception as e:
            self.log(Log.C_LOG_TYPE_E, e)
            return False
        else:
            try:
                observation = self.ur5.get_tcp()
            except:
                return False
            else:
                observation = observation[:3]
                observation.extend(self._goal_pos_plain)
                self.init_distance = np.linalg.norm(np.array(observation[:3]) - np.array(observation[3:]))
                obs = DataObject(observation)
                state = State(self._state_space)
                state.set_values(obs.get_data())
                self._set_state(state)

    ## -------------------------------------------------------------------------------------------------
    def _reset_ros(self, p_seed=None):
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
    def _simulate_reaction(self, p_state: State, p_action: Action) -> State:

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
    def _compute_reward_plain(self, p_state_old: State = None, p_state_new: State = None) -> Reward:
        obs = p_state_new.get_values()
        distance = np.linalg.norm(np.array(obs[:3]) - np.array(obs[3:]))
        ratio = distance / self.init_distance
        reward = -np.ones(1) * ratio
        reward = reward - 10e-2
        
        rewards = Reward(Reward.C_TYPE_OVERALL)
        rewards.set_overall_reward(reward)
        return rewards

    ## -------------------------------------------------------------------------------------------------
    def _compute_reward_ros(self, p_state_old: State = None, p_state_new: State = None) -> Reward:
        return self._last_reward

    ## -------------------------------------------------------------------------------------------------
    def _compute_broken(self, p_state: State) -> bool:
        return self.get_broken()

    ## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure=None):
        pass

    ## -------------------------------------------------------------------------------------------------
    def update_plot(self):
        pass

    ## -------------------------------------------------------------------------------------------------
    def get_cycle_limit(self):
        if self._com_method == self.C_COM_MODE_ROS:
            return self._gym_env._max_episode_steps
        else:
            return self._max_step_episode

    ## -------------------------------------------------------------------------------------------------
    def _compute_success(self, p_state: State) -> bool:
        obs = p_state.get_values()
        close = np.allclose(a=obs[:3],
                            b=obs[3:],
                            atol=0.1)

        if close:
            self._state.set_terminal(True)

        return close

    ## -------------------------------------------------------------------------------------------------
    def _build(self):
        # Check OS
        self.log(Log.C_LOG_TYPE_I, "Checking Operating System ......")
        if platform.system() != "Linux":
            self.log(Log.C_LOG_TYPE_E, "Operating System is not supported!")
            self.log(Log.C_LOG_TYPE_E, "Please use Linux")
            self.log(Log.C_LOG_TYPE_E, "Exiting....")
            sys.exit()
        else:
            self.log(Log.C_LOG_TYPE_S, "Operating System is supported")

        # Check if ROS is installed
        process = subprocess.run("which roscore", shell=True, stdout=subprocess.PIPE)
        output = process.stdout
        self.log(Log.C_LOG_TYPE_I, "Checking ROS Installation ......")
        if output==bytes():
            self.log(Log.C_LOG_TYPE_E, "ROS is not installed!")
            self.log(Log.C_LOG_TYPE_E, "Please install ROS")
            self.log(Log.C_LOG_TYPE_E, "Exiting....")
            sys.exit()
        else:
            self.log(Log.C_LOG_TYPE_S, "ROS is installed")

        import rospkg

        # Check if UR5 Workspace is installed
        installed = False
        rospack = rospkg.RosPack()
        try:
            rospack.get_path("ur5_lab")
        except rospkg.common.ResourceNotFound:
            self.log(Log.C_LOG_TYPE_E, "UR5 Workspace is not installed!")
            self.log(Log.C_LOG_TYPE_W, "If you have ran this script, please CTRL+C and restart terminal")
        else:
            installed = True

        if not installed:
            self.log(Log.C_LOG_TYPE_W, "Building ROS Workspace in 10 Seconds")
            for sec in range(10):
                time.sleep(1)
                self.log(Log.C_LOG_TYPE_W, str(9-sec)+"...")

            ros_workspace = os.path.dirname(mlpro.__file__)+"/rl/pool/envs/ur5jointcontrol"
            command = "cd " + ros_workspace + " && catkin_make"
            try:
                process = subprocess.check_output(command, shell=True)
            except subprocess.CalledProcessError as e:
                self.log(Log.C_LOG_TYPE_E, "Build Failed")
                sys.exit()

            self.log(Log.C_LOG_TYPE_S, "Successfully Built")
            command = "echo 'source "+ros_workspace+"/devel/setup.bash"+"' >> ~/.bashrc"
            process = subprocess.run(command, shell=True)
            self.log(Log.C_LOG_TYPE_W, "Please restart your terminal and run the Howto script again")
            sys.exit()
        else:
            self.log(Log.C_LOG_TYPE_S, "UR5 Workspace is installed")