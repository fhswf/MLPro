## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : howto_rl_011a_load_and_run_ur5_environment.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-06-14  0.0.0     MRD      Creation
## -- 2022-06-14  1.0.0     MRD      Initial Release
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2022-06-14)

This module shows how to load trained policy for UR5 robot (derivate for paper).
"""

from mlpro.rl.models import *
from mlpro.rl.pool.envs.ur5jointcontrol import UR5JointControl


# 1 Implement your own RL scenario
class ScenarioUR5A2C(RLScenario):
    C_NAME = 'Matrix'

    def _setup(self, p_mode, p_ada, p_logging):
        # 1.1 Setup environment
        self._env = UR5JointControl(
            p_build=True, 
            p_real=p_mode,
            p_start_simulator=True,
            p_start_ur_driver=True,
            # p_ros_server_ip="172.19.10.199",
            p_net_interface="enp0s31f6",
            p_robot_ip="172.19.10.41",
            # p_reverse_ip="172.19.10.140", 
            p_visualize=self._visualize, 
            p_logging=p_logging)

        return self.load("/home/at-lab/MLPRO/MLPro/src/mlpro/rl/examples", "trained_policy.pkl")


# 6 Instatiate new scenario
scenario = ScenarioUR5A2C(p_mode=Mode.C_MODE_REAL, 
                        p_ada=False,
                        p_cycle_limit=10,
                        p_visualize=False,
                        p_logging=Log.C_LOG_WE)


# 7 Reset Scenario
scenario.reset()  


# 8 Run Scenario
scenario.run()
