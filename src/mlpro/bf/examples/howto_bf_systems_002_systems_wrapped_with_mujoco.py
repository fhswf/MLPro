## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_systems_002_systems_wrapped_with_mujoco.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-01-06  0.0.0     MRD       Creation
## -- 2023-01-06  1.0.0     MRD       Release
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-01-06)

This module demonstrates the principles of using classes System and use MuJoCo wrapper to do
the simulation for pre defined model.
"""


import mlpro
from mlpro.bf.various import Log
from mlpro.wrappers.mujoco import WrMujoco
from mlpro.bf.systems import *
import random



class MyController (Controller):

    C_NAME      = 'Dummy'

    def _reset(self) -> bool:
        self.log(Log.C_LOG_TYPE_S, 'Pseudo-reset of the controller')
        return True


    def _get_sensor_value(self, p_id):
        """
        Pseudo-implementation for getting a sensor value.
        """
        self.log(Log.C_LOG_TYPE_S, 'Pseudo-import of a sensor value...')
        return random.random()


    def _set_actuator_value(self, p_id, p_value) -> bool:
        self.log(Log.C_LOG_TYPE_S, 'Pseudo-export of an actuator value:', str(p_value))
        return True




class PendulumSystem (System):

    C_NAME      = 'PendulumSystem'
    C_PLOT_ACTIVE       = True

    def __init__(self, p_mode=Mode.C_MODE_SIM, p_latency: timedelta = None, p_fct_strans: FctSTrans = None, p_fct_success: FctSuccess = None, p_fct_broken: FctBroken = None, p_visualize: bool = False, p_logging=Log.C_LOG_ALL):
        super().__init__(p_mode, p_latency, p_fct_strans, p_fct_success, p_fct_broken, p_visualize, p_logging)
        self._state = State(self._state_space)

    @staticmethod
    def setup_spaces():
        
        # 1 State space
        state_space = ESpace()
        state_space.add_dim( p_dim = Dimension( p_name_short='State 1') )

        # 2 Action space
        action_space = ESpace()
        action_space.add_dim( p_dim = Dimension( p_name_short='Action 1') )

        return state_space, action_space


    def _reset(self, p_seed=None) -> None:
        pass


# 0 Prepare Demo/Unit test mode
if __name__ == '__main__':
    logging     = Log.C_LOG_ALL
    latency     = timedelta(0,0.01,0)
else:
    logging     = Log.C_LOG_NOTHING
    latency     = timedelta(0,0,100000)


# 1 Instantiate own system in simulation mode
sys = PendulumSystem(p_latency=latency, p_logging=logging, p_visualize=True)

# 2 Wrapped with MuJoCo with pendulum model
model_path = os.path.join(os.path.dirname(mlpro.__file__), "rl/pool/envs/mujoco/assets", "pendulum.xml")
sys = WrMujoco(sys, p_model_file=model_path, p_system_type=WrMujoco.C_SYSTEM)

# 3 Reset system
sys.reset()

# 4 Init Visualization
sys.init_plot()

# 6 Process an action
for x in range(2000):
    # Random Action
    action = np.random.uniform(-1, 1, size=(1,))
    sys.process_action( p_action= Action( p_agent_id=0, 
                                        p_action_space=sys.get_action_space(),
                                        p_values=action ) )
    sys.update_plot()


