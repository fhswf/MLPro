## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_systems_003_cartpole_continuous_systems_wrapped_with_mujoco.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-02-23  0.0.0     MRD       Creation
## -- 2023-02-23  1.0.0     MRD       Release
## -- 2023-03-07  1.0.1     MRD       Remove CartPoleSystem Class
## -------------------------------------------------------------------------------------------------


"""
Ver. 1.0.1 (2023-03-07)

This module demonstrates the principles of using classes System and uses MuJoCo wrapper to simulate 
the pre-defined model.

You will learn:
    
1) How to set up a Cartpole Continuous System wrapped with MuJoCo

2) How to run the system
    
"""


import mlpro
from mlpro.bf.various import Log
from mlpro.bf.systems import *




# 0 Prepare Demo/Unit test mode
if __name__ == '__main__':
    logging     = Log.C_LOG_ALL
    visualize   = True
    loop_cycle  = 1000
else:
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    loop_cycle  = 100


# 1 Instantiate own system in simulation mode
model_file = os.path.join(os.path.dirname(mlpro.__file__), "bf/systems/pool/mujoco", "cartpole.xml")
sys = System(p_logging=logging, p_mujoco_file=model_file, p_visualize=visualize)

# 2 Reset system
sys.reset()

# 3 Process an action
for x in range(loop_cycle):
    # Random Action
    action = np.random.uniform(-50, 50, size=(1,))
    sys.process_action( p_action= Action( p_agent_id=0, 
                                        p_action_space=sys.get_action_space(),
                                        p_values=action ) )


