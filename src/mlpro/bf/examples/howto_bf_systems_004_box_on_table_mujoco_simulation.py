## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_systems_004_box_on_table_mujoco_simulation.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-04-10  0.0.0     MRD       Creation
## -------------------------------------------------------------------------------------------------


"""
Ver. 1.0.0 (2023-04-10)

This module demonstrates the principles of using classes System and uses MuJoCo wrapper to simulate 
the pre-defined model. A camera is integrated in the simulation model. The camera is extracted from
the simulation and shown with Matplotlib.

You will learn:
    
1) How to set up a custom System for MuJoCo with custom reset function.

2) How to include the MuJoCo model in the System.

3) How to visualize image data from the simulation with Matplotlib.
    
"""


import random
import mlpro
from mlpro.bf.various import Log
from mlpro.bf.systems import *




# 0 Create Custom System for Custom reset
class RandomBoxOnTable(System):
    def _reset(self, p_seed=None) -> None:
        # Random Box Position
        box_1 = [random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2), 0.95, 1, 0, 0, 0]
        box_2 = [random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2), 0.95, 1, 0, 0, 0]
        box_3 = [random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2), 0.95, 1, 0, 0, 0]
        
        init_qpos = [*box_1, *box_2, *box_3]
        init_qvel = self._mujoco_handler.init_qvel
        reset_state = [[init_qpos], [init_qvel]]
        ob = self._mujoco_handler._reset_simulation(reset_state)
        self._state = State(self.get_state_space())
        self._state.set_values(ob)
        
        
# 11 Prepare Demo/Unit test mode
if __name__ == '__main__':
    logging     = Log.C_LOG_ALL
    visualize   = True
    loop_cycle  = 50
else:
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    loop_cycle  = 10


# 1 Instantiate own system in simulation mode
model_file = os.path.join(os.path.dirname(mlpro.__file__), "bf/systems/pool/mujoco", "boxontable.xml")
sys = RandomBoxOnTable(p_logging=logging, p_mujoco_file=model_file, p_visualize=visualize)

# 2 Reset system
sys.reset()

# 3 Run System
plt.ion()

for x in range(5):
    for y in range(loop_cycle):
        # Random Action
        action = 1
        sys.process_action( p_action= Action( p_agent_id=0, 
                                            p_action_space=sys.get_action_space(),
                                            p_values=action ) )
        
        plt.clf()
        plt.imshow(sys.get_state().get_values()[-1])
        plt.pause(0.01)
        
    sys.reset()
    
plt.ioff()
