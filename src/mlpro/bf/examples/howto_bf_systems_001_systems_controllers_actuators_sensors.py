## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_systems_001_systems_controllers_actuators_sensors.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-12-05  1.0.0     DA       Creation
## -- 2022-12-09  1.1.0     DA       Simplification
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2022-12-09)

This module demonstrates the principles of using classes System, Controller, Actuator and Sensor. To
this regard we assume a custom system with two state and action components and a custom controller
that represents the hardware pendant with two sensors and actuators.

You will learn:

1) How to set up an own state based system.

2) How to implement an own controller.

3) How to assign states to sensors and actions to actuators.

4) How to communicate with sensors and actuators using a controller.

"""


from mlpro.bf.various import Log
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




class MySystem (System):

    C_NAME      = 'Dummy'

    @staticmethod
    def setup_spaces():
        
        # 1 State space
        state_space = ESpace()
        state_space.add_dim( p_dim = Dimension( p_name_short='State 1') )
        state_space.add_dim( p_dim = Dimension( p_name_short='State 2') )

        # 2 Action space
        action_space = ESpace()
        action_space.add_dim( p_dim = Dimension( p_name_short='Action 1') )
        action_space.add_dim( p_dim = Dimension( p_name_short='Action 2') )

        return state_space, action_space


    def _reset(self, p_seed=None) -> None:
        pass





# 0 Prepare Demo/Unit test mode
if __name__ == '__main__':
    logging     = Log.C_LOG_ALL
    latency     = timedelta(0,1,0)
else:
    logging     = Log.C_LOG_NOTHING
    latency     = timedelta(0,0,100000)


# 1 Instantiate own system in simulation mode
sys = MySystem( p_latency=latency, p_logging=logging )


# 2 Instantiate and configure own controller
con = MyController( p_id=0, p_name='2x2', p_logging=logging )

s1 = Sensor(p_name_short='Sensor 1')
s2 = Sensor(p_name_short='Sensor 2')
a1 = Actuator(p_name_short='Actuator 1')
a2 = Actuator(p_name_short='Actuator 2')

con.add_sensor( p_sensor=s1 )
con.add_sensor( p_sensor=s2 )
con.add_actuator( p_actuator=a1 )
con.add_actuator( p_actuator=a2 )


# 3 Add controller to system and assign sensors to states and actuators to actions
sys.add_controller( p_controller=con, 
                    p_mapping=[ ( 'S', 'State 1', 'Sensor 1' ), 
                                ( 'S', 'State 2', 'Sensor 2' ), 
                                ( 'A', 'Action 1', 'Actuator 1' ), 
                                ( 'A', 'Action 2', 'Actuator 2') ] )


# 4 Switch system to real mode
sys.set_mode( p_mode=Mode.C_MODE_REAL )


# 5 Reset system
sys.reset()


# 6 Process an action
sys.process_action( p_action= Action( p_agent_id=0, 
                                      p_action_space=sys.get_action_space(),
                                      p_values=np.array([1,2]) ) )


