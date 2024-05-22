## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_systems_021_multisystem_flipflop.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-07  0.0.0     LSB      Creation
## -------------------------------------------------------------------------------------------------

"""
MultiSystem ...

You will learn:

1. ..

2. ..

"""

from mlpro.bf.systems import *
from mlpro.bf.systems.pool.flipflops import Flipflop


# Checking for dark mode:
if __name__ == '__main__':
    logging = Log.C_LOG_ALL
    visualize = True
    cycle_limit = 10
else:
    logging = Log.C_LOG_NOTHING
    visualize = False
    cycle_limit = 2




# 1. Creating subsystems
sub_system_1 = Flipflop(p_logging=logging)

sub_system_2 = Flipflop(p_logging=logging)



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MultiFlipFlop(MultiSystem):

    C_NAME = 'MultiFlipFlop'

## -------------------------------------------------------------------------------------------------
    @staticmethod
    def setup_spaces():
        action_space = ESpace()
        state_space = ESpace()

        state_space.add_dim(Dimension(p_name_long='State', p_name_short='S', p_boundaries=[0, 1],
                                      p_base_set=Dimension.C_BASE_SET_Z, p_description='Internal State of the flip flop'))

        action_space.add_dim(Dimension(p_name_long='input signal', p_name_short='i/p', p_boundaries=[-1, 1],
                                       p_description='I/p signal to the flip flop. Acceped values [-1,0,1]',
                                       p_base_set=Dimension.C_BASE_SET_Z))

        return state_space, action_space




# Creating MultiSystem Object
system = MultiFlipFlop(p_logging=logging)


# Add systems to MultiSystem
system.add_system(p_system=sub_system_1,
                  p_mappings=[(('A', 'A'),
                               (system.get_id(), system.get_action_space().get_dim_ids()[0]),
                               (sub_system_1.get_id(), sub_system_1.get_action_space().get_dim_ids()[0]))])

system.add_system(p_system=sub_system_2,
                  p_mappings=[(('S', 'A'),
                               (sub_system_1.get_id(), sub_system_1.get_state_space().get_dim_ids()[0]),
                               (sub_system_2.get_id(), sub_system_2.get_action_space().get_dim_ids()[0])),

                              (('S', 'S'),
                               (sub_system_2.get_id(), sub_system_2.get_state_space().get_dim_ids()[0]),
                               (system.get_id(), system.get_state_space().get_dim_ids()[0]))])


# Create the demo scenario object
scenario = DemoScenario(p_system=system,
                        p_mode=Mode.C_MODE_SIM,
                        p_action_pattern=DemoScenario.C_ACTION_RANDOM,
                        p_cycle_limit=cycle_limit,
                        p_visualize=visualize,
                        p_logging=logging)


# Run the scenario
scenario.run()