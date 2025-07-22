## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.systems.pool
## -- Module  : mutli_flipflops.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-03-05  0.0.0     LSB       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2023-06-08)

This module provides the implementations for multi-flipflops multisystem.
"""

from mlpro.bf.systems import *





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Flipflop(System):
    """

    """
    C_NAME = 'FlipFlop'

    C_STATE_ZERO = 0
    C_STATE_ONE = 1

    C_ACTION_ZERO = 0       # Change the state to Zero
    C_ACTION_ONE = 1        # Change the inner state to One
    C_ACTION_TOGGLE = -1    # Toggle the inner state


## -------------------------------------------------------------------------------------------------
    @staticmethod
    def setup_spaces():
        """
        This method set's up the state and action space of the System.

        Returns
        -------
        state_space, action_space

        """
        action_space = ESpace()
        state_space = ESpace()


        state_space.add_dim(Dimension(p_name_long='State', p_name_short='S', p_boundaries=[0,1], p_base_set=Dimension.C_BASE_SET_Z,
                                      p_description='Internal State of the flip flop'))

        action_space.add_dim(Dimension(p_name_long='input signal', p_name_short='i/p', p_boundaries=[-1,1],
                                      p_description='I/p signal to the flip flop. Acceped values [-1,0,1]', p_base_set=Dimension.C_BASE_SET_Z))


        return state_space, action_space


## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State, p_action: Action, p_step:timedelta = None) -> State:
        """

        Parameters
        ----------
        p_state
        p_action
        p_step

        Returns
        -------

        """
        state = p_state.get_values()

        for action in p_action.get_sorted_values():
            if action == self.C_ACTION_ZERO:
                state = self.C_STATE_ZERO

            if action == self.C_ACTION_ONE:
                state = self.C_STATE_ONE

            if action == self.C_ACTION_TOGGLE:
                if state == self.C_STATE_ONE:
                    state = self.C_STATE_ZERO
                else:
                    state = self.C_STATE_ONE

        # print('state:',state)
        current_state = State(p_state_space=self.get_state_space())

        current_state.set_values([state])

        return current_state


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed = None):
        """

        Parameters
        ----------
        p_seed

        Returns
        -------

        """
        state_values = random.randint(0,1)
        state = State(self.get_state_space())
        state.set_values([state_values])
        self._set_state(state)







