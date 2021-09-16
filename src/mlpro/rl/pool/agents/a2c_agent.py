## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro.pool.agent
## -- Module  : actorcritic_agent
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-09-16  0.0.0     MRD      Creation
## -- 2021-09-17  1.0.0     MRD      Release test version on seperate branch
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2021-09-17)

This module provides agent to deal with actor critic algorithm.
"""

from mlpro.rl.models import *

class A2CAgent(Agent):
    """
    Implementation of Agent for actor critic algorithm.
    """

    C_NAME      = 'Actor Critic Agent'
    def __init__(self, p_policy: Policy, p_sarbuffer_size, p_envmodel: EnvModel, p_name, p_id, p_ada, p_logging):
        super().__init__(p_policy, p_sarbuffer_size=p_sarbuffer_size, p_envmodel=p_envmodel, p_name=p_name, p_id=p_id, p_ada=p_ada, p_logging=p_logging)

        self._value = None

## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_state: State) -> Action:
        """
        Default implementation of a single agent.
        """

        self.log(self.C_LOG_TYPE_I, 'Action computation: state received = ', p_state.get_values())
        self._previous_state  = self._state
        self._state           = p_state
        self._previous_action, self._value  = self._policy.compute_action(p_state)
        return self._previous_action

## -------------------------------------------------------------------------------------------------
    def adapt(self, *p_args) -> bool:
        """
        Default adaption implementation of a single agent.

        Parameters:
            p_args[0]       Reward object (see class Reward)

        Returns:
            True, if something has beed adapted
        """

        # 1 Check: Adaption possible?
        if self._adaptivity == False:
            self.log(self.C_LOG_TYPE_I, 'Adaption disabled')
            return False

        if self._previous_state is None:
            self.log(self.C_LOG_TYPE_I, 'Adaption: previous state None -> adaptivity skipped')
            return False

        reward = p_args[0]
        done = p_args[1]
        self.log(self.C_LOG_TYPE_I, 'Adaption: previous state =', self._previous_state.get_values(), '; reward = ', p_args[0].get_agent_reward(self._id))

        # 2 Add data to SAR buffer
        self._sar_buffer.add_element(SARBufferElement(self._previous_state, self._previous_action, self._state, reward, done, self._value))

        # 3 Adapt policy if SAR Buffer full
        if self._sar_buffer.is_full():
            return self._policy.adapt(self._sar_buffer)
        else:
            return False