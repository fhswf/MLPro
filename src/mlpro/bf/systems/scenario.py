## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.systems
## -- Module  : scenario.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-02-23  0.0.0     LSB      Creation
## -------------------------------------------------------------------------------------------------
import random

from mlpro.bf.ml import *
from mlpro.bf.various import Log
from mlpro.bf.systems import Action



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SystemScenario(Scenario):


    C_NAME = 'Systems Scenario'

    C_OP_RND = 'Random Operation'
    C_OP_PRE = 'Predefined Operation '


    ## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_mode=Mode.C_MODE_SIM,
                 p_ada:bool = True,
                 p_cycle_limit=0,
                 p_auto_setup:bool = True,
                 p_visualize:bool = True,
                 p_logging = Log.C_LOG_ALL):


        Scenario.__init__(self,
            p_mode=p_mode,
            p_ada=p_ada,
            p_cycle_limit=p_cycle_limit,
            p_auto_setup = p_auto_setup,
            p_visualize=p_visualize,
            p_logging=p_logging)



## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada:bool, p_visualize:bool, p_logging) -> Model:
        """
        Please set system as self._system attribute
        Parameters
        ----------
        p_mode
        p_ada
        p_visualize
        p_logging

        Returns
        -------

        """
        raise NotImplementedError

## -------------------------------------------------------------------------------------------------
    def _run_cycle(self):

        end_of_data = False


        state = self._system.get_state()
        state.set_tstamp(self._timer.get_time())


        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': Action generation started')

        if self._operation_mode == self.C_OP_PRE:
            action = self._agent.compute_action(state)


        else:
            action = Action(p_action_space=self._system.get_action_space(), p_values=np.random.uniform(-10, 10,
                size=(1,)))

        action.set_tstamp(self._timer.get_time())


        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': Env processes action...')
        self._system.process_action(action)
        self._system.get_state().set_tstamp(self._timer.get_time())

        # Since this is a random trial scenario class, setting adaptations to zero
        adapted = False


        if self._operation_mode == self.C_OP_PRE:
            success = self._env.get_state().get_success()
            error = self._env.get_state().get_terminal()

            if success:
                self.log(self.C_LOG_TYPE_S, 'Process time', self._timer.get_time(), ': Environment goal achieved')

            if error:
                self.log(self.C_LOG_TYPE_E, 'Process time', self._timer.get_time(), ': Environment terminated')

        else:
            success = False
            error = False

        return success, error, adapted, end_of_data


    def get_latency(self) -> timedelta:
        return self._system.get_latency()