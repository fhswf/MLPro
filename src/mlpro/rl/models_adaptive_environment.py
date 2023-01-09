## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl
## -- Module  : models_adaptive_environment.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- yyyy-mm-dd  0.0.0     LSB      Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (yyyy-mm-dd)

This module provides model classes for adaptive environments
"""


from mlpro.oa.models import *
from mlpro.rl.models import *





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAFctSTrans(FctSTrans, OAWorkflow):

    """

        Parameters
        ----------
        p_name
        p_range_max
        p_class_shared
        p_visualize
        p_logging
        p_kwargs
    """


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name: str = None,
                 p_range_max=Async.C_RANGE_THREAD,
                 p_class_shared=None,
                 p_visualize: bool = False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):

        FctSTrans.__init__(self,
                           p_logging=p_logging)

        OAWorkflow.__init__(self,
                            p_name = p_name,
                            p_range_max = p_range_max,
                            p_class_shared = p_class_shared,
                            p_visualize = p_visualize,
                            p_logging=p_logging,
                            p_kwargs=p_kwargs)
        pass


## -------------------------------------------------------------------------------------------------
    def simulate_reaction(self, p_state: State, p_action: Action) -> State:
        """

        Parameters
        ----------
        p_state
        p_action

        Returns
        -------

        """

        self.log(Log.C_LOG_TYPE_I, 'Start simulating a state transition...')

        # 1. Check if there exists a list of pre-processing tasks
        if len(self._tasks) != 0:
            p_inst_new = [Instance(p_feature_data=p_state)]
            self.get_so().reset(p_inst_new)
            self.run(p_inst_new = p_inst_new)
            p_state = self.get_so().get_result(p_tid=self._tasks[-1])

        return self._simulate_reaction( p_state = p_state, p_action = p_action )


## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State, p_action: Action) -> State:
        """

        Parameters
        ----------
        p_state
        p_action

        Returns
        -------

        """
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _run( self,
              p_inst_new : list,
              p_inst_del : list ):
        """

        Parameters
        ----------
        p_inst_new
        p_inst_del

        Returns
        -------

        """
        pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAFctReward(FctReward, OAWorkflow):
    """

    Parameters
    ----------
    p_name
    p_range_max
    p_class_shared
    p_visualize
    p_logging
    p_kwargs
    """


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name: str = None,
                 p_range_max=Async.C_RANGE_THREAD,
                 p_class_shared=None,
                 p_visualize: bool = False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):


        FctReward.__init__(self,
                           p_logging = p_logging)

        OAWorkflow.__init__(self,
                            p_name = p_name,
                            p_range_max = p_range_max,
                            p_class_shared = p_class_shared,
                            p_visualize = p_visualize,
                            p_logging = p_logging,
                            p_kwargs = p_kwargs)


## -------------------------------------------------------------------------------------------------
    def compute_reward(self, p_state: State = None, p_state_new: State = None) -> Reward:
        """

        Parameters
        ----------
        p_state
        p_state_new

        Returns
        -------

        """
        self.log(Log.C_LOG_TYPE_I, 'Start Computing the Reward....')

        # 1. Check if there exists a list of pre-processing tasks
        if len(self._tasks) != 0:
            p_inst_new = [Instance(p_feature_data=p_state_new), Instance(p_feature_data=p_state)]
            self.get_so().reset(p_inst_new)
            self.run(p_inst_new=p_inst_new)
            state_new,state = self.get_so().get_result(p_tid=self._tasks[-1])
        else:
            state = p_state
            state_new = p_state_new

        return self._compute_reward(p_state=state, p_state_new=state_new)

    ## -------------------------------------------------------------------------------------------------
    def _compute_reward(self, p_state: State = None, p_state_new: State = None) -> Reward:
        """

        Parameters
        ----------
        p_state
        p_state_new

        Returns
        -------

        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _run( self,
              p_inst_new : list,
              p_inst_del : list ):
        """

        Parameters
        ----------
        p_inst_new
        p_inst_del

        Returns
        -------

        """
        pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAFctSuccess(FctSuccess, OAWorkflow):
    """

    Parameters
    ----------
    p_name
    p_range_max
    p_class_shared
    p_visualize
    p_logging
    p_kwargs
    """


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name: str = None,
                 p_range_max=Async.C_RANGE_THREAD,
                 p_class_shared=None,
                 p_visualize: bool = False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs
                 ):

        """

        Parameters
        ----------
        p_name
        p_range_max
        p_class_shared
        p_visualize
        p_logging
        p_kwargs
        """

        FctSuccess.__init__(self,
                            p_logging = p_logging)

        OAWorkflow.__init__(self,
                            p_name = p_name,
                            p_range_max = p_range_max,
                            p_class_shared = p_class_shared,
                            p_visualize = p_visualize,
                            p_logging = p_logging,
                            p_kwargs = p_kwargs)


## -------------------------------------------------------------------------------------------------
    def compute_success(self, p_state: State) -> bool:
        """

        Parameters
        ----------
        p_state

        Returns
        -------

        """
        self.log(Log.C_LOG_TYPE_I, 'Start simulating a state transition...')

        # 1. Check if there exists a list of pre-processing tasks
        if len(self._tasks) != 0:
            p_inst_new = [Instance(p_feature_data=p_state)]
            self.get_so().reset(p_inst_new)
            self.run(p_inst_new=p_inst_new)
            p_state = self.get_so().get_result(p_tid=self._tasks[-1])

        return self._compute_success(p_state = p_state)

    ## -------------------------------------------------------------------------------------------------
    def _compute_success(self, p_state: State) -> bool:
        """

        Parameters
        ----------
        p_state

        Returns
        -------

        """
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _run( self,
              p_inst_new : list,
              p_inst_del : list ):
        """

        Parameters
        ----------
        p_inst_new
        p_inst_del

        Returns
        -------

        """
        pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAFctBroken(FctBroken, OAWorkflow):
    """

    Parameters
    ----------
    p_name
    p_range_max
    p_class_shared
    p_visualize
    p_logging
    p_kwargs
    """


    ## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name:str=None,
                 p_range_max=Async.C_RANGE_THREAD,
                 p_class_shared=None,
                 p_visualize:bool=False,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):


        FctBroken.__init__(self,
                            p_logging=p_logging)

        OAWorkflow.__init__(self,
                            p_name=p_name,
                            p_range_max=p_range_max,
                            p_class_shared=p_class_shared,
                            p_visualize=p_visualize,
                            p_logging=p_logging,
                            p_kwargs=p_kwargs)


## -------------------------------------------------------------------------------------------------
    def compute_broken(self, p_state: State) -> bool:
        """

        Parameters
        ----------
        p_state

        Returns
        -----

        """
        self.log(Log.C_LOG_TYPE_I, 'Start simulating a state transition...')

        # 1. Check if there exists a list of pre-processing tasks
        if len(self._tasks) != 0:
            p_inst_new = [Instance(p_feature_data=p_state)]
            self.get_so().reset(p_inst_new)
            self.run(p_inst_new=p_inst_new)
            state_values = self.get_so().get_result(p_tid=self._tasks[-1])
            state = State(p_state_space=p_state.get_related_set())

        return self._compute_broken(p_state = p_state)

    ## -------------------------------------------------------------------------------------------------
    def _compute_broken(self, p_state: State) -> bool:
        """

        Parameters
        ----------
        p_state

        Returns
        -------

        """
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _run( self,
              p_inst_new : list,
              p_inst_del : list ):
        """

        Parameters
        ----------
        p_inst_new
        p_inst_del

        Returns
        -------

        """
        pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AdaptiveEnvironment(Environment, Model):
    """

    Parameters
    ----------
    p_mode
    p_latency
    p_ada
    p_fct_strans
    p_fct_reward
    p_fct_success
    p_fct_broken
    p_visualize
    p_logging
    """


    ## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_mode=Mode.C_MODE_SIM,
                 p_latency: timedelta = None,
                 p_ada: bool = True,
                 p_fct_strans: FctSTrans = None,
                 p_fct_reward: FctReward = None,
                 p_fct_success: FctSuccess = None,
                 p_fct_broken: FctBroken = None,
                 p_visualize: bool = False,
                 p_logging=Log.C_LOG_ALL
                 ):


        Environment.__init__(self,
                             p_mode=p_mode,
                             p_latency = p_latency,
                             p_fct_strans=p_fct_strans,
                             p_fct_reward = p_fct_reward,
                             p_fct_success = p_fct_success,
                             p_fct_broken= p_fct_broken,
                             p_visualize = p_visualize,
                             p_logging=p_logging)

        Model.__init__(self,
                       p_logging=p_logging,
                       p_ada = p_ada,
                       p_visualize=p_visualize)




## -------------------------------------------------------------------------------------------------
    def _set_adapted(self, p_adapted:bool):
        """

        Parameters
        ----------
        p_adapted

        Returns
        -------

        """
        pass


