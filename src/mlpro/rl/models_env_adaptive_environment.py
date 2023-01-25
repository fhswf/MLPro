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

        self._kwargs = p_kwargs.copy()


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
        # if len(self._tasks) != 0:
        #     p_inst_new = [Instance(p_feature_data=p_state)]
        #     self.get_so().reset(p_inst_new)
        #     self.run(p_inst_new = p_inst_new)
        #     p_state = self.get_so().get_result(p_tid=self._tasks[-1])
        #     return p_state
        #
        # return self._simulate_reaction( p_state = p_state, p_action = p_action )


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
        # for inst in p_inst_new:
        #     p_state = inst.get_feature_data()
        #     p_state_new = self._simulate_reaction(p_state)






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAFctReward(FctReward, OAWorkflow):
    """
    Online adaptive function for reward computation.
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
                 p_class_shared: StreamShared=None,
                 p_visualize: bool = False,
                 p_logging=Log.C_LOG_ALL,
                 p_processing_wf: StreamWorkflow = None,
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

        self._processing_wf:StreamWorkflow = p_processing_wf
        self._shared = p_class_shared
        self._logging = p_logging
        self._visualize = p_visualize
        self._instance_new = None
        self._instance_old = None
        self._state:State = None
        self._state_new:State = None
        self._reward_task:StreamTask = None


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


        # 1. check if the user has already created a workflow and added to tasks
        if self._processing_wf is None:
            # Create a shared object if not provided
            if self._shared is None:
                self._shared = StreamShared()

            # Create an OA workflow
            self._processing_wf = OAWorkflow(p_name = 'Reward Wf', p_class_shared=self._shared)


        # 2. Create a reward task
        if self._reward_task is None:

            # Create a pseudo reward task
            self._reward_task = OATask(p_name='Compute Reward',
                                       p_visualize = self._visualize,
                                       p_range_max=self.get_range(),
                                       p_duplicate_data = True)

            # Assign the task method to custom implementation
            self._reward_task._run = self._run

            # Add the task to workflow
            self._processing_wf.add_task(self._reward_task)



        # 4. Creating task level attributes for states
        try:
            self._reward_task._state
        except AttributeError:
            self._reward_task._state = p_state.copy()
        try:
            self._reward_task._state_new
        except AttributeError:
            self._reward_task._state_new = p_state_new.copy()



        # 5. Creating new and old instances
        # creating old instance object if this is the first run
        if self._instance_new == None:
            self._instance_old = Instance(p_state)

        # assigning the previous new instance to old instance
        else:
            self._instance_old = self._instance_new.copy()

        # creating new instance with new state
        self._instance_new = Instance(p_state_new)


        # 6. Run the workflow
        self._processing_wf.run(p_inst_new=[self._instance_new, self._instance_old])

        # 7. Return the results
        return self._processing_wf.get_so().get_results()[self._reward_task.get_tid()]



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
    def _run(self, p_inst_new, p_inst_del):
        """

        Parameters
        ----------
        p_inst_new
        p_inst_del

        Returns
        -------

        """
        self._state.set_values(p_inst_new[0].get_feature_data().get_values())
        self._state_new.set_values(p_inst_new[1].get_feature_data().get_values())

        return self._compute_reward(p_state= self._state, p_state_new= self._state_new)





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

        # # 1. Check if there exists a list of pre-processing tasks
        # if len(self._tasks) != 0:
        #     p_inst_new = [Instance(p_feature_data=p_state)]
        #     self.get_so().reset(p_inst_new)
        #     self.run(p_inst_new=p_inst_new)
        #     p_state = self.get_so().get_result(p_tid=self._tasks[-1])
        #
        # return self._compute_success(p_state = p_state)

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
        # if len(self._tasks) != 0:
        #     p_inst_new = [Instance(p_feature_data=p_state)]
        #     self.get_so().reset(p_inst_new)
        #     self.run(p_inst_new=p_inst_new)
        #     state_values = self.get_so().get_result(p_tid=self._tasks[-1])
        #     state = State(p_state_space=p_state.get_related_set())

        # return self._compute_broken(p_state = p_state)

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


