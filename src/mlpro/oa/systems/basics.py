## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa.systems
## -- Module  : systems.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-mm-mm  0.0.0     LSB      Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2023-02-16)

This module provides modules and template classes for adaptive systems and adaptive functions.
"""





from mlpro.bf.ml.systems import *
from mlpro.bf.systems import *
from mlpro.bf.ml import Model
from mlpro.bf.streams import *
from mlpro.oa.streams import *



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAFctSTrans(AFctSTrans):
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
              p_processing_wf: StreamWorkflow = None,
              p_logging=Log.C_LOG_ALL,
              **p_kwargs):

        AFctSTrans.__init__(self,
              p_name=p_name,
              p_range_max=p_range_max,
              p_class_shared=p_class_shared,
              p_visualize=p_visualize,
              p_logging=p_logging,
              **p_kwargs)

        self._processing_wf = p_processing_wf
        self._strans_task:StreamTask = None
        self._instance: Instance = None
        self._shared = p_class_shared
        self._state:State = None
        self._action:Action = None


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
        self.log(Log.C_LOG_TYPE_I, 'Reaction Simulation Started...')


        # 1. check if the user has already created a workflow and added to tasks
        if self._processing_wf is None:
            # Create a shared object if not provided
            if self._shared is None:
                self._shared = StreamShared()

            # Create an OA workflow
            self._processing_wf = OAWorkflow(p_name='State Transition Wf', p_class_shared=self._shared)


        # 2. Create a reward task
        if self._strans_task is None:
            # Create a pseudo reward task
            self._strans_task = OATask(p_name='Simulate Reaction',
                p_visualize=self._visualize,
                p_range_max=self.get_range(),
                p_duplicate_data=True)

            # Assign the task method to custom implementation
            self._strans_task._run = self._run

            # Add the task to workflow
            self._processing_wf.add_task(self._strans_task)


        # 4. Creating task level attributes for states
        try:
            self._strans_task._state
        except AttributeError:
            self._strans_task._state = p_state.copy()


        # 5. creating new instance with new state
        self._instance = Instance(p_state)


        # 6. Run the workflow
        self._processing_wf.run(p_inst_new=[self._instance])


        # 7. Return the results
        return self._processing_wf.get_so().get_results()[self._strans_task.get_tid()]


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
    def _adapt(self, **p_kwargs) -> bool:
        """

        Parameters
        ----------
        p_kwargs

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def _adapt_on_event(self, p_event_id:str, p_event_object:Event) -> bool:
        """

        Parameters
        ----------
        p_event_id
        p_event_object

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def add_task(self, p_task:StreamTask):
        """

        Parameters
        ----------
        p_task

        Returns
        -------

        """
        pass


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
        self._state.set_values(p_inst_new.get_feature_data().get_values())

        return self._simulate_reaction(p_state=self._state, p_action=self._action)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAFctSuccess(FctSuccess):
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
              p_processing_wf:StreamWorkflow = None,
              p_logging=Log.C_LOG_ALL,
              **p_kwargs):

        AFctSuccess.__init__(self,
              p_name=p_name,
              p_range_max=p_range_max,
              p_class_shared=p_class_shared,
              p_visualize=p_visualize,
              p_logging=p_logging,
              **p_kwargs)

        Model.__init__()

        self._shared = p_class_shared
        self._processing_wf = p_processing_wf
        self._success_task = None
        self._instance:Instance = None
        self._state:State = None

## -------------------------------------------------------------------------------------------------
    def compute_success(self, p_state: State) -> bool:
        """

        Parameters
        ----------
        p_state

        Returns
        -------

        """
        self.log(Log.C_LOG_TYPE_I, 'Assessing Success...')

        # 1. check if the user has already created a workflow and added to tasks
        if self._processing_wf is None:
            # Create a shared object if not provided
            if self._shared is None:
                self._shared = StreamShared()

            # Create an OA workflow
            self._processing_wf = OAWorkflow(p_name='Success Assessment Wf', p_class_shared=self._shared)


        # 2. Create a reward task
        if self._success_task is None:
            # Create a pseudo reward task
            self._success_task = OATask(p_name='Compute Success',
                p_visualize=self._visualize,
                p_range_max=self.get_range(),
                p_duplicate_data=True)

            # Assign the task method to custom implementation
            self._success_task._run = self._run

            # Add the task to workflow
            self._processing_wf.add_task(self._success_task)


        # 4. Creating task level attributes for states
        try:
            self._success_task._state
        except AttributeError:
            self._success_task._state = p_state.copy()



        # 5. creating new instance with new state
        self._instance = Instance(p_state)


        # 6. Run the workflow
        self._processing_wf.run(p_inst_new=[self._instance])


        # 7. Return the results
        return self._processing_wf.get_so().get_results()[self._success_task.get_tid()]


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
    def _adapt(self, **p_kwargs) -> bool:
        """

        Parameters
        ----------
        p_kwargs

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def _adapt_on_event(self, p_event_id:str, p_event_object:Event) -> bool:
        """

        Parameters
        ----------
        p_event_id
        p_event_object

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def _add_task(self, p_task:StreamTask):
        """

        Parameters
        ----------
        p_task

        Returns
        -------

        """
        pass


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
        self._state.set_values(p_inst_new.get_feature_data().get_values())

        return self._compute_success(p_state=self._state)






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OAFCtBroken(FctBroken):
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
              p_processing_wf:StreamWorkflow = None,
              p_logging=Log.C_LOG_ALL,
              **p_kwargs):

        AFctBroken.__init__(self,
              p_name=p_name,
              p_range_max=p_range_max,
              p_class_shared=p_class_shared,
              p_visualize=p_visualize,
              p_logging=p_logging,
              **p_kwargs)

        Model.__init__()

        self._processing_wf = p_processing_wf
        self._broken_task:StreamTask = None
        self._shared = p_class_shared
        self._instance:Instance = None
        self._state:State = None


## -------------------------------------------------------------------------------------------------
    def compute_broken(self, p_state: State) -> bool:
        """

        Parameters
        ----------
        p_state

        Returns
        -------

        """
        self.log(Log.C_LOG_TYPE_I, 'Assessing Broken...')


        # 1. check if the user has already created a workflow and added to tasks
        if self._processing_wf is None:
            # Create a shared object if not provided
            if self._shared is None:
                self._shared = StreamShared()


            # Create an OA workflow
            self._processing_wf = OAWorkflow(p_name='Broken Assessment Wf', p_class_shared=self._shared)


        # 2. Create a reward task
        if self._broken_task is None:
            # Create a pseudo reward task
            self._broken_task = OATask(p_name='Compute Broken',
                p_visualize=self._visualize,
                p_range_max=self.get_range(),
                p_duplicate_data=True)

            # Assign the task method to custom implementation
            self._broken_task._run = self._run

            # Add the task to workflow
            self._processing_wf.add_task(self._broken_task)


        # 4. Creating task level attributes for states
        try:
            self._broken_task._state
        except AttributeError:
            self._broken_task._state = p_state.copy()


        # 5. creating new instance with new state
        self._instance = Instance(p_state)


        # 6. Run the workflow
        self._processing_wf.run(p_inst_new=[self._instance])


        # 7. Return the results
        return self._processing_wf.get_so().get_results()[self._broken_task.get_tid()]


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
    def _adapt(self, **p_kwargs) -> bool:
        """

        Parameters
        ----------
        p_kwargs

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def _adapt_on_event(self, p_event_id:str, p_event_object:Event) -> bool:
        """

        Parameters
        ----------
        p_event_id
        p_event_object

        Returns
        -------

        """


## -------------------------------------------------------------------------------------------------
    def add_task(self, p_task: StreamTask):
        """

        Parameters
        ----------
        p_task

        Returns
        -------

        """
        pass


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
        self._state.set_values(p_inst_new.get_feature_data().get_values())

        return self._compute_broken(p_state=self._state)






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class OASystem(ASystem):
    """

    Parameters
    ----------
    p_mode
    p_latency
    p_fct_strans
    p_fct_success
    p_fct_broken
    p_processing_wf
    p_visualize
    p_logging
    """


    ## -------------------------------------------------------------------------------------------------
    def __init__(self,
               p_mode = Mode.C_MODE_SIM,
               p_latency = None,
               p_fct_strans : FctSTrans = None,
               p_fct_success : FctSuccess = None,
               p_fct_broken : FctBroken = None,
               p_processing_wf : StreamWorkflow = None,
               p_visualize : bool = False,
               p_logging = Log.C_LOG_ALL):

        ASystem.__init__(self,
               p_mode = p_mode,
               p_latency = p_latency,
               p_fct_strans = p_fct_strans,
               p_fct_success = p_fct_success,
               p_fct_broken = p_fct_broken,
               p_visualize = p_visualize,
               p_logging = p_logging)

        self._processing_wf = p_processing_wf



## -------------------------------------------------------------------------------------------------
    @staticmethod
    def setup_spaces():

        return None, None


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


## -------------------------------------------------------------------------------------------------
    def _adapt(self, **p_kwargs) -> bool:
        """

        Parameters
        ----------
        p_kwargs

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def switch_adaptivity(self, p_ada:bool):
        """

        Parameters
        ----------
        p_ada

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def _adapt_on_event(self, p_event_id:str, p_event_object:Event) -> bool:
        """

        Parameters
        ----------
        p_event_id
        p_event_object

        Returns
        -------

        """
        pass


## -------------------------------------------------------------------------------------------------
    def _run_processing_wf(self):
        """

        Returns
        -------

        """
        pass