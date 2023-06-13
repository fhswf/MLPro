## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.sl
## -- Module  : basics.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-13  0.0.0     LSB       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2023-06-13)

This module provides training classes for supervised learning tasks.
"""


from mlpro.bf.ml import *
from mlpro.sl.basics import *




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SLScenario (Scenario):
    """
    To be designed.
    """

    C_TYPE = 'SL-Scenario'


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_mode=Mode.C_MODE_SIM,
                 p_ada: bool = True,
                 p_cycle_limit: int = 0,
                 p_visualize: bool = True,
                 p_logging=Log.C_LOG_ALL):

        # If dataset config given, setup dataset
        # Assign Datasets
        self._dataset = None
        # save the lengths of all the datasets
        self._model : SLAdaptiveFunction = None
        self._data_loggers : dict = {}
        Scenario.__init__(self,
                          p_mode = p_mode,
                          p_ada = p_ada,
                          p_cycle_limit = p_cycle_limit,
                          p_visualize = p_visualize,
                          p_logging = p_logging)

        self.connect_dataloggers()
        # raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _run_cycle(self):

        # Check if the first run
        success = False
        error = False
        adapted = False
        end_of_data = False

        # if self._cycle_id >= ( len(self.get_dataset()) - 1 ):
        #     end_of_data = True
        # else:
        #     end_of_data = False

        # adapted = self._model.adapt(p_dataset = self.get_dataset().get_next())

        # Need to optimize the adapt method of the SLAdaptiveFunction which currently just adapts only when the
        # distance is more than threshold

        # get success from the model

        # Error computations such as Stagnation, etc.

        return success, error, adapted, end_of_data


## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada:bool, p_visualize:bool, p_logging) -> Model:
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed):

        self._model.reset(p_seed)
        if self._visualize:
            self._model.init_plot()


## -------------------------------------------------------------------------------------------------
    def connect_dataloggers(self, p_ds = None, p_name = None):

        if p_ds is not None and p_name is not None:
            self._data_loggers[p_name] = p_ds


## -------------------------------------------------------------------------------------------------
    def get_dataset(self):
        return self._dataset


## -------------------------------------------------------------------------------------------------
    def _init_plot(self):
        pass


## -------------------------------------------------------------------------------------------------
    def _update_plot(self):
        pass




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SLTraining (Training):
    """
    To be designed.
    """

    C_NAME = 'SL'


## -------------------------------------------------------------------------------------------------
    def __init__(self, **p_kwargs):

        Training.__init__(self)

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _run(self):
        pass


## -------------------------------------------------------------------------------------------------
    def _run_cycle(self) -> bool:
        pass


## -------------------------------------------------------------------------------------------------
    def _init_epoch(self):
        pass


## -------------------------------------------------------------------------------------------------
    def setup_dataset(self):
        pass


## -------------------------------------------------------------------------------------------------
    def _update_epoch(self):
        pass


## -------------------------------------------------------------------------------------------------
    def _close_epoch(self):
        pass


## -------------------------------------------------------------------------------------------------
    def _init_eval(self):
        pass


## -------------------------------------------------------------------------------------------------
    def _update_eval(self):
        pass


## -------------------------------------------------------------------------------------------------
    def _close_eval(self):
        pass


## -------------------------------------------------------------------------------------------------
    def get_results(self) -> TrainingResults:
        pass


## -------------------------------------------------------------------------------------------------
    def add_metric(self):
        pass
