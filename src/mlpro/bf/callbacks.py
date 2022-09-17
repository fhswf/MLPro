## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf
## -- Module  : callbacks
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-09-02  0.0.0     MRD       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2022-09-02)

This module provides classes for callback function.
"""

from mlpro.bf.various import Log



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Callback(Log):
    """
    Base class for callback.
    """

    C_TYPE      = 'Callback'

    def __init__(self, p_logging=Log.C_LOG_ALL):
        super().__init__(p_logging=p_logging)

        self.training = None
        self.scenario = None
        self.folder_path = None
        self.locals = {}

## -------------------------------------------------------------------------------------------------
    def init_callback(self, p_training, p_path):
        """
        Initialization callback. Save reference for Training Class and Scenario Class
        """
        self.training = p_training
        self.scenario = p_training.get_scenario()
        self.folder_path = p_path
        self._init_callback()


## -------------------------------------------------------------------------------------------------
    def _init_callback(self):
        """
        For customize initialization.
        """
        pass

## -------------------------------------------------------------------------------------------------
    def _training_start(self):
        """
        For customize function. This will be called when the training starts.
        """
        pass

## -------------------------------------------------------------------------------------------------
    def training_start(self):
        """
        This function will be called when the training starts.
        """
        self._training_start()

## -------------------------------------------------------------------------------------------------
    def _training_end(self):
        """
        For customize function. This will be called when the training ends.
        """
        pass

## -------------------------------------------------------------------------------------------------
    def training_end(self):
        """
        This function will be called when the training ends.
        """
        self._training_end()
