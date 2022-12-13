## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.streams.tasks
## -- Module  : rearranger.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-12-12  0.0.0     DA       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2022-12-12)

This module provides a stream task class Rearranger to rearrange the feature and label space of
instances.
"""


from mlpro.bf.exceptions import *
from mlpro.bf.various import Log
from mlpro.bf.mt import Task
from mlpro.bf.math import Set
from mlpro.bf.streams import Instance, StreamTask




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Rearranger (StreamTask):
    """
    ...

    Parameters
    ----------
    p_name : str
        Optional name of the task. Default is None.
    p_range_max : int
        Maximum range of asynchonicity. See class Range. Default is Range.C_RANGE_PROCESS.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_features_new : list
        List of feature ids.
    p_labels_new : list
        List of label ids.
    p_kwargs : dict
        Further optional named parameters.
    """

    C_NAME                  = 'Rearranger'

    C_PLOT_STANDALONE       = False

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_name: str = None, 
                  p_range_max = Task.C_RANGE_THREAD, 
                  p_visualize : bool = False, 
                  p_logging = Log.C_LOG_ALL, 
                  p_features_new : list = None,
                  p_labels_new : list = None,
                  **p_kwargs ):

        super().__init__( p_name = p_name, 
                          p_range_max = p_range_max, 
                          p_visualize = p_visualize, 
                          p_logging = p_logging, 
                          **p_kwargs )

        if ( p_features_new is None or len(p_features_new)==0 ) and ( p_labels_new is None or len(p_labels_new)==0 ):
            raise ParamError('Please provide ids for features and/or labels')
                
        self._features_new = p_features_new
        self._labels_new   = p_labels_new
        self._prepared     = False


## -------------------------------------------------------------------------------------------------
    def _prepare_rearrangement(self, p_inst:Instance):
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _rearrange(self, p_inst:Instance):
        
        # 1 Create new feature and label data structures
        raise NotImplementedError

        

## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst_new: set, p_inst_del: set):

        # 1 Late preparation based on first incoming instance
        if not self._prepared:
            try:
                inst = p_inst_new[0]
            except:
                inst = p_inst_del[0]

            self._prepare_rearrangement(p_inst=inst)
            self._prepared = True


        # 2 Rearrange new instances
        for inst in p_inst_new:
            self._rearrange(p_inst=inst)


        # 3 Rearrange instances to be deleted
        for inst in p_inst_del:
            self._rearrange(p_inst=inst)
