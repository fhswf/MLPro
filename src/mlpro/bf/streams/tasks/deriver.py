## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.bf.streams.tasks
## -- Module  : deriver.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-02-02  0.0.0     SY       Creation
## -- 2023-02-05  1.0.0     SY       First version release
## -- 2024-05-10  1.0.1     DA/SY    Bugfix in Deriver.__init__()
## -- 2024-05-22  1.1.0     DA       Refactoring
## -- 2024-07-17  1.1.1     SY       Method Deriver._prepare_derivation(): takeover of feature 
## --                                and label space from first instance
## -- 2025-04-25  1.1.2     DA       Bugfix in DerivativeFunction._custom_function()
## -- 2025-06-06  1.2.0     DA       Refactoring: p_inst -> p_instances
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2025-06-06)

This module provides a stream task class Deriver to derive the data of instances.
"""


from mlpro.bf.exceptions import *
from mlpro.bf.various import Log
from mlpro.bf.mt import Task
from mlpro.bf.math import Element
from mlpro.bf.streams import Instance, InstDict, StreamTask, Feature, Label
from mlpro.bf.physics import TransferFunction
import numpy as np



# Export list for public API
__all__ = [ 'Deriver',
            'DerivativeFunction' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Deriver(StreamTask):
    """
    This stream task extend the feature and label data of incoming instances with a derivation of
    a pre-selected feature.

    Parameters
    ----------
    p_name : str
        Optional name of the task. Default is None.
    p_range_max : int
        Maximum range of asynchonicity. See class Range. Default is Range.C_RANGE_PROCESS.
    p_duplicate_data : bool
        If True, instances will be duplicated before processing. Default = False.
    p_visualize : bool
        Boolean switch for visualisation. Default = False.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL.
    p_derived_feature : Feature
        A pre-selected feature that would like to be derived. Default = None.
    p_derived_label : Feature
        A correspondig label of the derived feature. Default = None.
    p_order_derivative : int
        The derivative of order n. Default = None.
    p_kwargs : dict
        Further optional named parameters.
    """

    C_NAME              = 'Deriver'
    C_PLOT_STANDALONE   = True

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_name:str = None, 
                  p_range_max = Task.C_RANGE_THREAD, 
                  p_duplicate_data:bool = False,
                  p_visualize:bool = False, 
                  p_logging = Log.C_LOG_ALL,
                  p_derived_feature:Feature = None,
                  p_derived_label:Label = None,
                  p_order_derivative:int = 1,
                  **p_kwargs ):

        super().__init__( p_name = p_name, 
                          p_range_max = p_range_max,
                          p_duplicate_data = p_duplicate_data, 
                          p_visualize = p_visualize, 
                          p_logging = p_logging, 
                          **p_kwargs )

        if p_order_derivative < 1:
            raise ParamError('p_order_derivative can not be lower than 1')

        if not isinstance(p_derived_feature, Feature):
            raise ParamError('Please provide a feature to be derived')
        
        self._order_derivative  = p_order_derivative
        self._tic               = 0
        self._mem_values        = []
        self._mem_time_stamp    = []

        self._derived_feature   = p_derived_feature
        self._derived_label     = p_derived_label
        self._feature_space     = None
        self._label_space       = None
        self._prepared          = False
        
        self._derivative_func = DerivativeFunction(p_name='derivative_func',
                                                   p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                                   p_logging=p_logging,
                                                   p_dt=0,
                                                   order=self._order_derivative)


## -------------------------------------------------------------------------------------------------
    def _prepare_derivation(self, p_instance:Instance):

        # 1 Feature space
        features                = p_instance.get_feature_data().get_dim_ids()
        self._feature_space     = type(p_instance.get_feature_data().get_related_set())()
        self._idx_feature       = features.index(self._derived_feature.get_id())
        
        for feature in p_instance.get_feature_data().get_related_set().get_dims():
            self._feature_space.add_dim(p_dim=feature)

        feature                 = self._derived_feature.copy()
        feature._name_short     = feature._name_short + ' OD-' + str(self._order_derivative)
        feature._name_long      = feature._name_long + ' OD-' + str(self._order_derivative)
        self._feature_space.add_dim(p_dim=feature)
        
        # 2 Label space
        try:
            labels              = p_instance.get_label_data().get_dim_ids()
            self._label_space   = type(p_instance.get_label_data().get_related_set())()
            self._idx_label     = labels.index(self._derived_label.get_id())
            
            for label in p_instance.get_label_data().get_related_set().get_dims():
                self._label_space.add_dim(p_dim=label)

            label               = self._derived_label.copy()
            label._name_short   = label._name_short + ' OD-' + str(self._order_derivative)
            label._name_long    = label._name_long + ' OD-' + str(self._order_derivative)
            self._label_space.add_dim(p_dim=label)
        except:
            labels              = []


## -------------------------------------------------------------------------------------------------
    def _derive_data(self, p_instance:Instance):
        
        # 1 Preparation
        f_values_old = p_instance.get_feature_data().get_values()
        f_data_new = Element(p_set=self._feature_space)

        try:
            l_values_old = p_instance.get_label_data().get_values()
            l_data_new = Element(p_set=self._label_space)
            l_values_new = np.append(l_values_old.copy(), l_values_old.copy()[self._idx_feature])
            l_data_new.set_values(l_values_new)
        except:
            l_data_new = None
        
        if p_instance.tstamp is not None:
            t_values_old = p_instance.tstamp
        else:
            t_values_old = None

        # 2 Calculate extended feature value
        if self._tic < self._order_derivative:
            f_values_new = np.append(f_values_old, 0)
            self._tic += 1
        else:
            for od in range(self._order_derivative):
                if od == 0:
                    f_values = self._mem_values.copy()
                    t_stamps = self._mem_time_stamp.copy()
                    f_values.append(f_values_old.copy()[self._idx_feature])
                    t_stamps.append(t_values_old)
                f_values = self._derivative_func(
                    p_input=f_values,
                    p_range=t_stamps
                    )
            f_values_new = np.append(f_values_old, f_values[-1])
            self._mem_values.pop(0)
            self._mem_time_stamp.pop(0)
        
        self._mem_values.append(f_values_old.copy()[self._idx_feature])
        self._mem_time_stamp.append(t_values_old)
            

        # 3 Add feature and label data in origin instance
        f_data_new.set_values(f_values_new)
        p_instance.set_feature_data(p_feature_data=f_data_new)

        if l_data_new is not None:
            p_instance.set_label_data(p_label_data=l_data_new)
        

## -------------------------------------------------------------------------------------------------
    def _run(self, p_instances:InstDict):

        if not self._prepared:
            try:
                (inst_type, inst) = next(iter(p_instances.values()))
                self._prepare_derivation(p_instance=inst)
                self._prepared = True
            except:
                return

        for (inst_type,inst) in sorted(p_instances.values()):
            self._derive_data(p_instance=inst)



                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DerivativeFunction(TransferFunction):
  
    
## -------------------------------------------------------------------------------------------------      
    def _set_function_parameters(self, p_args) -> bool:
        if self.get_type() == self.C_TRF_FUNC_CUSTOM:
            return True
  
    
## -------------------------------------------------------------------------------------------------      
    def _custom_function(self, p_input, p_range):
        output = []
        for x in range(len(p_input)):
            if x == 0:
                output.append(0)
            else:
                try:
                    delta_t = (p_range[x]-p_range[x-1]).total_seconds() #seconds
                except:
                    delta_t = 1

                output.append((p_input[x]-p_input[x-1])/delta_t)

        return output