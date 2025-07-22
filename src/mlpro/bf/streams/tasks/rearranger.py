## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.streams.tasks
## -- Module  : rearranger.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-12-12  0.0.0     DA       Creation
## -- 2022-12-13  1.0.0     DA       First implementation
## -- 2022-12-14  1.0.1     DA       Corrections
## -- 2022-12-16  1.0.2     DA       Little refactoring
## -- 2022-12-19  1.0.3     DA       New parameter p_duplicate_data
## -- 2024-05-22  1.1.0     DA       Refactoring
## -- 2024-06-17  1.1.1     DA       Method Rearranger._prepare_rearrangement(): takeover of feature 
## --                                and label space from first instance
## -- 2025-06-06  1.2.0     DA       Refactoring: p_inst -> p_instances
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2025-06-06)

This module provides a stream task class Rearranger to rearrange the feature and label space of
instances.

"""


from mlpro.bf.exceptions import *
from mlpro.bf.various import Log
from mlpro.bf.mt import Task
from mlpro.bf.math import Element
from mlpro.bf.streams import Instance, InstDict, StreamTask



# Export list for public API
__all__ = [ 'Rearranger' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Rearranger (StreamTask):
    """
    This stream task rearrange the feature and/or label data of incoming instances. To this regard,
    two additional parameters p_features_new and p_labels_new describe the dimensions of the 
    feature/label space of the resulting instances.

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
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_features_new : list
        List of resulting features that are described as tuples ( 'F' or 'L', list[Dimension] ). The
        first component specifies the origin ('F' = feature space, 'L' = label space). The second
        component is a list of dimension objects.
    p_labels_new : list[Label]
         List of resulting labels that are described as tuples ( 'F' or 'L', list[Dimension] ).
    p_kwargs : dict
        Further optional named parameters.
    """

    C_NAME                  = 'Rearranger'
    C_PLOT_STANDALONE       = True

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_name: str = None, 
                  p_range_max = Task.C_RANGE_THREAD, 
                  p_duplicate_data : bool = False,
                  p_visualize : bool = False, 
                  p_logging = Log.C_LOG_ALL, 
                  p_features_new : list = [],
                  p_labels_new : list = [],
                  **p_kwargs ):

        super().__init__( p_name = p_name, 
                          p_range_max = p_range_max,
                          p_duplicate_data = p_duplicate_data, 
                          p_visualize = p_visualize, 
                          p_logging = p_logging, 
                          **p_kwargs )

        if ( len(p_features_new)==0 ) and ( len(p_labels_new)==0 ):
            raise ParamError('Please provide ids for features and/or labels')
                
        self._features_new  = p_features_new
        self._labels_new    = p_labels_new
        self._feature_space = None
        self._label_space   = None
        self._prepared      = False


## -------------------------------------------------------------------------------------------------
    def _prepare_rearrangement(self, p_instance : Instance):

        # 1 Preparation
        self._mapping_f2f = []
        self._mapping_l2f = []
        self._mapping_f2l = []
        self._mapping_l2l = []

        # 1.1 Feature space
        features            = p_instance.get_feature_data().get_dim_ids()
        self._feature_space = type(p_instance.get_feature_data().get_related_set())()

        for f_entry in self._features_new: 
            for feature in f_entry[1]:
                self._feature_space.add_dim(p_dim=feature)

        # 1.2 Label space
        try:
            labels            = p_instance.get_label_data().get_dim_ids()
            self._label_space = type(p_instance.get_label_data().get_related_set())()

            for l_entry in self._labels_new: 
                for label in l_entry[1]:
                    self._label_space.add_dim(p_dim=label)
        except:
            labels = []


        # 2 Set up mappings for feature space rearrangement
        i_new    = 0

        for f_entry in self._features_new:

            if f_entry[0] == 'F':
                for feature in f_entry[1]:
                    self._mapping_f2f.append( ( i_new, features.index(feature.get_id())) )
                    i_new += 1
            
            else:
                for label in f_entry[1]:
                    self._mapping_l2f.append( ( i_new, labels.index(label.get_id())) )
                    i_new += 1


        # 3 Set up mappings for label space rearrangement
        i_new    = 0

        for l_entry in self._labels_new:

            if l_entry[0] == 'F':
                for feature in l_entry[1]:
                    self._mapping_f2l.append( ( i_new, features.index(feature.get_id())) )
                    i_new += 1
            
            else:
                for label in l_entry[1]:
                    self._mapping_l2l.append( ( i_new, labels.index(label.get_id())) )
                    i_new += 1


## -------------------------------------------------------------------------------------------------
    def _rearrange(self, p_inst:Instance):
        
        # 1 Preparation
        f_values_old = p_inst.get_feature_data().get_values()
        f_data_new   = Element(p_set=self._feature_space)
        f_values_new = f_data_new.get_values()

        try:
            l_values_old = p_inst.get_label_data().get_values()
            l_data_new   = Element(p_set=self._label_space)
            l_values_new = l_data_new.get_values()
        except:
            l_values_old = []
            l_data_new   = None
            l_values_new = []
  

        # 2 Collect new feature values
        for ids in self._mapping_f2f:
            f_values_new[ids[0]] = f_values_old[ids[1]]

        for ids in self._mapping_l2f:
            f_values_new[ids[0]] = l_values_old[ids[1]]


        # 3 Collect new label values
        for ids in self._mapping_f2l:
            l_values_new[ids[0]] = f_values_old[ids[1]]

        for ids in self._mapping_l2l:
            l_values_new[ids[0]] = l_values_old[ids[1]]


        # 4 Replace feature and label data in origin instance
        p_inst.set_feature_data(p_feature_data=f_data_new)

        if l_data_new is not None:
            p_inst.set_label_data(p_label_data=l_data_new)
        

## -------------------------------------------------------------------------------------------------
    def _run(self, p_instances : InstDict):

        # 1 Late preparation based on first incoming instance
        if not self._prepared:
            try:
                (inst_type, inst) = next(iter(p_instances.values()))
                self._prepare_rearrangement(p_instance=inst)
                self._prepared = True
            except:
                return

        # 2 Rearrange new instances (order doesn't matter)
        for (inst_type,inst) in p_instances.values(): 
            self._rearrange(p_inst=inst)