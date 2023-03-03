## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.streams.streams
## -- Module  : csv_file.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-03-02  0.0.0     SY       Creation 
## -- 2023-03-03  1.0.0     SY       First release
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-03-03)

This module provides the native stream class  StreamMLProCSV.
This stream provides a functionality to convert csv file to a MLPro compatible stream data.
"""


import numpy as np
from mlpro.bf.data import *
from mlpro.bf.streams.models import *
from mlpro.bf.streams.streams.provider_mlpro import StreamMLProBase





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamMLProCSV(StreamMLProBase):

    C_ID        = 'CSV2MLPro'
    C_NAME      = 'CSV Format to MLPro Stream'
    C_VERSION   = '1.0.0'

    C_SCIREF_TYPE   = ScientificObject.C_SCIREF_TYPE_ONLINE
    C_SCIREF_AUTHOR = 'MLPro'
    C_SCIREF_URL    = 'https://mlpro.readthedocs.io'


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_logging=Log.C_LOG_ALL, **p_kwargs):
        
        if 'p_path_load' not in p_kwargs:
            p_kwargs['p_path_load'] = None
            
        if 'p_csv_filename' not in p_kwargs:
            p_kwargs['p_csv_filename'] = None
            
        if 'p_delimiter' not in p_kwargs:
            p_kwargs['p_delimiter'] = "\t"
            
        if 'p_frame' not in p_kwargs:
            p_kwargs['p_frame'] = True
            
        if 'p_header' not in p_kwargs:
            p_kwargs['p_header'] = True
            
        if 'p_list_features' not in p_kwargs:
            self._list_features = None
        else:
            self._list_features = p_kwargs['p_list_features']
            
        if 'p_list_labels' not in p_kwargs:
            self._list_labels = None
        else:
            self._list_labels = p_kwargs['p_list_labels']
        
        p_variable          = []
        self._from_csv      = DataStoring(p_variable)
        self._from_csv.load_data(p_kwargs['p_path_load'],
                                 p_kwargs['p_csv_filename'],
                                 p_kwargs['p_delimiter'],
                                 p_kwargs['p_frame'],
                                 p_kwargs['p_header'])

        super().__init__(p_logging = p_logging)
    

## -------------------------------------------------------------------------------------------------
    def _setup_feature_space(self) -> MSpace:
        
        feature_space : MSpace = MSpace()
        
        if self._list_features is not None:
            for ftrs in self._list_features:
                if ftrs in self._from_csv.names:
                    feature_space.add_dim(Feature(p_name_short = ftrs,
                                                  p_base_set = Feature.C_BASE_SET_R,
                                                  p_name_long = ftrs,
                                                  p_name_latex = '',
                                                  p_description = '',
                                                  p_symmetrical = False,
                                                  p_logging=Log.C_LOG_NOTHING)
                                          )            
        return feature_space


## -------------------------------------------------------------------------------------------------
    def _setup_label_space(self) -> MSpace:
        
        label_space : MSpace = MSpace()
        
        if self._list_labels is not None:
            for ftrs in self._list_labels:
                if ftrs in self._from_csv.names:
                    label_space.add_dim(Label(p_name_short = ftrs,
                                              p_base_set = Feature.C_BASE_SET_R,
                                              p_name_long = ftrs,
                                              p_name_latex = '',
                                              p_description = '',
                                              p_symmetrical = False,
                                              p_logging=Log.C_LOG_NOTHING)
                                        )
        return label_space


## -------------------------------------------------------------------------------------------------
    def _init_dataset(self):
        
        dim             = self._feature_space.get_num_dim()
        dim_l           = self._label_space.get_num_dim()
        self._dataset   = np.zeros((0,dim))
        self._dataset_l = np.zeros((0,dim_l))
        extended_data   = {}
        ids             = self._feature_space.get_dim_ids()
        
        x = 0
        for id_ in ids:
            ft_name = self._feature_space.get_dim(id_).get_name_short()
            extended_data[ft_name] = []
            for fr in self._from_csv.memory_dict[ft_name]:
                extended_data[ft_name].extend(self._from_csv.memory_dict[ft_name][fr])
            np.append(self._dataset[:,x], extended_data[ft_name])
            x += 1
        
        x = 0        
        ids = self._label_space.get_dim_ids()    
        for id_ in ids:
            lbl_name = self._label_space.get_dim(id_).get_name_short()
            extended_data[lbl_name] = []
            for fr in self._from_csv.memory_dict[lbl_name]:
                extended_data[lbl_name].extend(self._from_csv.memory_dict[lbl_name][fr])
            np.append(self._dataset[:,x], extended_data[lbl_name])
            x += 1