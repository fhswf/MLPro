## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.streams.streams
## -- Module  : csv_file.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-03-02  0.0.0     SY       Creation 
## -- 2023-03-06  1.0.0     SY       First release
## -- 2023-04-10  1.0.1     SY       Refactoring
## -- 2023-04-14  1.1.0     SY       Make StreamMLProCSV independent from StreamMLProBase
## -- 2023-04-16  1.1.1     SY       Refactoring
## -- 2023-04-17  1.1.2     DA       Method StreamMLProCSV.set_options(): changed exception type 
## --                                to ParamError
## -- 2024-06-04  1.1.3     DA       Bugfix: ESpace instead of MSpace
## -- 2024-07-04  1.1.4     SY       Allowing string in the datasets 
## -- 2024-07-19  1.1.5     SY       Allowing string in the datasets 
## -- 2025-05-22  1.1.6     DA       Explicit import of depending classes
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.6 (2025-05-22)

This module provides the native stream class StreamMLProCSV.
This stream provides a functionality to convert csv file to a MLPro compatible stream data.
"""


import math
import numpy as np

from mlpro.bf.various import Log, ScientificObject
from mlpro.bf.exceptions import *
from mlpro.bf.data import DataStoring
from mlpro.bf.math import Element, MSpace, ESpace
from mlpro.bf.streams import *



# Export list for public API
__all__ = [ 'StreamMLProCSV' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamMLProCSV(Stream):
    """
    Reusable class for converting data from csv files to MLPro's data streams.

    Parameters
    ----------
    p_id
        Optional id of the stream. Default = None.
    p_name : str
        Optional name of the stream. Default = ''.
    p_num_instances : int
        Optional number of instances in the stream. Default = 0.
    p_version : str
        Optional version of the stream. Default = ''.
    p_feature_space : MSpace
        Optional feature space. Default = None.
    p_label_space : MSpace
        Optional label space. Default = None.
    p_sampler
        Optional sampler. Default: None.
    p_mode
        Operation mode. Default: Mode.C_MODE_SIM.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL.
    p_path_load
        Path of the loaded file.
    p_csv_filename
        File name of the loaded CSV file. 
    p_delimiter
        Delimiter of the CSV data. Default: "\t"
    p_frame
        Availability of framed in the loaded CSV file. Default: True
    p_header
        Availability of header in the first row of the loaded CSV file. Default: True
    p_list_features
        List of the file's headers that is loaded as features in the stream. Default: None
    p_list_labels
        List of the file's headers that is loaded as labels in the stream. Default: None

    """

    C_ID        = 'CSV2MLPro'
    C_TYPE      = 'Stream CSV File'
    C_NAME      = ''
    C_VERSION   = '1.0.0'

    C_SCIREF_TYPE   = ScientificObject.C_SCIREF_TYPE_ONLINE
    C_SCIREF_AUTHOR = 'MLPro'
    C_SCIREF_URL    = 'https://mlpro.readthedocs.io'


## -------------------------------------------------------------------------------------------------
    def set_options(self, **p_kwargs):
        """
        Method to set specific options for the stream. The possible options depend on the 
        stream provider and stream itself.
        """

        self._kwargs = p_kwargs.copy()
        self._loaded = False

        if 'p_path_load' not in self._kwargs:
            raise ParamError('p_path_load is not defined!')
            
        if 'p_csv_filename' not in self._kwargs:
            raise ParamError('p_csv_filename is not defined!')
        else:
            self.C_NAME = self._kwargs['p_csv_filename']
            
        if 'p_delimiter' not in self._kwargs:
            self._kwargs['p_delimiter'] = "\t"
            
        if 'p_frame' not in self._kwargs:
            self._kwargs['p_frame'] = True
            
        if 'p_header' not in self._kwargs:
            self._kwargs['p_header'] = True
            
        if 'p_list_features' not in self._kwargs:
            self._list_features = None
        else:
            self._list_features = self._kwargs['p_list_features']
            
        if 'p_list_labels' not in self._kwargs:
            self._list_labels = None
        else:
            self._list_labels = self._kwargs['p_list_labels']
    

## -------------------------------------------------------------------------------------------------
    def _setup_feature_space(self, p_data_type_numeric=True) -> MSpace:
        
        feature_space : MSpace = ESpace()
        
        if self._list_features is not None:
            for num, ftrs in enumerate(self._list_features):
                if ftrs in self._from_csv.names:
                    try:
                        if p_data_type_numeric[num]:
                            base_set = Feature.C_BASE_SET_R
                        else:
                            base_set = Feature.C_BASE_SET_DO
                    except:
                        if p_data_type_numeric:
                            base_set = Feature.C_BASE_SET_R
                        else:
                            base_set = Feature.C_BASE_SET_DO
                    feature_space.add_dim(Feature(p_name_short = ftrs,
                                                  p_base_set = base_set,
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
                                              p_base_set = Feature.C_BASE_SET_DO,
                                              p_name_long = ftrs,
                                              p_name_latex = '',
                                              p_description = '',
                                              p_symmetrical = False,
                                              p_logging=Log.C_LOG_NOTHING)
                                        )
        return label_space


## -------------------------------------------------------------------------------------------------
    def _init_dataset(self):

        if self._loaded == False:
            p_variable      = []
            self._from_csv  = DataStoring(p_variable)

            if not self._from_csv.load_data( p_path = self._kwargs['p_path_load'],
                                             p_filename = self._kwargs['p_csv_filename'],
                                             p_delimiter = self._kwargs['p_delimiter'],
                                             p_frame = self._kwargs['p_frame'],
                                             p_header = self._kwargs['p_header'] ):
                raise Error('CSV import from file "' + self._kwargs['p_csv_filename'] + '" failed!')
            
            try:
                d_type_numeric  = []
                for num, el in enumerate(self._list_features):
                    idx             = list(self._from_csv.memory_dict.keys()).index(el)
                    extended_data   = []
                    key_0           = list(self._from_csv.memory_dict.keys())[idx]
                    for fr in self._from_csv.memory_dict[key_0]:
                        extended_data.extend(self._from_csv.memory_dict[key_0][fr])
                    try:
                        math.prod(extended_data)
                        d_type_numeric.append(True)
                    except:
                        d_type_numeric.append(False)
                    if num == 0:
                        self.C_NUM_INSTANCES = self._num_instances = len(extended_data)
            except:
                d_type_numeric  = False
                self.C_NUM_INSTANCES = self._num_instances = 0
            
            if self._sampler is not None:
                self._sampler.set_num_instances(self._num_instances)

            self._feature_space = self._setup_feature_space(d_type_numeric)
            self._label_space   = self._setup_label_space()
            
            dim             = self._feature_space.get_num_dim()
            dim_l           = self._label_space.get_num_dim()
            self._dataset   = np.empty((self.C_NUM_INSTANCES,dim), dtype=object)
            self._dataset_l = np.empty((self.C_NUM_INSTANCES,dim_l), dtype=object)
            extended_data   = {}
            ids             = self._feature_space.get_dim_ids()
            
            x = 0
            for id_ in ids:
                ft_name = self._feature_space.get_dim(id_).get_name_short()
                extended_data[ft_name] = []
                for fr in self._from_csv.memory_dict[ft_name]:
                    extended_data[ft_name].extend(self._from_csv.memory_dict[ft_name][fr])
                self._dataset[:,x] = np.array(extended_data[ft_name])
                x += 1
            
            x = 0        
            ids = self._label_space.get_dim_ids()    
            for id_ in ids:
                lbl_name = self._label_space.get_dim(id_).get_name_short()
                extended_data[lbl_name] = []
                for fr in self._from_csv.memory_dict[lbl_name]:
                    extended_data[lbl_name].extend(self._from_csv.memory_dict[lbl_name][fr])
                self._dataset_l[:,x] = np.array(extended_data[lbl_name])
                x += 1
            
            self._loaded = True


## -------------------------------------------------------------------------------------------------
    def _reset(self):

        self._index = 0
        self._init_dataset()


## -------------------------------------------------------------------------------------------------
    def _get_next(self) -> Instance:

        if self._index == self.C_NUM_INSTANCES: raise StopIteration

        feature_data = Element(self._feature_space)
        feature_data.set_values(p_values=self._dataset[self._index])

        self._index += 1

        return Instance( p_feature_data=feature_data )