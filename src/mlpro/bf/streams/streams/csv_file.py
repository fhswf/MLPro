## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.streams.streams
## -- Module  : csv_file.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-03-02  0.0.0     SY       Creation 
## -- 2023-03-??  1.0.0     SY       First implementation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-03-??)

This module provides the native stream class  StreamMLProCSV.
This stream provides a functionality to convert csv file to a MLPro compatible stream data.
"""

import numpy as np
from mlpro.bf.streams.models import *
from mlpro.bf.streams.streams.provider_mlpro import StreamMLProBase



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StreamMLProCSV(StreamMLProBase):

    C_ID                = 'CSV2MLPro'
    C_NAME              = 'CSV Format to MLPro Stream'
    C_VERSION           = '1.0.0'

## -------------------------------------------------------------------------------------------------
    def __init__( self, p_logging=Log.C_LOG_ALL ):
        
        # 1. load data from csv files , using bf.data.DataStoring?
        
            # DataStoring (Update Header)
            
            # features and labels selection
            
            # dict keys / csv file headers as features or labels names
        
        # 2. update feature space and label space according to the loaded data

        super().__init__( p_id = self.C_ID, 
                          p_name = self.C_NAME, 
                          p_num_instances = self.C_NUM_INSTANCES, 
                          p_version = self.C_VERSION,
                          p_feature_space = self._setup_feature_space(), 
                          p_label_space = self._setup_label_space(), 
                          p_mode=Mode.C_MODE_SIM,
                          p_logging = p_logging )

## -------------------------------------------------------------------------------------------------
    def _setup_feature_space(self) -> MSpace:
        # feature_space : MSpace = MSpace()

        # for i in range(10):
        #     feature_space.add_dim( Feature( p_name_short = 'f' + str(i),
        #                                     p_base_set = Feature.C_BASE_SET_R,
        #                                     p_name_long = 'Feature #' + str(i),
        #                                     p_name_latex = '',
        #                                     #p_boundaries = self.C_BOUNDARIES,
        #                                     p_description = '',
        #                                     p_symmetrical = False,
        #                                     p_logging=Log.C_LOG_NOTHING ) )

        # return feature_space
        pass


## -------------------------------------------------------------------------------------------------
    def _setup_label_space(self) -> MSpace:
        # label_space : MSpace = MSpace()

        # for i in range(2):
        #     label_space.add_dim( Label( p_name_short = 'l' + str(i),
        #                                 p_base_set = Label.C_BASE_SET_R,
        #                                 p_name_long = 'Label #' + str(i),
        #                                 p_name_latex = '',
        #                                 #p_boundaries = self.C_BOUNDARIES,
        #                                 p_description = '',
        #                                 p_symmetrical = False,
        #                                 p_logging=Log.C_LOG_NOTHING ) )

        # return label_space
        pass


## -------------------------------------------------------------------------------------------------
    def _init_dataset(self):
        
        # 1. transform dict to numpy.array
        
        # 2. assign numpy array to self._dataset and self.dataset_l
        
        pass