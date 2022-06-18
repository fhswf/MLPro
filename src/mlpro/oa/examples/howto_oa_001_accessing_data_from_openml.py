## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : howto001_oa_accessing_data_from_openml
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-06-09  0.0.0     LSB      Creation
## -- 2022-06-09  1.0.0     LSB      Release of first version
## -- 2022-06-13  1.0.1     LSB      Bug Fix
## -- 2022-06-18  1.0.2     LSB      Restructured logging output
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.2 (2022-06-18)

This module shows how to wrap mlpro's Stream and StreamProvider class to OpenML .
"""
from mlpro.wrappers.openml import WrStreamProviderOpenML


# 1. Create a Wrapper for OpenML stream provider
open_ml = WrStreamProviderOpenML()


# 2. Get a list of streams available at the stream provider
stream_list = open_ml.get_stream_list(p_display_list=True)


# 3. Get a specific stream from the stream provider
stream = open_ml.get_stream(2)


# 4. get the feature space of the stream
feature_space = stream.get_feature_space()
open_ml.log(stream.C_LOG_TYPE_I,"Number of features in the stream:",feature_space.get_num_dim(),'\n\n')
# print("Number of features in the stream:",stream.get_feature_space().get_num_dim())


# 5. resetting the stream
stream.reset()

stream.log(stream.C_LOG_TYPE_W,'Fetching the stream instances')
# 6. Loading stream instances
for i in range(10):
    curr_instance = stream.get_next().get_values()
    stream.log(stream.C_LOG_TYPE_I,'\n\nCurrent Instance:' , curr_instance)


# 7. resetting the stream
stream.reset()

stream.log(stream.C_LOG_TYPE_W,'Fetching the stream instances')
# 8. Getting stream instances
for i in range(5):
    curr_instance = stream.get_next().get_values()
    stream.log(stream.C_LOG_TYPE_I, '\n\nCurrent Instance:' , curr_instance)
