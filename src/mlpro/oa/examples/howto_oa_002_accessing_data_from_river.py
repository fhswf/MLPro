## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : howto002_oa_accessing_data_from_river
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-06-14  0.0.0     LSB      Creation
## -- 2022-06-14  1.0.0     LSB      Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2022-06-14)

This module shows how to wrap mlpro's Stream and StreamProvider class to River.
"""
from mlpro.wrappers.river import *


# 1. Create a Wrapper for OpenML stream provider
river_wrap = WrStreamProviderRiver()


# 2. Get a list of streams available at the stream provider
stream_list = river_wrap.get_stream_list(p_display_list=True)


# 3. Get a specific stream from the stream provider
stream = river_wrap.get_stream('Insects')


# 4. get the feature space of the stream
feature_space = stream.get_feature_space()
river_wrap.log(stream.C_LOG_TYPE_I,"Number of features in the stream:",feature_space.get_num_dim(),'\n\n')



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
