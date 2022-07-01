## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : howto_oa_001_accessing_data_from_openml.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-06-09  0.0.0     LSB      Creation
## -- 2022-06-09  1.0.0     LSB      Release of first version
## -- 2022-06-13  1.0.1     LSB      Bug Fix
## -- 2022-06-18  1.0.2     LSB      Restructured logging output
## -- 2022-06-25  1.0.3     LSB      Refactoring for new instance and Label class
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.3 (2022-06-25)

This module shows how to wrap MLPro's Stream and StreamProvider class to OpenML, including how to fetch the list of
streams and downloading a stream from the list of streams available with the stream provider, getting the feature
spaces of the particular stream. This module also illustrates how to reset the stream and fetch the stream instances
as needed.
Please run the following code to understand the wrapper functionality and to produce similar results.
"""


from mlpro.wrappers.openml import WrStreamProviderOpenML
from mlpro.bf.various import Log


# checking for internal unit test
if not __name__ == '__main__':
    p_logging = Log.C_LOG_NOTHING
else:
    p_logging = Log.C_LOG_ALL


# 1. Create a Wrapper for OpenML stream provider
open_ml = WrStreamProviderOpenML(p_logging = p_logging)


# 2. Get a list of streams available at the stream provider
stream_list = open_ml.get_stream_list(p_logging = p_logging)


# 3. Get a specific stream from the stream provider
stream = open_ml.get_stream(75)


# 4. get the feature space of the stream
feature_space = stream.get_feature_space()
open_ml.log(stream.C_LOG_TYPE_I,"Number of features in the stream:",feature_space.get_num_dim(),'\n\n')


# 5. resetting the stream
stream.reset()


# 6. Loading stream instances
stream.log(stream.C_LOG_TYPE_W,'Fetching the stream instances')
for i in range(10):
    curr_instance = stream.get_next()
    curr_features = curr_instance.get_feature_data().get_values()
    curr_label = curr_instance.get_label_data().get_values()
    stream.log(stream.C_LOG_TYPE_I, '\n\nCurrent Instance:', curr_features, '\nLabel:', curr_label, '\n')



# 7. resetting the stream
stream.reset()


# 8. Getting stream instances
stream.log(stream.C_LOG_TYPE_W,'Fetching the stream instances')
for i in range(5):
    curr_instance = stream.get_next()
    curr_features = curr_instance.get_feature_data().get_values()
    curr_label = curr_instance.get_label_data().get_values()
    stream.log(stream.C_LOG_TYPE_I, '\n\nCurrent Instance:' , curr_features, '\nLabel:', curr_label, '\n')
