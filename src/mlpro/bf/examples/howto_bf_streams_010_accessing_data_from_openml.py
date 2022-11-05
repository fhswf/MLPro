## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_streams_010_accessing_data_from_openml.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-06-09  0.0.0     LSB      Creation
## -- 2022-06-09  1.0.0     LSB      Release of first version
## -- 2022-06-13  1.0.1     LSB      Bug Fix
## -- 2022-06-18  1.0.2     LSB      Restructured logging output
## -- 2022-06-25  1.0.3     LSB      Refactoring for new instance and Label class
## -- 2022-10-12  1.0.4     DA       Renaming
## -- 2022-11-05  1.1.0     DA       Refactoring after changes on class Stream
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2022-11-05)

This module shows how to wrap MLPro's Stream and StreamProvider class to OpenML, including how to fetch the list of
streams and downloading a stream from the list of streams available with the stream provider, getting the feature
spaces of the particular stream. This module also illustrates how to reset the stream and fetch the stream instances
as needed.
Please run the following code to understand the wrapper functionality and to produce similar results.
"""


from mlpro.wrappers.openml import WrStreamProviderOpenML
from mlpro.bf.various import Log



# 0 Prepare Demo/Unit test mode
if __name__ == '__main__':
    num_inst    = 10
    logging     = Log.C_LOG_ALL
else:
    num_inst    = 2
    logging     = Log.C_LOG_NOTHING


# 1. Create a Wrapper for OpenML stream provider
open_ml = WrStreamProviderOpenML(p_logging = logging)


# 2. Get a list of streams available at the stream provider
stream_list = open_ml.get_stream_list(p_logging = logging)


# 3. Get stream "BNG(autos,nominal,1000000)" from the stream provider OpenML
mystream = open_ml.get_stream( p_id=75, p_logging=logging)


# 4. Get the feature space of the stream
feature_space = mystream.get_feature_space()
open_ml.log(mystream.C_LOG_TYPE_I,"Number of features in the stream:",feature_space.get_num_dim())


# 5. Set up an iterator for the stream
myiterator = iter(mystream)


# 6. Fetching some stream instances
myiterator.log(mystream.C_LOG_TYPE_W,'Fetching first', str(num_inst), 'stream instances...')
for i in range(num_inst):
    curr_instance   = next(myiterator)
    curr_data       = curr_instance.get_feature_data().get_values()
    curr_label      = curr_instance.get_label_data().get_values()
    myiterator.log(mystream.C_LOG_TYPE_I, 'Instance', str(i) + ': \n   Data:', curr_data[0:14], '...\n   Label:', curr_label)


# 7. Resetting the iterator
myiterator = iter(mystream)


# 8. Fetching all 1,000,000 instances dark
myiterator.log(mystream.C_LOG_TYPE_W,'Fetching all 1,000,000 instances...')
for i, curr_instance in enumerate(myiterator):
    if i == ( num_inst -1 ): 
        myiterator.log(Log.C_LOG_TYPE_W, 'Rest of the 1,000,000 instances dark...')
        myiterator.switch_logging(p_logging=Log.C_LOG_NOTHING)

    curr_data       = curr_instance.get_feature_data().get_values()
    curr_label      = curr_instance.get_label_data().get_values()
    myiterator.log(mystream.C_LOG_TYPE_I, 'Instance', str(i) + ': \n   Data:', curr_data[0:14], '...\n   Label:', curr_label)

myiterator.switch_logging(p_logging=logging)
myiterator.log(Log.C_LOG_TYPE_W, 'Done!')    
