## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_streams_051_accessing_data_from_openml.py
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
## -- 2022-11-08  1.1.1     DA       Minor improvements
## -- 2022-11-11  1.1.2     LSB      Refactoring for the new set options method
## -- 2023-02-02  1.1.3     DA       Bugfix in time measurement
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.3 (2023-02-02)

This module demonstrates the use of OpenML datasets as streams in MLPro. To this regard, MLPro
provides wrapper classes to standardize stream access in own ML applications.

You will learn:

1) How to access datasets of the OpenML project.

2) How to iterate the instances of an OpenML stream.

3) How to access feature and label data of a data stream.

"""


from datetime import datetime
from mlpro.bf.various import Log
from mlpro.wrappers.openml import WrStreamProviderOpenML




# 0 Prepare Demo/Unit test mode
if __name__ == '__main__':
    num_inst    = 10
    logging     = Log.C_LOG_ALL
else:
    print('\n', datetime.now(), __file__)
    num_inst    = 2
    logging     = Log.C_LOG_NOTHING



# 1 Create a Wrapper for OpenML stream provider
openml = WrStreamProviderOpenML(p_logging = logging)



# 2 Get a list of streams available at the stream provider
stream_list = openml.get_stream_list(p_logging = logging)



# 3 Get stream "credit-g" from the stream provider OpenML
mystream = openml.get_stream( p_name='BNG(autos,nominal,1000000)', p_logging=logging)


# 4 Setting up additional stream options, the target label in this case
#mystream.set_options(target = 'checking_status')


# 5 Get the feature space of the stream
feature_space = mystream.get_feature_space()
openml.log(mystream.C_LOG_TYPE_I,"Number of features in the stream:",feature_space.get_num_dim())


# 6 Set up an iterator for the stream
myiterator = iter(mystream)


# 7 Fetching some stream instances
myiterator.log(mystream.C_LOG_TYPE_W, 'Fetching first', str(num_inst), 'stream instances...')
for i in range(num_inst):
    curr_instance   = next(myiterator)
    curr_data       = curr_instance.get_feature_data().get_values()
    curr_label      = curr_instance.get_label_data().get_values()
    myiterator.log(mystream.C_LOG_TYPE_I, 'Instance', str(i) + ': \n   Data:', curr_data[0:14], '...\n   Label:', curr_label)


# 8 Resetting the iterator
myiterator = iter(mystream)


# 9 Fetching all 1,000 instances
myiterator.log(mystream.C_LOG_TYPE_W,'Fetching all', myiterator.get_num_instances(), 'instances...')
for i, curr_instance in enumerate(myiterator):
    if i == num_inst: 
        myiterator.log(Log.C_LOG_TYPE_W, 'Rest of the', myiterator.get_num_instances(), 'instances dark...')
        myiterator.switch_logging(p_logging=Log.C_LOG_NOTHING)
        tp_start = datetime.now()

    curr_data       = curr_instance.get_feature_data().get_values()
    curr_label      = curr_instance.get_label_data().get_values()
    myiterator.log(mystream.C_LOG_TYPE_I, 'Instance', str(i) + ': \n   Data:', curr_data[0:14], '...\n   Label:', curr_label)

# 9.1 Some statistics...
tp_end = datetime.now()
duration = tp_end - tp_start
duration_sec = ( duration.seconds * 1000000 + duration.microseconds + 1 ) / 1000000
rate = ( myiterator.get_num_instances() - num_inst ) / duration_sec

myiterator.switch_logging(p_logging=logging)
myiterator.log(Log.C_LOG_TYPE_W, 'Done in', round(duration_sec,3), ' seconds (throughput =', round(rate), 'instances/sec)')    