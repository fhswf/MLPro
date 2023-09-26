## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_streams_001_accessing_native_data_from_mlpro.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-11-08  1.0.0     DA       Creation
## -- 2022-12-14  1.0.1     DA       Corrections
## -- 2023-02-02  1.0.2     DA       Correction of time measurement
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.2 (2023-02-02)

This module demonstrates the use of native generic data streams provided by MLPro. To this regard,
all data streams of the related provider class will be determined and iterated. 

You will learn:

1) How to access MLPro's native data streams.

2) How to iterate the instances of a native stream.

3) How to access feature data of a native stream.

"""


from datetime import datetime
from mlpro.bf.streams.streams import *
from mlpro.bf.various import Log



# 0 Prepare Demo/Unit test mode
if __name__ == '__main__':
    logging     = Log.C_LOG_ALL
else:
    logging     = Log.C_LOG_NOTHING


# 1 Create a Wrapper for OpenML stream provider
mlpro = StreamProviderMLPro(p_logging=logging)


# 2 Determine native data streams provided by MLPro
for stream in mlpro.get_stream_list( p_logging=logging ):
    stream.switch_logging( p_logging=logging )
    try:
        labels = stream.get_label_space().get_num_dim()
    except:
        labels = 0

    stream.log(Log.C_LOG_TYPE_W, 'Features:', stream.get_feature_space().get_num_dim(), ', Labels:', labels, ', Instances:', stream.get_num_instances() )

if __name__ == '__main__':
    input('\nPress ENTER to iterate all streams dark...\n')


# 3 Performance test: iterate all data streams dark and measure the time
for stream in mlpro.get_stream_list( p_logging=logging ):
    stream.switch_logging( p_logging=logging )
    stream.log(Log.C_LOG_TYPE_W, 'Number of instances:', stream.get_num_instances() )
    stream.switch_logging( p_logging=Log.C_LOG_NOTHING )

    # 3.1 Iterate all instances of the stream
    tp_start = datetime.now()
    myiterator = iter(stream)
    for i, curr_instance in enumerate(myiterator):
        curr_data = curr_instance.get_feature_data().get_values()

    tp_end       = datetime.now()
    duration     = tp_end - tp_start
    duration_sec = ( duration.seconds * 1000000 + duration.microseconds + 1 ) / 1000000
    rate         = myiterator.get_num_instances() / duration_sec

    myiterator.switch_logging( p_logging=logging )
    myiterator.log(Log.C_LOG_TYPE_W, 'Done in', round(duration_sec,3), ' seconds (throughput =', round(rate), 'instances/sec)')    
