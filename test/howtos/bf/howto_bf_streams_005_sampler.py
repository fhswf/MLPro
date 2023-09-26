## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_streams_005_sampler.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-04-10  0.0.0     SY       Creation 
## -- 2023-04-14  1.0.0     SY       First release
## -- 2023-04-16  1.0.1     SY       Add more sampler methods to this howto
## -- 2023-04-17  1.1.0     SY       Replace StreamMLProCSV to Rnd10Dx1000
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2023-04-17)

This howto file demonstrates the incorporation of a stream sampler in a stream scenario.

You will learn:

1) How to access a MLPro's native data stream.

2) How to iterate the instances of a native stream.
    
3) How to incorporate a stream sampler.

4) Update a sampler method with another sampler method after instantiation of a stream

"""


from datetime import datetime
from mlpro.bf.streams.streams import *
from mlpro.bf.streams.samplers import *
from mlpro.bf.various import *
from mlpro.bf.data import *



# 0 Prepare Demo/Unit test mode
if __name__ == '__main__':
    logging     = Log.C_LOG_ALL
else:
    logging     = Log.C_LOG_NOTHING


# 1 Determine a native data stream provided by MLPro
stream = StreamMLProRnd10D( p_logging=logging, p_sampler=SamplerRND(p_max_step_rate=10, p_seed=10) )

if __name__ == '__main__':
    input('\nPress ENTER to iterate all streams dark...\n')


# 2 Iterate all instances of the stream
tp_start = datetime.now()
myiterator = iter(stream)

stream.switch_logging( p_logging=logging )
try:
    labels = stream.get_label_space().get_num_dim()
except:
    labels = 0
stream.log(Log.C_LOG_TYPE_W, 'Features:', stream.get_feature_space().get_num_dim(), ', Labels:', labels, ', Instances:', stream.get_num_instances() )

for i, curr_instance in enumerate(myiterator):
    curr_data = curr_instance.get_feature_data().get_values()
    
myiterator.switch_logging( p_logging=logging )
stream.log(Log.C_LOG_TYPE_W, 'Number of instances being sampled:', int(i+1) )

tp_end       = datetime.now()
duration     = tp_end - tp_start
duration_sec = ( duration.seconds * 1000000 + duration.microseconds + 1 ) / 1000000
rate         = myiterator.get_num_instances() / duration_sec
myiterator.log(Log.C_LOG_TYPE_W, 'Done in', round(duration_sec,3), ' seconds (throughput =', round(rate), 'instances/sec)')   


# 3 Change the sampler method

# 3.1 Weigthed random sampling
stream._sampler = SamplerWeightedRND(p_threshold=0.75, p_seed=10)

for i, curr_instance in enumerate(myiterator):
    curr_data = curr_instance.get_feature_data().get_values()
    
myiterator.switch_logging( p_logging=logging )
stream.log(Log.C_LOG_TYPE_W, 'Number of instances being sampled:', int(i+1) )

# 3.2.1 Reservoir sampling with number of instances
stream._sampler = SamplerReservoir(p_num_instances=stream.get_num_instances(), p_reservoir_size=100, p_seed=10)

for i, curr_instance in enumerate(myiterator):
    curr_data = curr_instance.get_feature_data().get_values()
    
myiterator.switch_logging( p_logging=logging )
stream.log(Log.C_LOG_TYPE_W, 'Number of instances being sampled:', int(i+1) )

# 3.2.2 Reservoir sampling without number of instances
stream._sampler = SamplerReservoir(p_reservoir_size=100, p_seed=10)

for i, curr_instance in enumerate(myiterator):
    curr_data = curr_instance.get_feature_data().get_values()
    
myiterator.switch_logging( p_logging=logging )
stream.log(Log.C_LOG_TYPE_W, 'Number of instances being sampled:', int(i+1) )

# 3.3 Min-wise sampling
stream._sampler = SamplerMinWise(p_cluster_size=20, p_seed=10)

for i, curr_instance in enumerate(myiterator):
    curr_data = curr_instance.get_feature_data().get_values()
    
myiterator.switch_logging( p_logging=logging )
stream.log(Log.C_LOG_TYPE_W, 'Number of instances being sampled:', int(i+1) )

