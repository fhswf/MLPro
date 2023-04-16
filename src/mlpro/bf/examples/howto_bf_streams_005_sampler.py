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
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2023-04-16)

This howto file demonstrates the incorporation of a stream sampler in a stream scenario with loading
and converting data stored in csv files to be compatible for data streams provided by MLPro.

You will learn:

1) How to load and convert data from csv files.

2) How to iterate the instances of a native stream.

3) How to access feature data of a native stream.
    
4) How to incorporate a stream sampler.

5) Update a sampler method with another sampler method after instantiation of a stream

"""


from datetime import datetime
from mlpro.bf.streams.streams import *
from mlpro.bf.streams.samplers import *
from mlpro.bf.various import *
from mlpro.bf.data import *
import random
from pathlib import Path



# 0 Prepare Demo/Unit test mode
if __name__ == '__main__':
    logging     = Log.C_LOG_ALL
    
    
    # 1 Generate random data and store them in csv format
    num_eps         = 10
    num_cycles      = 10000
    data_names      = ["action","states_1","states_2","model_loss"]
    data_printing   = {"action":        [True,0,10],
                       "states_1":      [True,0,4],
                       "states_2":      [True,0,4],
                       "model_loss":    [True,0,-1]}

    mem = DataStoring(data_names)
    for ep in range(num_eps):
        ep_id = ("ep. %s"%str(ep+1))
        mem.add_frame(ep_id)
        for i in range(num_cycles):
            mem.memorize("action",ep_id,random.uniform(0+(ep*0.5),5+(ep*0.5)))
            mem.memorize("states_1",ep_id,random.uniform(2-(ep*0.2),4-(ep*0.2)))
            mem.memorize("states_2",ep_id,random.uniform(0+(ep*0.2),2+(ep*0.2)))
            mem.memorize("model_loss",ep_id,random.uniform(0.25-(ep*0.02),1-(ep*0.07)))
            
    path_save = str(Path.home())
    mem.save_data(path_save, "data_storage", "\t")


    # 2 Instantiate Stream
    stream = StreamMLProCSV(p_logging=logging,
                           p_path_load=path_save,
                           p_csv_filename="data_storage.csv",
                           p_delimiter="\t",
                           p_frame=True,
                           p_header=True,
                           p_sampler=SamplerRND(p_max_step_rate=10, p_seed=10),
                           p_list_features=["action", "states_1", "states_2"],
                           p_list_labels=["model_loss"])


    # 3. load data from the csv file
    data_names = []
    mem_from_csv = DataStoring(data_names)
    mem_from_csv.load_data(path_save, "data_storage.csv", "\t")
    

    # 4 Performance test: iterate all data streams dark and measure the time
    input('\nPress ENTER to iterate all streams dark...\n')

    # 4.1 Iterate all instances of the stream
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
    
    
    # 5 Change the sampler method

    # 5.1 Weigthed random sampling
    stream._sampler = SamplerWeightedRND(p_threshold=0.75, p_seed=10)

    for i, curr_instance in enumerate(myiterator):
        curr_data = curr_instance.get_feature_data().get_values()
        
    myiterator.switch_logging( p_logging=logging )
    stream.log(Log.C_LOG_TYPE_W, 'Number of instances being sampled:', int(i+1) )

    # 5.2.1 Reservoir sampling with number of instances
    stream._sampler = SamplerReservoir(p_num_instances=stream.get_num_instances(), p_reservoir_size=100, p_seed=10)

    for i, curr_instance in enumerate(myiterator):
        curr_data = curr_instance.get_feature_data().get_values()
        
    myiterator.switch_logging( p_logging=logging )
    stream.log(Log.C_LOG_TYPE_W, 'Number of instances being sampled:', int(i+1) )

    # 5.2.2 Reservoir sampling without number of instances
    stream._sampler = SamplerReservoir(p_reservoir_size=100, p_seed=10)

    for i, curr_instance in enumerate(myiterator):
        curr_data = curr_instance.get_feature_data().get_values()
        
    myiterator.switch_logging( p_logging=logging )
    stream.log(Log.C_LOG_TYPE_W, 'Number of instances being sampled:', int(i+1) )

    # 5.3 Min-wise sampling
    stream._sampler = SamplerMinWise(p_cluster_size=20, p_seed=10)

    for i, curr_instance in enumerate(myiterator):
        curr_data = curr_instance.get_feature_data().get_values()
        
    myiterator.switch_logging( p_logging=logging )
    stream.log(Log.C_LOG_TYPE_W, 'Number of instances being sampled:', int(i+1) )
    