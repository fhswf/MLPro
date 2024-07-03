## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Module  : howto_bf_streams_002_accessing_data_from_csv_files.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2024-07-01  0.0.0     SK       Creation 
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2024-07-01)

"""


from datetime import datetime
from mlpro.bf.streams.streams import *
from mlpro.bf.various import *
from mlpro.bf.data import *
import random
from pathlib import Path



# 0 Prepare Demo/Unit test mode
if __name__ == '__main__':
    logging     = Log.C_LOG_ALL
    

    path= str(Path.home())
    # 2 Instantiate Stream
    stream = StreamMLProCSV(p_logging=logging,
                           p_path_load=path,
                           p_csv_filename="0.csv",
                           p_delimiter=",",
                           p_frame=True,
                           p_header=True,
                           p_list_features=["from","to","frame_proto","protocol","control_type","type_cont_messg","DOAGID","DOAGID.1","DOAGID.2","DOAGID.3","DOAG_info","DIO_info","object_cont_pt","lifetime","prefix_info","valid_lifetime","preferred_liftime","reserved","desti_prefix","desti_prefix.1","desti_prefix.2","desti_prefix.3"],
                           p_list_labels=["label"])


    # 3. load data from the csv file
    data_names = []
    mem_from_csv = DataStoring(data_names)
    mem_from_csv.load_data(path, "0.csv", ",")
    
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

    tp_end       = datetime.now()
    duration     = tp_end - tp_start
    duration_sec = ( duration.seconds * 1000000 + duration.microseconds + 1 ) / 1000000
    rate         = myiterator.get_num_instances() / duration_sec

    myiterator.switch_logging( p_logging=logging )
    myiterator.log(Log.C_LOG_TYPE_W, 'Done in', round(duration_sec,3), ' seconds (throughput =', round(rate), 'instances/sec)')    
