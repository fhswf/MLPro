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
                           p_csv_filename="SDN-DDoS_Traffic_Dataset.csv",
                           #p_csv_filename="0.csv",
                           p_delimiter=",",
                           p_frame=False,
                           p_header=True,
                           p_list_features=["switch","host","src_ip","dst_ip","pkt_count","byte_count","duration","duration_nsec","tot_duration","flows","packet_per_massg","pktper_flow","byte_per_flow","pkt_rate","pair_flow","Protocol","port_no","tx_bytes","rx_bytes","tx_kbps","rx_kbps","tot_kbps","delay","jitter","packet_loss_rate"],
                           #p_list_features=["from","to","frame_proto","protocol","control_type","type_cont_messg","DOAGID","DOAGID.1","DOAGID.2","DOAGID.3","DOAG_info","DIO_info","object_cont_pt","lifetime","prefix_info","valid_lifetime","preferred_liftime","reserved","desti_prefix","desti_prefix.1","desti_prefix.2","desti_prefix.3"],
                           p_list_labels=["label"])
    
    # 4 Performance test: iterate all data streams dark and measure the time
    input('\nPress ENTER to iterate all streams dark...\n')

    # 4.1 Iterate all instances of the stream
    tp_start = datetime.now()
    myiterator = iter(stream)
    num = myiterator.get_num_instances()

    stream.switch_logging( p_logging=logging )
    try:
        labels = stream.get_label_space().get_num_dim()
    except:
        labels = 0
    stream.log(Log.C_LOG_TYPE_W, 'Features:', stream.get_feature_space().get_num_dim(), ', Labels:', labels, ', Instances:', stream.get_num_instances() )


    for i, curr_instance in enumerate(myiterator):
        curr_data = curr_instance.get_feature_data().get_values()

    print(curr_data[0])





