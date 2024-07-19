
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
from mlpro.bf.streams.tasks import Rearranger
from mlpro.oa.streams import *

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyScenario(OAScenario):

    C_NAME = 'Experiment'

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging):

        # 1 Prepare the stream from CSV
        path = str(Path.home())
        
        # 2 Instantiate Stream
        stream = StreamMLProCSV(p_logging=p_logging,
                            p_path_load=path,
                            p_csv_filename="SDN-DDoS_Traffic_Dataset.csv",
                            p_delimiter=",",
                            p_frame=True,
                            p_header=True,
                            p_list_features=["switch", "host", "src_ip", "dst_ip", "pkt_count", "byte_count", "duration", "duration_nsec", "tot_duration", "flows", "packet_per_massg", "pktper_flow", "byte_per_flow", "pkt_rate", "pair_flow", "Protocol", "port_no", "tx_bytes", "rx_bytes", "tx_kbps", "rx_kbps", "tot_kbps", "delay", "jitter", "packet_loss_rate"],
                            p_list_labels=["label"])

        tp_start = datetime.now()
        myiterator = iter(stream)
        print('Features:', stream.get_feature_space().get_num_dim(),
              ', Labels:', stream.get_label_space().get_num_dim(),
              ', Instances:', stream.get_num_instances() )
        for i, curr_instance in enumerate(myiterator):
            curr_data = curr_instance.get_feature_data().get_values()


        # 3.1 Creation of a workflow
        workflow = OAWorkflow(p_name='Input Signal',
                              p_range_max=OAWorkflow.C_RANGE_NONE,
                              p_ada=p_ada,
                              p_visualize=False, 
                              p_logging=p_logging)
                              

        # 3.2 Rearranger to reduce the number of features
        features = stream.get_feature_space().get_dims()
        features_new = [('F', features[10])]

        task_rearranger = Rearranger(p_name='T1 - Rearranger',
                                     p_range_max=Task.C_RANGE_NONE,
                                     p_visualize=True,
                                     p_logging=p_logging,
                                     p_features_new=features_new)

        workflow.add_task(p_task=task_rearranger)

        # 4 Return stream and workflow
        return stream, workflow

# 1 Preparation of demo/unit test mode
if __name__ == '__main__':
    cycle_limit = 1000
    logging = Log.C_LOG_ALL
    visualize = True
    step_rate = 1
else:
    cycle_limit = 2
    logging = Log.C_LOG_NOTHING
    visualize = False
    step_rate = 1

# 2 Instantiate the stream scenario
myscenario = MyScenario(p_mode=Mode.C_MODE_SIM,
                        p_cycle_limit=cycle_limit,
                        p_visualize=visualize,
                        p_logging=logging)

# 3 Reset and run own stream scenario
myscenario.reset()

if __name__ == '__main__':
    myscenario.init_plot(p_plot_settings=PlotSettings(p_view=PlotSettings.C_VIEW_ND,
                                                      p_step_rate=step_rate))
    input('\nPlease arrange all windows and press ENTER to start stream processing...')

myscenario.run()

# 4 Summary
