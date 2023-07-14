## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.sl
## -- Module  : models_data.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-18  0.0.0     LSB      Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2023-06-13)

This module provides dataset classes for supervised learning tasks.
"""



from mlpro.sl.models_dataset import SASDataset
from mlpro.sl.models_train import *
from pathlib import Path
import os
from mlpro.bf.plot import DataPlotting
from mlpro.rl.pool.envs.doublependulum import *
from mlpro.sl.basics import *
from mlpro.sl.pool.afct.fnn.pytorch.mlp import PyTorchMLP
import torch.nn as nn
import torch.optim as opt
from mlpro.sl.models_eval import *


#
path = str(Path.home()) + os.sep

path_state = path + 'data3\env_states.csv'
path_action = path + 'data3\\agent_actions.csv'


dp = DoublePendulumS4()
state_space,action_space = dp.setup_spaces()



print([i.get_name_long() for i in state_space.get_dims()])



class MLPSLScenario(SLScenario):
    C_NAME = 'DP'
    def _setup(self, p_mode, p_ada:bool, p_visualize:bool, p_logging) -> Model:



        self._dataset = SASDataset(p_state_fpath=path_state,
                        p_action_fpath=path_action,
                        p_state_space=state_space,
                        p_action_space=action_space,
                        p_batch_size=16,
                        p_eval_split=0.5,
                        p_shuffle=False,
                        p_logging=Log.C_LOG_WE)


        return PyTorchMLP(p_input_space=self._dataset._feature_space,
                                 p_output_space=self._dataset._label_space,
                                 p_output_elem_cls=BatchElement,
                                 p_num_hidden_layers=3,
                                 p_activation_fct=nn.LeakyReLU,
                                 p_output_activation_fct=nn.LeakyReLU,
                                 p_optimizer=opt.Adam,
                                 p_batch_size=200,
                                 p_metrics=[MSEMetric(p_logging=Log.C_LOG_NOTHING),
                                            MetricAccuracy(p_threshold=10, p_logging=Log.C_LOG_NOTHING)],
                                 p_learning_rate=0.0005,
                                 p_hidden_size=256,
                                 p_loss_fct=nn.MSELoss,
                                 p_logging=Log.C_LOG_WE)


if __name__ == "__main__":
    # 2.1 Parameters for demo mode
    cycle_limit = 100
    num_epochs  = 2
    logging     = Log.C_LOG_WE
    visualize   = True
    path        = str(Path.home())
    plotting    = True
else:
    # 2.2 Parameters for internal unit test
    cycle_limit = 10000
    num_epochs  = 2
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    path        = None
    plotting    = False




training = SLTraining(p_scenario_cls = MLPSLScenario,
                      p_cycle_limit = cycle_limit,
                      p_num_epoch=num_epochs,
                      p_logging = logging,
                      p_path = path,
                      p_eval_freq=1,
                      p_collect_mappings=False,
                      p_plot_epoch_scores=True)



training.run()


scenario = MLPSLScenario.load(p_filename=training.get_scenario().get_filename(),
                              p_path=training.get_scenario()._get_path())

model = scenario.get_model()
model.switch_adaptivity(False)
model._output_elem_cls = Element



class InferenceScenario(SLScenario):
    C_NAME = 'Inference Scenario'
    def _setup(self, p_mode, p_ada:bool, p_visualize:bool, p_logging) -> Model:

        self._dataset = SASDataset(p_state_fpath="C:\\Users\\Baheti\\results-8\\env_states.csv",
                                   p_action_fpath="C:\\Users\\Baheti\\results-8\\agent_actions.csv",
                                   p_state_space=state_space,
                                   p_action_space=action_space,
                                   p_batch_size=1,
                                   p_shuffle=False,
                                   p_normalize = False,
                                   p_logging=Log.C_LOG_NOTHING)






        return model



new_scenario = InferenceScenario(p_path=path, p_collect_mappings=True, p_cycle_limit=300, p_get_mapping_plots=True, p_save_plots=True)
new_scenario.run()
