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



path = str(Path.home()) + os.sep

path_state = path + 'Results-2\env_states.csv'
path_action = path + 'Results-2\\agent_actions.csv'


dp = DoublePendulumS4()
state_space,action_space = dp.setup_spaces()

print([i.get_name_long() for i in state_space.get_dims()])

mydataset = SASDataset(p_state_fpath=path_state,
                        p_action_fpath=path_action,
                        p_state_space=state_space,
                        p_action_space=action_space,
                       p_batch_size=300,
                       p_logging=Log.C_LOG_WE)



myMLP = PyTorchMLP(p_input_space=mydataset._feature_space,
                   p_output_space=mydataset._label_space,
                   p_num_hidden_layers = 3,
                   p_activation_fct = nn.LeakyReLU,
                   p_output_activation_fct=nn.LeakyReLU,
                   p_optimizer = opt.Adam,
                   p_batch_size = 200,
                   p_metrics= [MetricAccuracy(p_threshold=100)],
                   p_learning_rate = 0.001,
                   p_hidden_size = 128,
                   p_loss_fct = nn.MSELoss,
                   p_logging = Log.C_LOG_WE)

class MLPSLScenario(SLScenario):

    def _setup(self, p_mode, p_ada:bool, p_visualize:bool, p_logging) -> Model:

        self._model = myMLP

        self._dataset = mydataset

        return self._model


training = SLTraining(p_scenario_cls = MLPSLScenario,
                      p_cycle_limit = 10000,
                      p_num_epoch=2,
                      p_logging = Log.C_LOG_WE,
                      p_path = str(Path.home()))



training.run()

plots = DataPlotting(p_data=training.get_results().ds_mapping_train,
                     p_printing={'input w1':[True, 0, -1],
                                'input th2':[True, 0 , -1],
                                'pred th1': [True, 0, -1],
                                'pred th2': [True, 0, -1]},
                     p_type=DataPlotting.C_PLOT_TYPE_EP,
                     p_window = 1)
plots.get_plots()
plots.save_plots(p_path = training.get_training_path(),
                 p_format = 'jpg')

acc_plot = DataPlotting(p_data=training.get_results().ds_cycles_train,
                        p_printing={'acc':[True, 0, -1]},
                        p_type=DataPlotting.C_PLOT_TYPE_EP,
                        p_window=100)
acc_plot.get_plots()