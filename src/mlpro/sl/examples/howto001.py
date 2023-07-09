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
                       p_batch_size=50,
                       p_eval_split=0.3,
                       p_logging=Log.C_LOG_WE)


        return PyTorchMLP(p_input_space=self._dataset._feature_space,
                                 p_output_space=self._dataset._label_space,
                                 p_output_elem_cls=BatchElement,
                                 p_num_hidden_layers=3,
                                 p_activation_fct=nn.LeakyReLU,
                                 p_output_activation_fct=nn.LeakyReLU,
                                 p_optimizer=opt.Adam,
                                 p_batch_size=200,
                                 # p_metrics= [MSEMetric(p_loggin
                                 p_metrics=[MSEMetric(p_logging=Log.C_LOG_NOTHING),
                                            MetricAccuracy(p_threshold=50, p_logging=Log.C_LOG_NOTHING)],
                                 p_learning_rate=0.0001,
                                 p_hidden_size=128,
                                 p_loss_fct=nn.MSELoss,
                                 p_logging=Log.C_LOG_WE)


if __name__ == "__main__":
    # 2.1 Parameters for demo mode
    cycle_limit = 100000
    num_epochs  = 50
    logging     = Log.C_LOG_ALL
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
                      p_logging = Log.C_LOG_WE,
                      p_path = path,
                      p_eval_freq=1)



training.run()




#
# plots = DataPlotting(p_data=training.get_results().ds_mapping_train,
#                      p_printing={'input w1':[True, 0, -1],
#                                 'input th2':[True, 0 , -1],
#                                 'pred th1': [True, 0, -1],
#                                 'pred th2': [True, 0, -1]},
#                      p_type=DataPlotting.C_PLOT_TYPE_EP,
#                      p_window = 1)
# plots.get_plots()
# plots.save_plots(p_path = training.get_training_path(),
#                  p_format = 'jpg')

acc_plot = DataPlotting(p_data=training.get_results().ds_epoch,
                        p_printing={'MSE':[True, 0, -1], 'acc' : [True, 0, -1]},
                        p_type=DataPlotting.C_PLOT_TYPE_EP,
                        p_window=1)
acc_plot.get_plots()

scenario_f_name = training.get_scenario().get_filename()
print(scenario_f_name)
scenario_path = training.get_scenario()._get_path()
print(scenario_f_name)
scenario = MLPSLScenario.load(p_filename=scenario_f_name, p_path=scenario_path)

model = scenario.get_model()
model.switch_adaptivity(False)


print(model._sl_model)



class InferenceScenario(SLScenario):

    def _setup(self, p_mode, p_ada:bool, p_visualize:bool, p_logging) -> Model:

        self._dataset = SASDataset(p_state_fpath=path_state,
                                   p_action_fpath=path_action,
                                   p_state_space=state_space,
                                   p_action_space=action_space,
                                   p_batch_size=1,
                                   p_logging=Log.C_LOG_WE)

        return model


new_scenario = InferenceScenario( p_mode=Mode.C_MODE_SIM,
                                p_ada= False,
                                p_cycle_limit= 1000,
                                p_visualize = False,
                                p_logging=Log.C_LOG_NOTHING)

new_scenario.run()