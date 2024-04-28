## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.sl
## -- Module  : models_data.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-06-18  0.0.0     LSB      Creation
## -- 2023-07-15  1.0.0     LSB      Release
## -- 2023-07-30  1.0.1     LSB      Updates regarding selected output variables
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2023-07-30)

This module provides dataset classes for supervised learning tasks.

You will learn:

1. How to setup a SLScenario and Training

2. How to setup a PytorchMLP

3. How to train an Adaptive Function () AFct on a Dataset

4. How to reload a trained AFct

5. How to do Inference.

Note::
    Please assign the paths for your corresponding states and action csv for training and inference respectively,
    in the corresponding sections 2.1 and 2.3 respectively.
"""



# 1. Importing necessary packages
import os
from pathlib import Path
from mlpro.rl.pool.envs.doublependulum import *
from mlpro.sl.pool.afct.fnn.pytorch.mlp import *
from mlpro.sl import *
from mlpro.bf.datasets.basics import *
import torch.optim as opt
import torch.nn as nn


# 2. Setting Path variables for training and offline dataset resources (CSV files in this case).
path = str(Path.home()) + os.sep

# 2.1 Training Resource
train_path = str(Path.home()) + os.sep + 'data3' + os.sep
name_train_states = 'env_states.csv'
name_train_actions = 'agent_actions.csv'

# 2.2 Inference Resources
inference_path = str(Path.home()) + os.sep + 'results-8' + os.sep
name_infer_states = 'env_states.csv'
name_infer_actions = 'agent_actions.csv'



# 3. Getting the state and action space of the Double Pendulum Environment
dp = DoublePendulumS4()
state_space,action_space = dp.setup_spaces()





# 4. Setting up Demo Scenario
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MLPSLScenario(SLScenario):

    C_NAME = 'DP'

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada:bool, p_visualize:bool, p_logging) -> Model:



        self._dataset = SASDataset(p_path=train_path,
                                    p_state_fname=name_train_states,
                                    p_action_fname=name_train_actions,
                                    p_state_space=state_space,
                                    p_action_space=action_space,
                                    p_op_state_indexes=[0,2],
                                    p_normalize=True,
                                    p_batch_size=16,
                                    p_eval_split=0.5,
                                    p_shuffle=False,
                                    p_logging=Log.C_LOG_WE)


        return PyTorchMLP(p_input_space=self._dataset._feature_space,
                                 p_output_space=self._dataset._label_space,
                                 p_output_elem_cls=BatchElement,
                                 p_num_hidden_layers=5,
                                 p_activation_fct=nn.LeakyReLU(0.5),
                                 p_output_activation_fct=nn.LeakyReLU(1),
                                 p_optimizer=opt.Adam,
                                 p_batch_size=200,
                                 p_metrics=[MSEMetric(p_logging=Log.C_LOG_NOTHING),
                                            MetricAccuracy(p_threshold=20, p_logging=Log.C_LOG_NOTHING)],
                                 p_learning_rate=0.0005,
                                 p_hidden_size=256,
                                 p_loss_fct=nn.MSELoss,
                                 p_logging=Log.C_LOG_WE)




# 5. Preparing parameters for Demo and Unit Test modes.
if __name__ == "__main__":
    # 2.1 Parameters for demo mode
    cycle_limit = 1000000
    num_epochs  = 50
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



# 6. Instantiating the Training Class
training = SLTraining(p_scenario_cls = MLPSLScenario,
                      p_cycle_limit = cycle_limit,
                      p_num_epoch=num_epochs,
                      p_logging = logging,
                      p_path = path,
                      p_eval_freq=1,
                      p_collect_mappings=False,
                      p_plot_epoch_scores=True)


# 7. Running the training
training.run()



# 8. Reloading the scenario from saved results of previous training
scenario = MLPSLScenario.load(p_filename=training.get_scenario().get_filename(),
                              p_path=training.get_scenario()._get_path())



# 9. Getting the model from the Scenario
# Get the model
model = scenario.get_model()
# Switch off the adaptivity of the model
model.switch_adaptivity(False)






# 10. Setting up an Inference Scenario
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class InferenceScenario(SLScenario):

    C_NAME = 'Inference'

    def _setup(self, p_mode, p_ada:bool, p_visualize:bool, p_logging) -> Model:

        self._dataset = SASDataset(p_path = inference_path,
                                   p_state_fname = name_infer_states,
                                   p_action_fname = name_infer_actions,
                                   p_state_space=state_space,
                                   p_action_space=action_space,
                                   p_op_state_indexes=[0,2],
                                   p_batch_size=1,
                                   p_shuffle=False,
                                   p_normalize = True,
                                   p_logging=Log.C_LOG_NOTHING)






        return model


# 11. Instantiating the scenario
new_scenario = InferenceScenario(p_path=path,
                                 p_collect_mappings=True,
                                 p_cycle_limit=300,
                                 p_get_mapping_plots=True,
                                 p_save_plots=True)



# 12. Running the scenario
new_scenario.run()
