## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.examples
## -- Module  : howto_bf_systems_030_adaptive_double_pendulum_system.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-07-26  0.0.0     LSB      Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.0.0 (2023-07-26)

This module gives an example of how to pretrain an AFctStrans for using in AdaptiveSystem,
in this case a double pendulum system.

You will learn:

1. ...

2. ...

"""


from mlpro.bf.various import Log
from mlpro.bf.systems.pool import DoublePendulumSystemS4
from mlpro.bf.ml.systems import *
from mlpro.sl.pool.afct.fnn.pytorch.mlp import *
import torch.nn as nn
import torch.optim as opt
from mlpro.sl.models_eval import *
from mlpro.sl.models_train import *
from mlpro.bf.datasets import SASDataset
from pathlib import Path
from mlpro.oa.systems.pool.doublependulum import *


# 0 Prepare Demo/Unit test mode
if __name__ == "__main__":
    cycle_limit = 1000000
    num_epochs  = 5
    logging     = Log.C_LOG_ALL
    visualize   = True
    loop_cycle  = 1000
else:
    cycle_limit = 100
    num_epochs  = 2
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    loop_cycle  = 100



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



# Setting up an AFctStrans
dp_instance = DoublePendulumSystemS4()
state_space, action_space = dp_instance.setup_spaces()


customAFctStrans = OAFctSTrans(p_afct_cls=PyTorchMLP,
                              p_state_space=state_space,
                              p_action_space=action_space,
                              p_output_elem_cls=State,
                              p_threshold=0,
                              p_buffer_size=100,
                              p_num_hidden_layers=4,
                              p_activation_fct=nn.LeakyReLU(0.5),
                              p_output_activation_fct=nn.LeakyReLU(0.5),
                              p_optimizer=opt.Adam,
                              p_batch_size=200,
                              p_metrics=[MSEMetric(p_logging=Log.C_LOG_NOTHING),
                                         MetricAccuracy(p_threshold=10, p_logging=Log.C_LOG_NOTHING)],
                              p_learning_rate=0.0001,
                              p_hidden_size=256,
                              p_loss_fct=nn.MSELoss,
                              p_logging=Log.C_LOG_WE)


# Pretraining the AFctStrans
class AFctStransPretrainingScenario(SLScenario):

    def _setup(self, p_mode, p_ada:bool, p_visualize:bool, p_logging) -> Model:

        self._dataset = SASDataset(p_path=train_path,
                                   p_state_fname=name_train_states,
                                   p_action_fname=name_train_actions,
                                   p_state_space=state_space,
                                   p_action_space=action_space,
                                   p_batch_size=16,
                                   p_shuffle=True,
                                    p_logging=p_logging,
                                   p_eval_split=0.2)

        return customAFctStrans._afct_strans.get_afct()




training = SLTraining(p_scenario_cls = AFctStransPretrainingScenario,
                      p_cycle_limit = 100000000,
                      p_num_epoch=1000,
                      p_logging = Log.C_LOG_WE,
                      p_path = path,
                      p_eval_freq=1,
                      p_collect_mappings=False,
                      p_plot_epoch_scores=True)

training.run()



# Loading the saved model
# Reloading the scenario from saved results of previous training
scenario = AFctStransPretrainingScenario.load(p_filename=training.get_scenario().get_filename(),
                              p_path=training.get_scenario()._get_path())


# Getting the model from the Scenario
# Get the model
afct = scenario.get_model()
# Switch off the adaptivity of the model
afct.switch_adaptivity(False)
# Assigning the model to AFctStrans
customAFctStrans._afct = afct


# Setting up an adaptive DoublePendulumSystem
adaptiveDoublePendulum = DoublePendulumOA4(p_max_torque=10,
                                           p_fct_strans=customAFctStrans,
                                           p_visualize=True,
                                           p_logging=Log.C_LOG_ALL)


demo_scenario_one = DemoScenario(p_system=adaptiveDoublePendulum,
                             p_mode=Mode.C_MODE_SIM,
                             p_cycle_limit=50)

input("Enter to start")
demo_scenario_one.run()
input("Enter to stop")

demo_scenario_two = DemoScenario(p_system=DoublePendulumOA4(p_max_torque=10, p_visualize=True),
                             p_mode=Mode.C_MODE_SIM,
                             p_cycle_limit=50)
input("Enter to start")
demo_scenario_two.run()
input("Enter to stop")





























