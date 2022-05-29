## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : howto_bf_007_hyperparameter_tuning_using_hyperopt.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-12-08  0.0.0     SY       Creation
## -- 2021-12-08  1.0.0     SY       Release of first version
## -- 2022-01-21  1.0.1     DA       Renaming: tupel -> tuple
## -- 2022-01-27  1.0.2     SY       Class WrHPTHyperopt enhancement
## -- 2022-02-25  1.0.3     SY       Refactoring due to auto generated ID in class Dimension
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.3 (2022-02-25)

This module demonstrates how to utilize wrapper class for Hyperopt in RL context.
"""


from mlpro.wrappers.hyperopt import *
from mlpro.rl.pool.envs.bglp import BGLP
from mlpro.rl.models import *
import random
from pathlib import Path




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

# 1. Create a policy and setup the hyperparameters
class myPolicy (Policy):

    C_NAME      = 'MyPolicy'
    

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_observation_space:MSpace, p_action_space:MSpace, p_buffer_size=1, p_ada=True, p_logging=True):
        """
         Parameters:
            p_observation_space     Subspace of an environment that is observed by the policy
            p_action_space          Action space object
            p_buffer_size           Size of the buffer
            p_ada                   Boolean switch for adaptivity
            p_logging               Boolean switch for logging functionality
        """
        super().__init__(p_observation_space, p_action_space, p_buffer_size, p_ada, p_logging)
        self._hyperparam_space  = HyperParamSpace()
        self._hyperparam_tuple  = None
        self._init_hyperparam()
    

## -------------------------------------------------------------------------------------------------
    def set_random_seed(self, p_seed=None):
        random.seed(p_seed)
    

## -------------------------------------------------------------------------------------------------
    def _init_hyperparam(self):
        self._hyperparam_space.add_dim(HyperParam('num_states','Z', p_boundaries = [1,100]))
        self._hyperparam_space.add_dim(HyperParam('smoothing','R', p_boundaries = [0.1,0.5]))
        self._hyperparam_space.add_dim(HyperParam('lr_rate','R', p_boundaries = [0.001,0.1]))
        self._hyperparam_space.add_dim(HyperParam('buffer_size','Z', p_boundaries = [10000,100000]))
        self._hyperparam_space.add_dim(HyperParam('update_rate','Z', p_boundaries = [5,20]))
        self._hyperparam_space.add_dim(HyperParam('sampling_size','Z', p_boundaries = [64,256]))
        self._hyperparam_tuple = HyperParamTuple(self._hyperparam_space)
        
        ids_ = self._hyperparam_tuple.get_dim_ids()
        self._hyperparam_tuple.set_value(ids_[0], 100)
        self._hyperparam_tuple.set_value(ids_[1], 0.035)
        self._hyperparam_tuple.set_value(ids_[2], 0.0001)
        self._hyperparam_tuple.set_value(ids_[3], 100000)
        self._hyperparam_tuple.set_value(ids_[4], 100)
        self._hyperparam_tuple.set_value(ids_[5], 256)
    

## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_state: State) -> Action:
        my_action_values = np.zeros(self._action_space.get_num_dim())
        for d in range(self._action_space.get_num_dim()):
            self.set_random_seed(None)
            my_action_values[d] = random.random() 
        return Action(self._id, self._action_space, my_action_values)
    

## -------------------------------------------------------------------------------------------------
    def _adapt(self, *p_args) -> bool:
        self.log(self.C_LOG_TYPE_W, 'Sorry, I am a stupid agent...')
        return False





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

# 2. Create a Scenario
class BGLP_Rnd(RLScenario):

    C_NAME      = 'BGLP_Dummy'
    

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada, p_logging):
        self._env       = BGLP(p_logging=True)
        self._agent     = MultiAgent(p_name='Dummy Policy', p_ada=1, p_logging=False)
        state_space     = self._env.get_state_space()
        action_space    = self._env.get_action_space()
        
        
        # Agent 1
        _name         = 'BELT_CONVEYOR_A'
        _id           = 0
        _ospace       = state_space.spawn([state_space.get_dim_ids()[0],state_space.get_dim_ids()[1]])
        _aspace       = action_space.spawn([action_space.get_dim_ids()[0]])
        _policy       = myPolicy(p_observation_space=_ospace, p_action_space=_aspace, p_buffer_size=1, p_ada=1, p_logging=False)
        self._agent.add_agent(
            p_agent=Agent(
                p_policy=_policy,
                p_envmodel=None,
                p_name=_name,
                p_id=_id,
                p_ada=True,
                p_logging=True),
            p_weight=1.0
            )
        
        
        # Agent 2
        _name         = 'VACUUM_PUMP_B'
        _id           = 1
        _ospace       = state_space.spawn([state_space.get_dim_ids()[1],state_space.get_dim_ids()[2]])
        _aspace       = action_space.spawn([action_space.get_dim_ids()[1]])
        _policy       = myPolicy(p_observation_space=_ospace, p_action_space=_aspace, p_buffer_size=1, p_ada=1, p_logging=False)
        self._agent.add_agent(
            p_agent=Agent(
                p_policy=_policy,
                p_envmodel=None,
                p_name=_name,
                p_id=_id,
                p_ada=True,
                p_logging=True),
            p_weight=1.0
            )
        
        
        # Agent 3
        _name         = 'VIBRATORY_CONVEYOR_B'
        _id           = 2
        _ospace       = state_space.spawn([state_space.get_dim_ids()[2],state_space.get_dim_ids()[3]])
        _aspace       = action_space.spawn([action_space.get_dim_ids()[2]])
        _policy       = myPolicy(p_observation_space=_ospace, p_action_space=_aspace, p_buffer_size=1, p_ada=1, p_logging=False)
        self._agent.add_agent(
            p_agent=Agent(
                p_policy=_policy,
                p_envmodel=None,
                p_name=_name,
                p_id=_id,
                p_ada=True,
                p_logging=True),
            p_weight=1.0
            )
        
        
        # Agent 4
        _name         = 'VACUUM_PUMP_C'
        _id           = 3
        _ospace       = state_space.spawn([state_space.get_dim_ids()[3],state_space.get_dim_ids()[4]])
        _aspace       = action_space.spawn([action_space.get_dim_ids()[3]])
        _policy       = myPolicy(p_observation_space=_ospace, p_action_space=_aspace, p_buffer_size=1, p_ada=1, p_logging=False)
        self._agent.add_agent(
            p_agent=Agent(
                p_policy=_policy,
                p_envmodel=None,
                p_name=_name,
                p_id=_id,
                p_ada=True,
                p_logging=True),
            p_weight=1.0
            )
        
        
        # Agent 5
        _name         = 'ROTARY_FEEDER_C'
        _id           = 4
        _ospace       = state_space.spawn([state_space.get_dim_ids()[4],state_space.get_dim_ids()[5]])
        _aspace       = action_space.spawn([action_space.get_dim_ids()[4]])
        _policy       = myPolicy(p_observation_space=_ospace, p_action_space=_aspace, p_buffer_size=1, p_ada=1, p_logging=False)
        self._agent.add_agent(
            p_agent=Agent(
                p_policy=_policy,
                p_envmodel=None,
                p_name=_name,
                p_id=_id,
                p_ada=True,
                p_logging=True),
            p_weight=1.0
            )
        
        return self._agent





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Parameters for demo mode
    logging         = Log.C_LOG_WE
    visualize       = False
    dest_path       = str(Path.home())
    cycle_limit     = 100
    cycle_per_ep    = 10
    eval_freq       = 2
    eval_grp_size   = 5
    adapt_limit     = 0
    stagnant_limit  = 5
    score_ma_hor    = 5
 
else:
    # Parameters for internal unit test
    logging         = Log.C_LOG_NOTHING
    visualize       = False
    dest_path       = str(Path.home())
    cycle_limit     = 100
    cycle_per_ep    = 10
    eval_freq       = 2
    eval_grp_size   = 1
    adapt_limit     = 0
    stagnant_limit  = 0
    score_ma_hor    = 0


# 3. Instantiate a hyperopt wrapper
myHyperopt = WrHPTHyperopt(p_logging=Log.C_LOG_ALL,
                           p_algo=WrHPTHyperopt.C_ALGO_TPE,
                           p_ids=None)
    

# 4. Train players in the scenario and turn the hyperparamter tuning on
training        = RLTraining(
    p_scenario_cls=BGLP_Rnd,
    p_cycle_limit=cycle_limit,
    p_cycles_per_epi_limit=cycle_per_ep,
    p_eval_frequency=eval_freq,
    p_eval_grp_size=eval_grp_size,
    p_adaptation_limit=adapt_limit,
    p_stagnation_limit=stagnant_limit,
    p_score_ma_horizon=score_ma_hor,
    p_hpt=myHyperopt,
    p_hpt_trials=10,
    p_collect_states=True,
    p_collect_actions=True,
    p_collect_rewards=True,
    p_path=dest_path,
    p_logging=logging
)

training.run()
