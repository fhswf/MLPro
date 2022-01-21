## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : Howto 07 - (ML) Hyperparameter Tuning using Hyperopt
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-12-08  0.0.0     SY       Creation
## -- 2021-12-08  1.0.0     SY       Release of first version
## -- 2022-01-21  1.0.1     DA       Renaming: tupel -> tuple
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2022-01-21)

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
        self._hyperparam_space.add_dim(HyperParam(0,'num_states','Z', p_boundaries = [1,100]))
        self._hyperparam_space.add_dim(HyperParam(1,'smoothing','R', p_boundaries = [0.1,0.5]))
        self._hyperparam_space.add_dim(HyperParam(2,'lr_rate','R', p_boundaries = [0.001,0.1]))
        self._hyperparam_space.add_dim(HyperParam(3,'buffer_size','Z', p_boundaries = [10000,100000]))
        self._hyperparam_space.add_dim(HyperParam(4,'update_rate','Z', p_boundaries = [5,20]))
        self._hyperparam_space.add_dim(HyperParam(5,'sampling_size','Z', p_boundaries = [64,256]))
        self._hyperparam_tuple = HyperParamTuple(self._hyperparam_space)
        
        self._hyperparam_tuple.set_value(0, 100)
        self._hyperparam_tuple.set_value(1, 0.035)
        self._hyperparam_tuple.set_value(2, 0.0001)
        self._hyperparam_tuple.set_value(3, 100000)
        self._hyperparam_tuple.set_value(4, 100)
        self._hyperparam_tuple.set_value(4, 256)
    

## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_state: State) -> Action:
        my_action_values = np.zeros(self._action_space.get_num_dim())
        for d in range(self._action_space.get_num_dim()):
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
        _ospace       = state_space.spawn([0,1])
        _aspace       = action_space.spawn([0])
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
        _ospace       = state_space.spawn([1,2])
        _aspace       = action_space.spawn([1])
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
        _ospace       = state_space.spawn([2,3])
        _aspace       = action_space.spawn([2])
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
        _ospace       = state_space.spawn([3,4])
        _aspace       = action_space.spawn([3])
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
        _ospace       = state_space.spawn([4,5])
        _aspace       = action_space.spawn([4])
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
    logging     = Log.C_LOG_ALL
    visualize   = False
    dest_path   = str(Path.home())
 
else:
    # Parameters for internal unit test
    logging     = Log.C_LOG_NOTHING
    visualize   = False
    dest_path   = None


# 3. Instantiate a hyperopt wrapper
myHyperopt = WrHPTHyperopt(p_logging=logging,
                           p_algo=WrHPTHyperopt.C_ALGO_TPE,
                           p_ids=None)
    

# 4. Instantiate a scenario
myscenario  = BGLP_Rnd(
    p_mode=Mode.C_MODE_SIM,
    p_ada=True,
    p_cycle_limit=0,
    p_visualize=visualize,
    p_logging=logging
)


# 5. Train players in the scenario and turn the hyperparamter tuning on
training        = RLTraining(
    p_scenario=myscenario,
    p_max_cycles_per_episode=1,
    p_cycle_limit=10,
    p_max_adaptations=0,
    p_max_stagnations=0,
    p_eval_frequency=0,
    p_hpt=myHyperopt,
    p_hpt_trials=10,
    p_collect_states=True,
    p_collect_actions=True,
    p_collect_rewards=True,
    p_path=dest_path,
    p_logging=logging
)

training.run()
