## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl
## -- Module  : models_train.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-04-18  0.0.0     DA       Creation
## -- 2021-06-06  1.0.0     DA       Release of first version
## -- 2021-06-25  1.1.0     DA       Extension of classes Scenario and Training by data logging; 
## --                                New method Training.save_data()
## -- 2021-07-01  1.1.1     DA       Bugfixes in class Training
## -- 2021-07-06  1.1.2     SY       Update method Training.save_data()
## -- 2021-08-26  1.2.0     DA       New class HPTuningRL; incompatible changes on class Scenario
## -- 2021-08-28  1.2.1     DA       Bugfixes and minor improvements
## -- 2021-09-11  1.2.2     MRD      Change Header information to match our new library name
## -- 2021-10-05  1.2.3     SY       Bugfixes and minor improvements
## -- 2021-10-08  1.2.4     DA       Class Scenario/constructor/param p_cycle_limit: new value -1
## --                                lets class get the cycle limit from the env
## -- 2021-10-28  1.2.5     DA       Bugfix method Scenario.reset(): agent's buffer was not cleared
## -- 2021-11-13  1.3.0     DA       Rework/improvement of class Training
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.3.0 (2021-11-13)

This module provides model classes to define and run rl scenarios and to train agents inside them.
"""


from os import error
from mlpro.bf.various import *
from mlpro.bf.math import *
from mlpro.bf.data import *
from mlpro.bf.ml import *
from mlpro.rl.models_env import *




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLDataStoring(DataStoring):
    """
    Derivate of basic class DataStoring that is specialized to store episodical training data in the
    context of reinforcement learning.
    """

    # Frame ID renamed
    C_VAR0              = 'Episode ID'

    # Variables for training header data storage
    C_VAR_NUM_CYCLES    = 'Number of cycles'
    C_VAR_ENV_DONE      = 'Goal reached'
    C_VAR_ENV_BROKEN    = 'Env broken'

    # Variables for episodical detail data storage
    C_VAR_CYCLE         = 'Cycle'
    C_VAR_DAY           = 'Day'
    C_VAR_SEC           = 'Second'
    C_VAR_MICROSEC      = 'Microsecond'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_space:Set=None):
        """
        Parameters:
            p_space         Space object that provides dimensional information for raw data. If None
                            a training header data object will be instantiated.
        """
        
        self.space = p_space

        if self.space is None:
            # Initialization as a training header data storage
            self.variables  = [ self.C_VAR_NUM_CYCLES, self.C_VAR_ENV_DONE, self.C_VAR_ENV_BROKEN ]

        else:
            # Initalization as an episodical detail data storage
            self.variables  = [ self.C_VAR_CYCLE, self.C_VAR_DAY, self.C_VAR_SEC, self.C_VAR_MICROSEC ]
            self.var_space  = []
    
            for dim_id in self.space.get_dim_ids():
                dim = self.space.get_dim(dim_id)
                self.var_space.append(dim.get_name_short())

            self.variables.extend(self.var_space)

        super().__init__(self.variables)


## -------------------------------------------------------------------------------------------------
    def get_variables(self):
        return self.variables


## -------------------------------------------------------------------------------------------------
    def get_space(self):
        return self.space


## -------------------------------------------------------------------------------------------------
    def add_episode(self, p_episode_id):
        self.add_frame(p_episode_id)
        self.current_episode = p_episode_id


## -------------------------------------------------------------------------------------------------
    def memorize_row(self, p_cycle_id, p_tstamp:timedelta, p_data):
        """
        Memorizes an episodical data row.

        Parameters: 
            p_cycle_id          Cycle id
            p_tstamp            Time stamp
            p_data              Data that meet the dimensionality of the related space
        """

        self.memorize(self.C_VAR_CYCLE, self.current_episode, p_cycle_id)
        self.memorize(self.C_VAR_DAY, self.current_episode, p_tstamp.days)
        self.memorize(self.C_VAR_SEC, self.current_episode, p_tstamp.seconds)
        self.memorize(self.C_VAR_MICROSEC, self.current_episode, p_tstamp.microseconds)

        for i, var in enumerate(self.var_space):
            self.memorize(var, self.current_episode, p_data[i])





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLScenario (Scenario):
    """
    Template class for an RL scenario consisting of an environment and an agent. 
    """

    C_TYPE              = 'RL-Scenario'
    C_NAME              = '????'

## -------------------------------------------------------------------------------------------------
    def __init__(self, 
                 p_mode=Mode.C_MODE_SIM,        # Operation mode (see class Mode)
                 p_ada:bool=True,               # Boolean switch for adaptivity of internal model
                 p_cycle_len:timedelta=None,    # Fixed cycle duration (optional)
                 p_cycle_limit=0,               # Maximum number of cycles (0=no limit, -1=get from env)
                 p_visualize=True,              # Boolean switch for env/agent visualisation
                 p_logging=Log.C_LOG_ALL ):     # Log level (see constants of class Log)

        # 1 Setup entire scenario
        self._env   = None
        super().__init__(p_mode=p_mode, p_ada=p_ada, p_cycle_len=p_cycle_len, p_cycle_limit=p_cycle_limit, p_visualize=p_visualize, p_logging=p_logging)
        if self._env is None: 
            raise ImplementationError('Please bind your RL environment to self._env')

        self._agent = self._model


        # 2 Finalize cycle limit
        if self._cycle_limit == -1: 
            self._cycle_limit = self._env.get_cycle_limit()


        # 3 Init data logging
        self.connect_data_logger()

         
## -------------------------------------------------------------------------------------------------
    def switch_logging(self, p_logging):
        super().switch_logging(p_logging)
        self._env.switch_logging(p_logging)


## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada: bool, p_logging: bool) -> Model:
        """
        Setup the ML scenario by redefinition. Please bind your environment to self._env and return 
        the agent as model. 

        Parameters:
            p_mode          Operation mode (see class Mode)
            p_ada           Boolean switch for adaptivity of internal model
            p_logging       Boolean switch for logging functionality

        Returns:
            Agent model (object of type Agent or Multiagent)
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure=None):
        super().init_plot(p_figure=p_figure)
        self._env.init_plot(p_figure=p_figure)


## -------------------------------------------------------------------------------------------------
    def update_plot(self):
        super().update_plot()
        self._env.update_plot()


## -------------------------------------------------------------------------------------------------
    def _set_mode(self, p_mode):
        self._env.set_mode(p_mode)


## -------------------------------------------------------------------------------------------------
    def get_latency(self) -> timedelta:
        return self._env.get_latency()


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed):
        """
        Environment and timer are reset. The random generators for environment and agent will
        also be reset. Optionally the agent's internal buffer data will be cleared but
        it's policy will not be touched.

        Parameters:
            p_seed                  New seed for environment's and agent's random generator
        """

        # Reset environment
        self._env.reset(p_seed)
        if self._visualize: self._env.init_plot()
            

## -------------------------------------------------------------------------------------------------
    def get_agent(self):
        return self._agent
      

## -------------------------------------------------------------------------------------------------
    def get_env(self):
        return self._env


## -------------------------------------------------------------------------------------------------
    def connect_data_logger(self, p_ds_states:RLDataStoring=None, p_ds_actions:RLDataStoring=None, p_ds_rewards:RLDataStoring=None):
        self._ds_states     = p_ds_states
        self._ds_actions    = p_ds_actions
        self._ds_rewards    = p_ds_rewards


## -------------------------------------------------------------------------------------------------
    def _run_cycle(self):
        """
        Processes a single control cycle.

        Returns:
            success         True, if environment goal has reached. False otherwise.
            error           True, if environment has broken. False otherwise.
        """

        # 1 Environment: get current state
        state = self._env.get_state()
        state.set_tstamp(self._timer.get_time())

        if self._ds_states is not None:
            self._ds_states.memorize_row(self._cycle_id, self._timer.get_time(), state.get_values())


        # 2 Agent: compute and log next action
        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': Agent computes action...')
        action  = self._agent.compute_action(state)
        ts      = self._timer.get_time()
        action.set_tstamp(ts)
        if self._ds_actions is not None:
            self._ds_actions.memorize_row(self._cycle_id, ts, action.get_sorted_values())


        # 3 Environment: process agent's action
        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': Env processes action...')
        self._env.process_action(action)
        self._timer.add_time(self._env.get_latency())     # in virtual mode only...
        self._env.get_state().set_tstamp(self._timer.get_time())


        # 4 Environment: compute and log reward
        reward  = self._env.compute_reward()
        ts      = self._timer.get_time()
        reward.set_tstamp(ts)
        if self._ds_rewards is not None:
            if ( reward.get_type() == Reward.C_TYPE_OVERALL ) or ( reward.get_type() == Reward.C_TYPE_EVERY_AGENT ):
                reward_values = np.zeros(self._ds_rewards.get_space().get_num_dim())

                for i, agent_id in enumerate(self._ds_rewards.get_space().get_dim_ids()): 
                    reward_values[i] = reward.get_agent_reward(agent_id)
                
                self._ds_rewards.memorize_row(self._cycle_id, ts, reward_values)


        # 5 Agent: adapt policy
        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': Agent adapts policy...')
        self._agent.adapt(self._env.get_state(), reward)


        # 6 Check for terminating events
        success = self._env.get_state().get_done()
        error   = self._env.get_state().get_broken()

        if success:
            self.log(self.C_LOG_TYPE_S, 'Process time', self._timer.get_time(), ': Environment goal achieved')

        if error:
            self.log(self.C_LOG_TYPE_E, 'Process time', self._timer.get_time(), ': Environment broken')

        return success, error





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLTrainingResults (TrainingResults):
    """
    Results of a RL training.
    """

    C_FNAME_TRAINING        = 'training'
    C_FNAME_ENV_STATES      = 'env_states'
    C_FNAME_AGENT_ACTIONS   = 'agent_actions'
    C_FNAME_ENV_REWARDS     = 'env_rewards'

    C_CPAR_NUM_EPI          = 'Number of episodes'
    C_CPAR_NUM_EVAL         = 'Number of evaluations'
    C_CPAR_NUM_ADAPT        = 'Number of adaptations'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_scenario: Scenario, p_run, p_cycle_id, p_path=None):
        super().__init__(p_scenario, p_run, p_cycle_id, p_path=p_path)

        self.num_episodes      = 0
        self.num_adaptations   = 0
        self.num_evaluations   = 0
        self.ds_states         = None
        self.ds_actions        = None
        self.ds_rewards        = None
        self.ds_training       = None


## -------------------------------------------------------------------------------------------------
    def close(self, p_cycle_id):
        super().close(p_cycle_id)

        self.add_custom_result(self.C_CPAR_NUM_EPI, self.num_episodes)
        self.add_custom_result(self.C_CPAR_NUM_EVAL, self.num_evaluations)
        self.add_custom_result(self.C_CPAR_NUM_ADAPT, self.num_adaptations)


## -------------------------------------------------------------------------------------------------
    def save(self, p_path, p_filename='summary.txt') -> bool:
        if not super().save(p_path, p_filename=p_filename): return False

        if self.ds_states is not None: self.ds_states.save_data(p_path, self.C_FNAME_ENV_STATES)
        if self.ds_actions is not None: self.ds_actions.save_data(p_path, self.C_FNAME_AGENT_ACTIONS)
        if self.ds_rewards is not None: self.ds_rewards.save_data(p_path, self.C_FNAME_ENV_REWARDS)
        if self.ds_training is not None: self.ds_training.save_data(p_path, self.C_FNAME_TRAINING)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLTraining (Training):
    """
    This class performs an episodical training on a (multi-)agent in a given environment. Both are 
    expected as parts of a reinforcement learning scenario (see class Scenario for more details).
    The class optionally collects all relevant data like environmenal states and rewards or agents
    actions. Furthermore overarching training data will be collected.

    The class provides the three methods run(), run_episode(), run_cycle() that can be called in 
    any order to proceed the training.

    https://github.com/fhswf/MLPro/issues/146
    """

    C_TYPE                  = 'Training'
    C_NAME                  = 'RL'

    C_CLS_RESULTS           = RLTrainingResults

## -------------------------------------------------------------------------------------------------
    def __init__(self, 
                 p_scenario:RLScenario,         # RL scenario object 
                 p_cycle_limit=0,               # Maximum number of training cycles (0=no limit)
                 p_max_cycles_per_episode=-1,   # Optional limit for cycles per episode
                                                # (0=no limit, -1=get environment limit)
                 p_max_adaptations=0,           # Optional limit for total number of adaptations
                 p_max_stagnations=5,           # Optional length of a sequence of evaluations without
                                                # training progress
                 p_eval_frequency=100,          # Optional evaluation frequency (0=no evaluation)
                 p_eval_grp_size=50,            # Number of evaluation episodes (eval group)
                 p_hpt:HyperParamTuner=None,    # Optional hyperparameter tuner (see class HyperParamTuner)
                 p_hpt_trials=0,                # Optional number of hyperparameter tuning trials
                 p_path=None,                   # Optional destination path to store training data
                 p_collect_states=True,         # If True, the environment states will be collected
                 p_collect_actions=True,        # If True, the agent actions will be collected
                 p_collect_rewards=True,        # If True, the environment reward will be collected
                 p_collect_training=True,       # If True, global training data will be collected
                 p_logging=Log.C_LOG_WE):       # Log level (see constants of class Log)

        if ( p_cycle_limit <= 0 ) and ( p_max_adaptations <= 0 ) and ( p_max_stagnations <= 0 ):
            raise ParamError('Please define a termination criterion')

        if ( p_max_stagnations > 0 ) and ( ( p_eval_frequency <= 0) or ( p_eval_grp_size <= 0 ) ):
            raise ParamError('Stagnation detection needs an eval freqency and eval group size > 0')

        if p_max_adaptations > 0: raise NotImplementedError
        if p_max_stagnations > 0: raise NotImplementedError

        super().__init__(p_scenario, p_cycle_limit=p_cycle_limit, p_hpt=p_hpt, p_hpt_trials=p_hpt_trials, p_path=p_path, p_logging=p_logging)
 
        self._env                   = self._scenario.get_env()
        self._agent                 = self._scenario.get_agent()

        self._eval_frequency        = p_eval_frequency
        self._eval_grp_size         = p_eval_grp_size
        self._max_adaptation        = p_max_adaptations

        if p_max_cycles_per_episode > 0:
            self._max_cycles_per_epi = p_max_cycles_per_episode
        else:
            self._max_cycles_per_epi = self._env.get_cycle_limit()

        self._collect_states        = p_collect_states
        self._collect_actions       = p_collect_actions
        self._collect_rewards       = p_collect_rewards
        self._collect_training      = p_collect_training

        self._cycles_episode        = 0
        self._eval_grp_id           = 0


## -------------------------------------------------------------------------------------------------
    def _init_results(self) -> TrainingResults:
        results = super()._init_results()

        if self._collect_states: 
            results.ds_states = RLDataStoring(self._env.get_state_space())

        if self._collect_actions: 
            results.ds_actions = RLDataStoring(self._env.get_action_space())

        if self._collect_rewards: 
            reward_type = self._env.get_reward_type()

            if ( reward_type == Reward.C_TYPE_OVERALL ) or ( reward_type == Reward.C_TYPE_EVERY_AGENT ):
                reward_space = Set()
                try:
                    agents = self._agent.get_agents()
                except:
                    agents = [ [self._agent, 1.0] ]

                for agent, weight in agents:
                    reward_space.add_dim(Dimension(agent.get_id(), agent.get_name()))

                if reward_space.get_num_dim() > 0:
                    results.ds_rewards  = RLDataStoring(reward_space)

        if self._collect_training:
            results.ds_training = RLDataStoring()

        self._scenario.connect_data_logger(p_ds_states=results.ds_states, p_ds_actions=results.ds_actions, p_ds_rewards=results.ds_rewards)

        return results


## -------------------------------------------------------------------------------------------------
    def _init_episode(self):
        self.log(self.C_LOG_TYPE_I, '--------------------------------------------------')
        self.log(self.C_LOG_TYPE_I, '-- Episode', self._results.num_episodes, 'started...')
        self.log(self.C_LOG_TYPE_I, '--------------------------------------------------\n')

        if self._results.ds_states is not None: self._results.ds_states.add_episode(self._results.num_episodes)
        if self._results.ds_actions is not None: self._results.ds_actions.add_episode(self._results.num_episodes)
        if self._results.ds_rewards is not None: self._results.ds_rewards.add_episode(self._results.num_episodes)

        self._scenario.reset(self._results.num_episodes + self._eval_grp_size)


## -------------------------------------------------------------------------------------------------
    def _close_episode(self):
        self.log(self.C_LOG_TYPE_I, '--------------------------------------------------')
        self.log(self.C_LOG_TYPE_I, '-- Episode', self._results.num_episodes, 'finished after', str(self._cycles_episode), 'cycles')
        self.log(self.C_LOG_TYPE_I, '--------------------------------------------------\n\n')

        self._cycles_episode         = 0
        self._results.num_episodes  += 1


## -------------------------------------------------------------------------------------------------
    def _progress_detected(self) -> bool:
        """
        Determines training progress after finishing a control group loop.

        Returns:
            True, if there was a progress. False otherwise.
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _run_cycle(self) -> bool:
        """
        Runs single training cycle.

        Returns:
            True, if training has finished. False otherwise.
        """

        # 0 Intro
        eof_episode     = False
        eof_training    = False


        # 1 Init next episode
        if self._cycles_episode == 0: 
            self._init_episode()


        # 2 Run a cycle
        success, error, timeout, limit = self._scenario.run_cycle()
        self._cycles_episode += 1


        # 3 Check: Episode finished?
        if success or error: 
            eof_episode = True

        elif ( self._max_cycles_per_epi > 0 ) and ( self._cycles_episode == self._max_cycles_per_epi ):
            self.log(self.C_LOG_TYPE_W, 'Episode cycle limit ', str(self._max_cycles_per_epi), ' reached')
            eof_episode = True

        if eof_episode: 
            self._close_episode()
            

        # 4 Check: Training finished?
        if ( self._max_adaptation > 0 ) and ( self._results.num_adaptations == self._max_adaptation ):
            self.log(self.C_LOG_TYPE_I, 'Adaptation limit ', str(self._max_adaptation), ' reached')
            eof_training = True


        # 5 Outro
        return eof_training