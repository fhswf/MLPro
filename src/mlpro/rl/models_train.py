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
## -- 2021-11-13  1.3.0     DA       Rework/improvement of class RLTraining
## -- 2021-12-07  1.3.1     DA       - Method RLScenario.__init__(): param p_cycle_len removed
## --                                - Method RLTraining.__init__(): par p_scenario replaced by p_scenario_cls
## -- 2021-12-09  1.3.2     DA       Class RLTraining: introduced dynamic parameters **p_kwargs
## -- 2021-12-12  1.4.0     DA       Class RLTraining: evaluation and stagnation detection added
## -- 2021-12-16  1.4.1     DA       Method RLTraining._close_evaluation(): optimized scoring
## -- 2021-12-21  1.5.0     DA       Class RLTraining: reworked evaluation strategy
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.5.0 (2021-12-21)

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
class RLDataStoring (DataStoring):
    """
    Derivate of basic class DataStoring that is specialized to store episodical training data in the
    context of reinforcement learning.
    """

    # Frame ID renamed
    C_VAR0              = 'Episode ID'

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
class RLDataStoringEval (DataStoring):
    """
    Derivate of basic class DataStoring that is specialized to store evaluation data of a training
    in the context of reinforcement learning.
    """

    # Frame ID renamed
    C_VAR0              = 'Evaluation ID'

    # Variables for training header data storage
    C_VAR_SCORE         = 'Score'
    C_VAR_NUM_CYCLES    = 'Number of cycles'
    C_VAR_NUM_SUCCESS   = 'Env goal reached'
    C_VAR_NUM_BROKEN    = 'Env broken'
    C_VAR_NUM_LIMIT     = 'Env out of limit'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_space:Set=None):
        """
        Parameters:
            p_space         Space object that provides dimensional information for raw data. If None
                            a training header data object will be instantiated.
        """
        
        self.space = p_space

        # Initalization as an episodical detail data storage
        self.variables  = [ self.C_VAR_SCORE, self.C_VAR_NUM_CYCLES, self.C_VAR_NUM_LIMIT, self.C_VAR_NUM_SUCCESS, self.C_VAR_NUM_BROKEN ]
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
    def add_evaluation(self, p_evaluation_id):
        self.add_frame(p_evaluation_id)
        self.current_evaluation = p_evaluation_id


## -------------------------------------------------------------------------------------------------
    def memorize_row(self, p_score, p_num_limit, p_num_cycles, p_num_success, p_num_broken, p_reward):
        """
        Memorizes an evaluation data row.

        Parameters
        ---------- 

        """

        self.memorize(self.C_VAR_SCORE, self.current_evaluation, p_score)
        self.memorize(self.C_VAR_NUM_LIMIT, self.current_evaluation, p_num_limit)
        self.memorize(self.C_VAR_NUM_CYCLES, self.current_evaluation, p_num_cycles)
        self.memorize(self.C_VAR_NUM_SUCCESS, self.current_evaluation, p_num_success)
        self.memorize(self.C_VAR_NUM_BROKEN, self.current_evaluation, p_num_broken)

        for i, var in enumerate(self.var_space):
            self.memorize(var, self.current_evaluation, p_reward[i])





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
                 p_cycle_limit=0,               # Maximum number of cycles (0=no limit, -1=get from env)
                 p_visualize=True,              # Boolean switch for env/agent visualisation
                 p_logging=Log.C_LOG_ALL ):     # Log level (see constants of class Log)

        # 1 Setup entire scenario
        self._env   = None
        super().__init__(p_mode=p_mode, p_ada=p_ada, p_cycle_limit=p_cycle_limit, p_visualize=p_visualize, p_logging=p_logging)
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
        Processes a single cycle.

        Returns
        -------
        success : bool
            True on success. False otherwise.
        error : bool
            True on error. False otherwise.
        adapted : bool
            True, if agent adapted something in this cycle. False otherwise.

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
        adapted = self._agent.adapt(self._env.get_state(), reward)


        # 6 Check for terminating events
        success = self._env.get_state().get_success()
        error   = self._env.get_state().get_terminal()

        if success:
            self.log(self.C_LOG_TYPE_S, 'Process time', self._timer.get_time(), ': Environment goal achieved')

        if error:
            self.log(self.C_LOG_TYPE_E, 'Process time', self._timer.get_time(), ': Environment terminated')

        return success, error, adapted





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLTrainingResults (TrainingResults):
    """
    Results of a RL training.

    Parameters
    ----------
    p_scenario : RLScenario
        Related reinforcement learning scenario.
    p_run : int
        Run id.
    p_cycle_id : int
        Id of first cycle of this run.
    p_path : str
        Optional destination path to store the results.
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL

    """

    C_NAME                  = 'RL'

    C_FNAME_EVAL            = 'evaluation'
    C_FNAME_ENV_STATES      = 'env_states'
    C_FNAME_AGENT_ACTIONS   = 'agent_actions'
    C_FNAME_ENV_REWARDS     = 'env_rewards'

    C_CPAR_NUM_EPI          = 'Training Episodes'
    C_CPAR_NUM_EVAL         = 'Evaluations'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_scenario:RLScenario, p_run, p_cycle_id, p_path=None, p_logging=Log.C_LOG_WE):
        super().__init__(p_scenario, p_run, p_cycle_id, p_path=p_path, p_logging=p_logging)

        self.num_episodes       = 0
        self.num_evaluations    = 0
        self.ds_states          = None
        self.ds_actions         = None
        self.ds_rewards         = None
        self.ds_eval            = None


## -------------------------------------------------------------------------------------------------
    def close(self):
        super().close()

        self.add_custom_result(self.C_CPAR_NUM_EPI, self.num_episodes)
        self.add_custom_result(self.C_CPAR_NUM_EVAL, self.num_evaluations)


## -------------------------------------------------------------------------------------------------
    def _log_results(self):
        super()._log_results()
        self.log(self.C_LOG_TYPE_W, '-- Training Episodes :', self.num_episodes)
        self.log(self.C_LOG_TYPE_W, '-- Evaluations       :', self.num_evaluations)


## -------------------------------------------------------------------------------------------------
    def save(self, p_path, p_filename='summary.csv') -> bool:
        if not super().save(p_path, p_filename=p_filename): return False

        if self.ds_states is not None: self.ds_states.save_data(p_path, self.C_FNAME_ENV_STATES)
        if self.ds_actions is not None: self.ds_actions.save_data(p_path, self.C_FNAME_AGENT_ACTIONS)
        if self.ds_rewards is not None: self.ds_rewards.save_data(p_path, self.C_FNAME_ENV_REWARDS)
        if self.ds_eval is not None: self.ds_eval.save_data(p_path, self.C_FNAME_EVAL)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLTraining (Training):
    """
    This class performs an episodical training on a (multi-)agent in a given environment. Both are 
    expected as parts of a reinforcement learning scenario (see class RLScenario for more details).
    The class optionally collects all relevant data like environmenal states and rewards or agents
    actions. Furthermore overarching training data will be collected.

    Parameters
    ----------
    p_scenario_cls 
        Name of RL scenario class, compatible to/inherited from class RLScenario.
    p_cycle_limit : int
        Maximum number of training cycles (0=no limit). Default = 0.
    p_cycles_per_epi_limit : int
        Optional limit of cycles per episode (0=no limit, -1=get environment limit). Default = -1.    
    p_adaptation_limit : int
        Maximum number of adaptations (0=no limit). Default = 0.
    p_stagnation_limit : int
        Optional limit of consecutive evaluations without training progress. Default = 0.
    p_eval_frequency : int
        Optional evaluation frequency (0=no evaluation). Default = 0.
    p_eval_grp_size : int
        Number of evaluation episodes (eval group). Default = 0.
    p_hpt : HyperParamTuner
        Optional hyperparameter tuner (see class mlpro.bf.ml.HyperParamTuner). Default = None.
    p_hpt_trials : int
        Optional number of hyperparameter tuning trials. Default = 0. Must be > 0 if p_hpt is supplied.
    p_path : str
        Optional destination path to store training data. Default = None.
    p_collect_states : bool
        If True, the environment states will be collected. Default = True.
    p_collect_actions : bool
        If True, the agent actions will be collected. Default = True.
    p_collect_rewards : bool
        If True, the environment reward will be collected. Default = True.
    p_collect_eval : bool
        If True, global evaluation data will be collected. Default = True.
    p_visualize : bool
        Boolean switch for env/agent visualisation. Default = False.
    p_logging
        Log level (see constants of class mlpro.bf.various.Log). Default = Log.C_LOG_WE.

    """

    C_NAME                  = 'RL'

    C_CLS_RESULTS           = RLTrainingResults

## -------------------------------------------------------------------------------------------------
    def __init__(self, **p_kwargs):

        # 1 Initialization of elementary training functionalities
        super().__init__( **p_kwargs )


        # 2 Check and completion of RL-specific parameters

        # 2.1 Optional parameter p_cycles_per_epi_limit
        try:
            self._cycles_per_epi_limit = self._kwargs['p_cycles_per_epi_limit']
        except:
            self._cycles_per_epi_limit = -1

        # 2.2 Optional parameter p_stagnation_limit
        try:
            self._stagnation_limit = self._kwargs['p_stagnation_limit']
        except:
            self._stagnation_limit = 0
            self._kwargs['p_stagnation_limit'] = self._stagnation_limit

        # 2.3 Optional parameter p_eval_frequency
        try:
            self._eval_frequency = self._kwargs['p_eval_frequency']
        except:
            self._eval_frequency = 0
            self._kwargs['p_eval_frequency'] = self._eval_frequency

        # 2.4 Optional parameter p_eval_grp_size
        try:
            self._eval_grp_size = self._kwargs['p_eval_grp_size']
        except:
            self._eval_grp_size = 0
            self._kwargs['p_eval_grp_size'] = self._eval_grp_size

        # 2.5 Optional parameter p_collect_states
        try:
            self._collect_states = self._kwargs['p_collect_states']
        except:
            self._collect_states = True
            self._kwargs['p_collect_states'] = self._collect_states

        # 2.6 Optional parameter p_collect_actions
        try:
            self._collect_actions = self._kwargs['p_collect_actions']
        except:
            self._collect_actions = True
            self._kwargs['p_collect_actions'] = self._collect_actions

        # 2.7 Optional parameter p_collect_rewards
        try:
            self._collect_rewards = self._kwargs['p_collect_rewards']
        except:
            self._collect_rewards = True
            self._kwargs['p_collect_rewards'] = self._collect_rewards

        # 2.8 Optional parameter p_collect_eval
        try:
            self._collect_eval = self._kwargs['p_collect_eval']
        except:
            self._collect_eval = True
            self._kwargs['p_collect_eval'] = self._collect_eval


        # 3 Check for further restrictions
        if ( self._cycle_limit <= 0 ) and ( self._adaptation_limit <= 0 ) and ( self._stagnation_limit <= 0 ):
            raise ParamError('Please define at least one termination criterion (p_cycle_limit, p_adaptation_limit, p_stagnation_limit')

        if ( self._stagnation_limit > 0 ) and ( ( self._eval_frequency <= 0) or ( self._eval_grp_size <= 0 ) ):
            raise ParamError('For stagnation detection both parameters p_eval_frequency and p_eval_grp_size must be > 0 as well')

        if ( ( self._eval_frequency > 0 ) and ( self._eval_grp_size <= 0 ) ) or ( ( self._eval_frequency <= 0 ) and ( self._eval_grp_size > 0) ):
            raise ParamError('For cyclic evaluation both parameters p_eval_frequency and p_eval_grp_size must be > 0') 

 
        # 4 Initialization of further rl-specific attributes
        if self._scenario is not None:
            self._mode                  = self.C_MODE_TRAIN
            self._cycles_episode        = 0
            self._eval_grp_id           = 0
            self._eval_last_score       = None
            self._eval_stagnations      = 0

            self._env   = self._scenario.get_env()
            self._agent = self._scenario.get_agent()

            if self._cycles_per_epi_limit == -1:
                self._cycles_per_epi_limit = self._env.get_cycle_limit()

            if self._cycles_per_epi_limit <= 0:
                raise ParamError('Please define a maximum number of training cylces per episode (env or param p_cycles_per_epi_limit')

            if self._eval_frequency > 0:
                # Training with evaluation starts with initial evaluation
                self._counter_epi_train     = 0
                self._counter_epi_eval      = 0
                self._mode                  = self.C_MODE_EVAL


## -------------------------------------------------------------------------------------------------
    def _init_results(self) -> TrainingResults:
        results = super()._init_results()

        if self._collect_states: 
            results.ds_states = RLDataStoring(self._env.get_state_space())

        if self._collect_actions: 
            results.ds_actions = RLDataStoring(self._env.get_action_space())

        if self._collect_rewards or self._collect_eval: 
            reward_type = self._env.get_reward_type()

            if ( reward_type == Reward.C_TYPE_OVERALL ) or ( reward_type == Reward.C_TYPE_EVERY_AGENT ):
                reward_space = Set()
                try:
                    agents = self._agent.get_agents()
                except:
                    agents = [ [self._agent, 1.0] ]

                for agent, weight in agents:
                    reward_space.add_dim(Dimension(agent.get_id(), agent.get_name()))

                if self._collect_rewards:
                    results.ds_rewards  = RLDataStoring(reward_space)

                if self._collect_eval:
                    results.ds_eval = RLDataStoringEval(reward_space)

        self._scenario.connect_data_logger(p_ds_states=results.ds_states, p_ds_actions=results.ds_actions, p_ds_rewards=results.ds_rewards)

        return results


## -------------------------------------------------------------------------------------------------
    def _init_episode(self):

        # 1 Evaluation handling  
        if self._eval_frequency > 0:

            if self._mode == self.C_MODE_TRAIN:

                if self._counter_epi_train == 0:
                    self._scenario.get_model().switch_adaptivity(True)

                    self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR)
                    self.log(self.C_LOG_TYPE_W, '-- Training period started...')
                    self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR, '\n')

                self._scenario.reset(self._results.num_episodes + self._eval_grp_size)

                self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR)
                self.log(self.C_LOG_TYPE_W, '-- Training episode', self._results.num_episodes, 'started...')
                self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR, '\n')

            else:
                if self._counter_epi_eval == 0:
                    self._scenario.get_model().switch_adaptivity(False)

                    self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR)
                    self.log(self.C_LOG_TYPE_W, '-- Evaluation period', self._results.num_evaluations, 'started...')
                    self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR, '\n')

                    self._init_evaluation()
                    
                self._scenario.reset(self._counter_epi_eval)

                self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR)
                self.log(self.C_LOG_TYPE_W, '-- Evaluation episode', self._counter_epi_eval, 'started...')
                self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR, '\n')

        else:
            self._scenario.reset(self._results.num_episodes + self._eval_grp_size)

            self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR)
            self.log(self.C_LOG_TYPE_W, '-- Training episode', self._results.num_episodes, 'started...')
            self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR, '\n')


        # 3 Preparation of data logging for next episode 
        if self._results.ds_states is not None: self._results.ds_states.add_episode(self._results.num_episodes)
        if self._results.ds_actions is not None: self._results.ds_actions.add_episode(self._results.num_episodes)
        if self._results.ds_rewards is not None: self._results.ds_rewards.add_episode(self._results.num_episodes)


## -------------------------------------------------------------------------------------------------
    def _close_episode(self):

        if self._eval_frequency > 0:

            if self._mode == self.C_MODE_TRAIN:
                self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR)
                self.log(self.C_LOG_TYPE_W, '-- Training episode', self._results.num_episodes, 'finished after', str(self._cycles_episode), 'cycles')
                self.log(self.C_LOG_TYPE_W, '-- Training cycles finished:', self._results.num_cycles + 1)
                self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR, '\n\n')

                self._results.num_episodes  += 1
                self._counter_epi_train     += 1

                if self._counter_epi_train >= self._eval_frequency:
                    self._counter_epi_eval  = 0
                    self._mode              = self.C_MODE_EVAL

            else:
                self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR)
                self.log(self.C_LOG_TYPE_W, '-- Evaluation episode', self._counter_epi_eval, 'finished after', str(self._cycles_episode), 'cycles')
                self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR, '\n\n')

                self._counter_epi_eval      += 1

                if self._counter_epi_eval >= self._eval_grp_size:

                    score = self._close_evaluation()
                    if ( self._results.highscore is None ) or ( score > self._results.highscore ):
                        self.log(self.C_LOG_TYPE_W, 'New temporal highscore', str(score))
                        self._results.highscore = score
                    else:
                        self.log(self.C_LOG_TYPE_W, 'New score', str(score))


                    self._results.num_evaluations += 1
                    self._counter_epi_train = 0
                    self._mode              = self.C_MODE_TRAIN
       
        else:
            self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR)
            self.log(self.C_LOG_TYPE_W, '-- Training episode', self._results.num_episodes, 'finished after', str(self._cycles_episode), 'cycles')
            self.log(self.C_LOG_TYPE_W, '-- Training cycles finished:', self._results.num_cycles + 1)
            self.log(self.C_LOG_TYPE_W, Training.C_LOG_SEPARATOR, '\n\n')

            self._results.num_episodes  += 1

        self._cycles_episode = 0


## -------------------------------------------------------------------------------------------------
    def _init_evaluation(self):
        """
        Initializes the next evaluation.
        """

        if self._results.ds_eval is not None: self._results.ds_eval.add_evaluation(self._results.num_evaluations)

        self._eval_num_cycles   = 0
        self._eval_num_limit    = 0 
        self._eval_num_success  = 0
        self._eval_num_broken   = 0
        self._eval_sum_reward   = None


## -------------------------------------------------------------------------------------------------
    def _update_evaluation(self, p_success:bool, p_error:bool, p_cycle_limit:bool):
        """
        Updates evaluation statistics.

        Parameters
        ----------
        p_success : bool
            True on success. False otherwise.
        p_error : bool
            True on error. False otherwise.
        p_cycle_limit : bool
            True, if cycle limit has reached. False otherwise.

        """

        self._eval_num_cycles       += 1
        if p_cycle_limit: self._eval_num_limit += 1
        if p_success: self._eval_num_success += 1
        if p_error: self._eval_num_broken += 1

        reward = self._env.get_last_reward()
        if reward is None: return

        reward_type = reward.get_type()

        if reward_type == Reward.C_TYPE_OVERALL:
            if self._eval_sum_reward is None:
                self._eval_sum_reward = np.zeros(1)

            self._eval_sum_reward[0] += reward.get_overall_reward()

        elif reward_type == Reward.C_TYPE_EVERY_AGENT:
            if self._eval_sum_reward is None:
                self._eval_sum_reward = np.zeros(len(reward.agent_ids))

            for i, agent_id in enumerate(reward.agent_ids): 
                # Get weight of agent
                try:
                    weight_agent = self._agent.get_agent(agent_id)[1]
                except:
                    weight_agent = 1.0

                self._eval_sum_reward[i] += reward.get_agent_reward(agent_id) * weight_agent
                
        else:
            raise Error('Reward type ' + str(reward_type) + ' not yet supported')


## -------------------------------------------------------------------------------------------------
    def _close_evaluation(self) -> float:
        """
        Closes the current evaluation and computes a related score.

        Returns
        -------
        score : float
            Score of current evalation.

        """

        # 1 Computation of score
        self._eval_sum_reward /= self._counter_epi_eval
        score = np.mean( self._eval_sum_reward )


        # 2 Store evaluation statistics
        if self._results.ds_eval is not None:

            self._results.ds_eval.memorize_row( p_score=score, 
                                                p_num_limit=self._eval_num_limit, 
                                                p_num_cycles=self._eval_num_cycles, 
                                                p_num_success=self._eval_num_success, 
                                                p_num_broken=self._eval_num_broken, 
                                                p_reward=self._eval_sum_reward )


        # 3 Stagnation detection
        if self._eval_last_score is None:
            self._eval_last_score = score

        elif score <= self._eval_last_score:
            self._eval_stagnations += 1
        
        else:
            self._eval_stagnations = 0


        # 4 Outro
        return score


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
        success, error, timeout, limit, adapted = self._scenario.run_cycle()
        self._cycles_episode += 1

        if adapted: 
            self._results.num_adaptations += 1


        # 3 Update current evaluation
        if self._mode == self.C_MODE_EVAL:
            self._update_evaluation(success, error, limit)


        # 4 Check: Episode finished?
        state       = self._env.get_state()
        eof_episode = state.get_terminal()

        if eof_episode:

            if state.get_broken():
                # 4.1 Environment reached a state of no return
                self.log(self.C_LOG_TYPE_E, 'Environment broken')

            elif state.get_success():
                # 4.2 Objective of environment reached
                self.log(self.C_LOG_TYPE_W, 'Objective of environment reached')

            elif state.get_timeout():
                # 4.3 Cycle limit of environment reached
                self.log(self.C_LOG_TYPE_W, 'Limit of', self._env.get_cycle_limit(), 'cylces per episde reached (Environment)')

            else:
                # 4.4 Environment terminated the episode because of unknown reasons
                self.log(self.C_LOG_TYPE_W, 'Environment terminated episode (reason unknown)')

        elif self._cycles_episode == self._cycles_per_epi_limit:
            # 4.5 Cycle limit of training setup reached
            self.log(self.C_LOG_TYPE_W, 'Limit of', self._cycles_per_epi_limit, 'cylces per episde reached (Training)')
            eof_episode = True

        if eof_episode: 
            self._close_episode()
            

        # 5 Check: Training finished?
        if ( self._adaptation_limit > 0 ) and ( self._results.num_adaptations == self._adaptation_limit ):
            self.log(self.C_LOG_TYPE_W, 'Adaptation limit ', str(self._adaptation_limit), ' reached')
            eof_training = True

        if ( self._stagnation_limit > 0 ) and ( self._eval_stagnations >= self._stagnation_limit ):
            self.log(self.C_LOG_TYPE_W, 'Stagnation limit ', str(self._stagnation_limit), ' reached')
            eof_training = True


        # 6 Outro
        return eof_training