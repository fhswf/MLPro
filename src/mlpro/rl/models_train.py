## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
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
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.4 (2021-10-08)

This module provides model classes to define and run rl scenarios and to train agents inside them.
"""


from mlpro.bf.various import *
from mlpro.bf.math import *
from mlpro.bf.data import *
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
    C_VAR_NUM_CYLCLES   = 'Number of cycles'
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
            self.variables  = [ self.C_VAR_NUM_CYLCLES, self.C_VAR_ENV_DONE, self.C_VAR_ENV_BROKEN ]

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
class Scenario(Log, LoadSave):
    """
    Template class for an rl sceario consisting of an environment and an agent. Please
    implement method setup() to setup env and agent structure.
    """

    C_TYPE              = 'RL-Scenario'
    C_NAME              = '????'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_mode=Environment.C_MODE_SIM, p_ada=True, p_cycle_len:timedelta=None, 
                p_cycle_limit=0, p_visualize=True, p_logging=True):
        """
        Parameters:
            p_mode              Operation mode of environment (see Environment.C_MODE_*)
            p_ada               Boolean switch for adaptivity of agent
            p_cycle_len         Fixed cycle duration (optional)
            p_cycle_limit       Maximum number of cycles (0=no limit, -1=get limit from env)
            p_visualize         Boolean switch for env/agent visualisation
            p_logging           Boolean switch for logging functionality
        """

        # 0 Intro
        self._env           = None
        self._agent         = None
        self._cycle_len     = p_cycle_len
        self._cycle_limit   = p_cycle_limit
        self._visualize     = p_visualize
        Log.__init__(self, p_logging=p_logging)

        # 1 Setup entire scenario
        self._setup(p_mode, p_ada, p_logging)

        # 2 Finalize cycle limit
        if self._cycle_limit == -1:
            self._cycle_limit = self._env.get_cycle_limit()

        # 2 Init timer
        if self._env.get_mode() == Environment.C_MODE_SIM:
            t_mode = Timer.C_MODE_VIRTUAL
        else:
            t_mode = Timer.C_MODE_REAL

        if self._cycle_len is not None:
            t_lap_duration = p_cycle_len
        else:
            t_lap_duration = self._env.get_latency()

        self._timer  = Timer(t_mode, t_lap_duration, self._cycle_limit)


## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada:bool, p_logging:bool):
        """
        Here's the place to explicitely setup the entire rl scenario. Please bind your env to
        self._env and your agent to self._agent. 

        Parameters:
            p_mode              Operation mode of environment (see Environment.C_MODE_*)
            p_ada               Boolean switch for adaptivity of agent
            p_logging           Boolean switch for logging functionality
       """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def reset(self):
        """
        Environment and timer will be resetted. Agent's internal buffer data will be cleared but
        it's policy will not be touched.
        """

        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': Scenario reset...')
        self._env.reset()

        if self._visualize:
            self._env.init_plot()
            self._agent.init_plot()

        self._timer.reset()
        self._env.get_state().set_tstamp(self._timer.get_time())


## -------------------------------------------------------------------------------------------------
    def get_env(self):
        return self._env


## -------------------------------------------------------------------------------------------------
    def get_agent(self):
        return self._agent
      

## -------------------------------------------------------------------------------------------------
    def run_cycle(self, p_cycle_id, p_ds_states:RLDataStoring=None, p_ds_actions:RLDataStoring=None, 
                p_ds_rewards:RLDataStoring=None):
        """
        Processes a single control cycle with optional data logging.

        Parameters:
            p_cycle_id          Cycle id
            p_ds_states         Optional external data storing object that collects environment state data
            p_ds_actions        Optional external data storing object that collects agent action data
            p_ds_rewards        Optional external data storing object that collects environment reeward data
        """

        # 0 Cycle intro
        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': Start of cycle', str(p_cycle_id))


        # 1 Environment: get and log current state
        state   = self._env.get_state()
        if p_ds_states is not None:
            p_ds_states.memorize_row(p_cycle_id, self._timer.get_time(), state.get_values())


        # 2 Agent: compute and log next action
        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': Agent computes action...')
        action  = self._agent.compute_action(state)
        ts      = self._timer.get_time()
        action.set_tstamp(ts)
        if p_ds_actions is not None:
            p_ds_actions.memorize_row(p_cycle_id, ts, action.get_sorted_values())


        # 3 Environment: process agent's action
        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': Env processes action...')
        self._env.process_action(action)
        self._timer.add_time(self._env.get_latency())     # in virtual mode only...
        self._env.get_state().set_tstamp(self._timer.get_time())


        # 4 Environment: compute and log reward
        reward  = self._env.compute_reward()
        ts      = self._timer.get_time()
        reward.set_tstamp(ts)
        if p_ds_rewards is not None:
            if ( reward.get_type() == Reward.C_TYPE_OVERALL ) or ( reward.get_type() == Reward.C_TYPE_EVERY_AGENT ):
                reward_values = np.zeros(p_ds_rewards.get_space().get_num_dim())

                for i, agent_id in enumerate(p_ds_rewards.get_space().get_dim_ids()): 
                    reward_values[i] = reward.get_agent_reward(agent_id)
                
                p_ds_rewards.memorize_row(p_cycle_id, ts, reward_values)


        # 5 Agent: adapt policy
        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': Agent adapts policy...')
        self._agent.adapt(self._env.get_state(), reward)


        # 6 Optional visualization
        if self._visualize:
            self._env.update_plot()
            self._agent.update_plot()


        # 7 Wait for next cycle (virtual mode only)
        if ( self._timer.finish_lap() == False ) and ( self.cycle_len is not None ):
            self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': Process timed out !!!')


        # 8 Cycle outro
        self.log(self.C_LOG_TYPE_I, 'Process time', self._timer.get_time(), ': End of cycle', str(p_cycle_id), '\n')


## -------------------------------------------------------------------------------------------------
    def run(self, p_exit_when_broken=True, p_exit_when_done=True, p_ds_states:RLDataStoring=None, 
            p_ds_actions:RLDataStoring=None, p_ds_rewards:RLDataStoring=None):
        """
        Processes control cycles in a loop. Termination depends on parameters.

        Parameters:
            p_exit_when_broken      If True, loop terminates when environment has boken
            p_exit_when_done        If True, loop terminates when goal of environment was achieved
            p_ds_states             Optional external data storing object that collects environment state data
            p_ds_actions            Optional external data storing object that collects agent action data
            p_ds_rewards            Optional external data storing object that collects environment reeward data

        Returns:
            done                    True if environment reached it's goal
            broken                  True if environment has broken
            num_cycles              Number of cycles
        """
        
        # 1 Preparation
        done = False
        self.reset()


        # 2 Start run
        self.log(self.C_LOG_TYPE_I, 'Run started')
        cycle_id  = 1

        while True:
            # 2.1 Process one cycle
            self.run_cycle(cycle_id, p_ds_states=p_ds_states, p_ds_actions=p_ds_actions, p_ds_rewards=p_ds_rewards)

            # 2.2 Check and handle environment's health
            if self._env.get_broken(): 
                self.log(self.C_LOG_TYPE_E, 'Environment broken!')
                if p_exit_when_broken: 
                    break
                else:
                    self.log(self.C_LOG_TYPE_I, 'Reset environment')
                    self._env.reset()

            # 2.3 Check and handle environment's done state
            if self._env.get_done() != done:
                done = self._env.get_done()
                if done == True:
                    self.log(self.C_LOG_TYPE_I, 'Environment goal achieved')
                else:
                    self.log(self.C_LOG_TYPE_W, 'Environment goal missed')

            if p_exit_when_done and done: break

            # 2.4 Next cycle id
            if self._cycle_limit > 0: 
                if cycle_id < self._cycle_limit: 
                    cycle_id = cycle_id + 1
                else:
                    break


        # 3 Finish run
        self.log(self.C_LOG_TYPE_I, 'Stop')
        return self._env.get_done(), self._env.get_broken(), cycle_id





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Training(Log):
    """
    This class performs an episodical training on a (multi-)agent in a given environment. Both are 
    expected as parts of a reinforcement learning process (see class Process for more details).
    The class optionally collects all relevant data like environmenal states and rewards or agents
    actions. Furthermore overarching training data will be collected.

    The class provides the three methods run(), run_episode(), run_cycle() that can be called in 
    any order to proceed the training.
    """

    C_TYPE                  = 'Training'
    C_NAME                  = 'RL'

    C_FNAME_TRAINING        = 'training'
    C_FNAME_ENV_STATES      = 'env_states'
    C_FNAME_AGENT_ACTIONS   = 'agent_actions'
    C_FNAME_ENV_REWARDS     = 'env_rewards'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_scenario:Scenario, p_episode_limit=50, p_cycle_limit=0, p_collect_states=True, 
                p_collect_actions=True, p_collect_rewards=True, p_collect_training=True, p_logging=True):
        """
        Parmeters:
            p_scenario              RL scenario object
            p_episode_limit         Maximum number of episodes
            p_cycle_limit           Naximum number of cycles within an episode (a value > 0 overrides
                                    the cycle limit provided by the enviroment)
            p_collect_states        If True, the environment states will be collected
            p_collect_actions       If True, the agent actions will be collected
            p_collect_rewards       If True, the environment reward will be collected
            p_collect_training      If True, global training data will be collected
            p_logging               Boolean switch for logging
        """

        super().__init__(p_logging=p_logging)

        self._scenario      = p_scenario
        self._env           = self._scenario.get_env()
        self._agent         = self._scenario.get_agent()

        self._episode_id    = 0
        self._episode_limit = p_episode_limit
        self._cycle_id      = 0

        if p_cycle_limit > 0:
            self._cycle_limit = p_cycle_limit
        else:
            self._cycle_limit = self._env.get_cycle_limit()

        if self._cycle_limit <= 0:
            raise ParamError('Invalid cycle limit')
        else:
            self.log(self.C_LOG_TYPE_I, 'Limit of cycles per episide:', str(self._cycle_limit))

        if p_collect_states:
            self._ds_states   = RLDataStoring(self._env.get_state_space())
        else:
            self._ds_states   = None

        if p_collect_actions:
            self._ds_actions  = RLDataStoring(self._env.get_action_space())
        else:
            self._ds_actions  = None

        if p_collect_rewards:
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
                    self._ds_rewards  = RLDataStoring(reward_space)

            else:
                # Futher reward type not yet supported
                self._ds_rewards  = None

        else:
            self._ds_rewards  = None

        if p_collect_training:
            self._ds_training = RLDataStoring()
        else:
            self._ds_training = None


## -------------------------------------------------------------------------------------------------
    def run_cycle(self):
        """
        Runs next training cycle.
        """

        # 1 Begin of new episode? Reset agent and environment 
        if self._cycle_id == 0:
            self.log(self.C_LOG_TYPE_I, '--------------------------------------')
            self.log(self.C_LOG_TYPE_I, '-- Episode', self._episode_id, 'started...')
            self.log(self.C_LOG_TYPE_I, '--------------------------------------\n')
            self._scenario.reset()
 
            # 1.1 Init frame for next episode in data storage objects
            if self._ds_training is not None: self._ds_training.add_episode(self._episode_id)
            if self._ds_states is not None: self._ds_states.add_episode(self._episode_id)
            if self._ds_actions is not None: self._ds_actions.add_episode(self._episode_id)
            if self._ds_rewards is not None: self._ds_rewards.add_episode(self._episode_id)


        # 2 Run a cycle
        self._scenario.run_cycle(self._cycle_id, p_ds_states=self._ds_states, p_ds_actions=self._ds_actions, p_ds_rewards=self._ds_rewards)


        # 3 Update training counters
        if self._env.get_done() or self._env.get_broken() or ( self._cycle_id == (self._cycle_limit-1) ):
            # 3.1 Episode finished
            self.log(self.C_LOG_TYPE_I, '--------------------------------------')
            self.log(self.C_LOG_TYPE_I, '-- Episode', self._episode_id, 'finished after', self._cycle_id + 1, 'cycles')
            self.log(self.C_LOG_TYPE_I, '--------------------------------------\n\n')

            # 3.1.1 Update global training data storage
            if self._ds_training is not None:
                if self._env.get_done()==True:
                    done_num = 1
                else:
                    done_num = 0

                if self._env.get_broken()==True:
                    broken_num = 1
                else:
                    broken_num = 0

                self._ds_training.memorize(RLDataStoring.C_VAR_NUM_CYLCLES, self._episode_id, self._cycle_id + 1)
                self._ds_training.memorize(RLDataStoring.C_VAR_ENV_DONE, self._episode_id, done_num)
                self._ds_training.memorize(RLDataStoring.C_VAR_ENV_BROKEN, self._episode_id, broken_num)
 
            # 3.1.2 Prepare next episode
            self._episode_id   += 1
            self._cycle_id      = 0

        else:
            # 3.2 Prepare next cycle
            self._cycle_id     += 1


## -------------------------------------------------------------------------------------------------
    def run_episode(self):
        """
        Runs/finishes current training episode.
        """

        current_episode_id = self._episode_id
        while self._episode_id == current_episode_id: self.run_cycle()


## -------------------------------------------------------------------------------------------------
    def run(self):
        """
        Runs/finishes entire training.
        """

        while self._episode_id < self._episode_limit: self.run_episode()


## -------------------------------------------------------------------------------------------------
    def get_data(self):
        return self._ds_training, self._ds_states, self._ds_actions, self._ds_rewards


## -------------------------------------------------------------------------------------------------
    def save_data(self, p_path, p_delimiter):
        result      = True
        num_files   = 0

        if self._ds_training is not None:
            if self._ds_training.save_data(p_path, self.C_FNAME_TRAINING, p_delimiter):
                self.log(self.C_LOG_TYPE_I, 'Saved training data to file "' + self.C_FNAME_TRAINING 
                        + '" in "' + p_path + '"')
                num_files  += 1
                result      = result and True
            else:
                result      = False

        if self._ds_states is not None:
            if self._ds_states.save_data(p_path, self.C_FNAME_ENV_STATES, p_delimiter):
                self.log(self.C_LOG_TYPE_I, 'Saved environment state data to file "' + self.C_FNAME_ENV_STATES
                        + '" in "' + p_path + '"')
                num_files  += 1
                result      = result and True
            else:
                result      = False

        if self._ds_actions is not None:
            if self._ds_actions.save_data(p_path, self.C_FNAME_AGENT_ACTIONS, p_delimiter):
                self.log(self.C_LOG_TYPE_I, 'Saved agent action data to file "' + self.C_FNAME_AGENT_ACTIONS 
                        + '" in "' + p_path + '"')
                num_files  += 1
                result      = result and True
            else:
                result      = False

        if self._ds_rewards is not None:
            if self._ds_rewards.save_data(p_path, self.C_FNAME_ENV_REWARDS, p_delimiter):
                self.log(self.C_LOG_TYPE_I, 'Saved environment reward data to file "' + self.C_FNAME_ENV_REWARDS 
                        + '" in "' + p_path + '"')
                num_files  += 1
                result      = result and True
            else:
                result      = False

        if num_files > 0: return result
        return False






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class HPTuningRL(HyperParamTuning):
    """
    Hyperparameter tuning for reinforcement learning.
    """

    C_NAME              = 'RL'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_scenario:Scenario, p_path:str, p_episode_limit=50, p_cycle_limit=0, p_logging=True):
        super().__init__(p_path, p_logging=p_logging)


## -------------------------------------------------------------------------------------------------
    def optimize(self, *p_hp):
        # 1 Set hyperparameters
        # 2 Create and process a training
        # 3 Return overall number of cycles a the value to be mininmized

        return 0
