## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.rl
## -- Module  : models_agents.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-04-18  0.0.0     DA       Creation
## -- 2021-06-06  1.0.0     DA       Release of first version
## -- 2021-06-13  1.0.1     DA       New method Agent.reset_episode() and call in class Training;
## --                                Some bugfixes in class MultiAgent
## -- 2021-06-25  1.0.2     DA       New methods Agent.get_name(), Agent.set_name()
## -- 2021-07-01  1.0.3     DA       Bugfixes in classes MultiAgent
## -- 2021-08-26  1.1.0     DA       New class Policy; enhancements of Agent class;
## --                                Incompatible changes on classes Agent, MultiAgent
## -- 2021-08-28  1.1.1     DA       Bugfixes and minor improvements
## -- 2021-09-11  1.1.2     MRD      Change Header information to match our new library name
## -- 2021-09-18  1.2.0     MRD      Implement new SARBuffer class and SARBufferelement. The buffer
## --                                is now working in dictionary. Now the SARBuffer is inside
## --                                the Policy and EnvModel instead of the Agent.
## --                                Added "Done" as default input for Agent.adapt()
## -- 2021-10-05  1.2.1     DA       Various changes:
## --                                - New class ActionPlanner
## --                                - Class Agent: preparation for model-based mode
## -- 2021-10-05  1.2.2     SY       Bugfixes and minor improvements
## -- 2021-10-18  1.2.3     DA       Refactoring Policy/Agent/MultiAgent: state space renamed to 
## --                                observation space
## -- 2021-11-14  1.3.0     DA       Model-based Agent functionality 
## -- 2021-11-26  1.3.1     DA       Minor changes
## -- 2021-12-17  1.3.2     DA       Added method MultiAgent.get_agent()
## -- 2021-12-30  1.4.0     DA       - Class Agent: added internal model-based policy training
## --                                - Class ActionPlanner completed
## --                                - Standardization of all docstrings
## -- 2022-01-01  1.4.1     MRD      Refactoring and Fixing some bugs
## -- 2022-01-28  1.4.2     SY       - Added switch_adaptivity method in MultiAgent class
## --                                - Update _adapt method in MultiAgent class
## -- 2022-02-17  1.5.0     DA/SY    Class Agent: redefinition of method _init_hyperparam()
## -- 2022-02-24  1.5.1     SY       Class MultiAgent: redefinition of method _init_hyperparam()
## -- 2022-02-27  1.5.2     SY       Refactoring due to auto generated ID in class Dimension
## -- 2022-03-02  1.5.3     SY       Class MultiAgent: remove init_hyperparam(), update add_agent()
## -- 2022-03-02  1.5.4     DA       Reformatting
## -- 2022-03-07  1.5.5     SY       Minor Improvement on Class MultiAgent
## -- 2022-08-09  1.5.6     SY       Add MPC to ActionPlanner as a default algorithm
## -- 2022-08-15  1.5.7     SY       - Renaming maturity to accuracy
## --                                - Move MPC implementation to the pool of objects
## --                                - Update compute_action in Agent for action planning
## -- 2022-09-26  1.5.8     SY       Minor Improvement on _extract_observation method (Agent class)
## -- 2022-11-02  1.6.0     DA       Refactoring: methods adapt(), _adapt()
## -- 2022-11-07  1.6.1     DA       Classes Policy, Agent, MultiAgent: new parameter p_visualize
## -- 2022-11-29  1.6.2     DA       Refactoring
## -- 2022-12-09  1.6.3     DA       Refactoring
## -- 2023-01-02  1.6.4     SY       Add multi-processing on MultiAgent
## -- 2023-02-04  1.6.5     SY       Temporarily remove multi-processing on MultiAgent
## -- 2023-02-21  1.6.6     DA       Class MultiAgent: removed methods load(), save()
## -- 2023-03-10  1.6.7     SY       Class Agent and RLScenarioMBInt : update logging
## -- 2023-03-27  1.7.0     DA       Refactoring of persistence
## -- 2025-04-24  1.7.1     DA       Bugfix in Policy.__init__(): param p_name
## -- 2025-07-17  1.8.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.8.0 (2025-07-17) 

This module provides model classes for policies, model-free and model-based agents and multi-agents.
"""

from mlpro.bf import Log, ParamError
from mlpro.bf.plot import Figure
from mlpro.bf.math import MSpace
from mlpro.bf.systems import State, Action 
from mlpro.bf.ml import Model, HyperParamDispatcher
from mlpro.rl.models_env import *
from mlpro.rl.models_env_ada import *
from mlpro.rl.models_train import RLScenario, RLTraining



# Export list for public API
__all__ = [ 'Policy',
            'ActionPlanner',           
            'Agent',
            'MultiAgent' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Policy (Model):
    """
    This class represents the policy of a single-agent. It is adaptive and can be trained with
    State-Action-Reward (SAR) data that will be expected as a SAR buffer object. 
    The three main learning paradigms of machine learning to train a policy are supported:

    a) Training by Supervised Learning
    The entire SAR data set inside the SAR buffer shall be adapted.

    b) Training by Reinforcement Learning
    The latest SAR data record inside the SAR buffer shall be adapted.

    c) Training by Unsupervised Learning
    All state data inside the SAR buffer shall be adapted.

    Furthermore, a policy class can compute actions from states.

    Hyperparameters of the policy should be stored in the internal object self._hp_list, so that
    they can be tuned from outside. Optionally a policy-specific callback method can be called on 
    changes. For more information see class HyperParameterList.

    Parameters
    ----------
    p_observation_space : MSpace     
        Subspace of an environment that is observed by the policy.
    p_action_space : MSpace
        Action space object.
    p_id
        Optional external id
    p_buffer_size : int           
        Size of internal buffer. Default = 1.
    p_ada : bool               
        Boolean switch for adaptivity. Default = True.
    p_visualize : bool
        Boolean switch for env/agent visualisation. Default = False.
    p_logging
        Log level (see constants of class Log). Default = Log.C_LOG_ALL.
    """

    C_TYPE          = 'Policy'
    C_NAME          = '????'
    C_BUFFER_CLS    = SARSBuffer

## -------------------------------------------------------------------------------------------------
    def __init__( self,
                  p_observation_space : MSpace,
                  p_action_space : MSpace,
                  p_id = None,
                  p_buffer_size : int = 1,
                  p_ada : bool = True,
                  p_visualize : bool = False,
                  p_logging = Log.C_LOG_ALL ):
                 
        Model.__init__( self, 
                        p_id = p_id,
                        p_name = self.get_name(),
                        p_buffer_size = p_buffer_size, 
                        p_ada = p_ada,  
                        p_visualize = p_visualize, 
                        p_logging = p_logging )

        self._observation_space = p_observation_space
        self._action_space      = p_action_space


## -------------------------------------------------------------------------------------------------
    def get_observation_space(self) -> MSpace:
        return self._observation_space


## -------------------------------------------------------------------------------------------------
    def get_action_space(self) -> MSpace:
        return self._action_space


## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_obs: State) -> Action:
        """
        Specific action computation method to be redefined. 

        Parameters
        ----------
        p_obs : State
            Observation data.

        Returns
        -------
        action : Action
            Action object.

        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_sars_elem:SARSElement) -> bool:
        """
        Adapts the policy based on State-Action-Reward-State (SARS) data.

        Parameters
        ----------
        p_sars_elem : SARSElement
            Object of type SARSElement.

        Returns
        -------
        adapted : bool
            True, if something has been adapted. False otherwise.

        """

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ActionPlanner (Log):
    """
    Template class for action planning algorithms to be used as part of model-based planning agents. 
    The goal is to find the shortest sequence of actions that leads to a maximum reward.

    Parameters
    ----------
    p_state_thsld : float
        Threshold for metric difference between two states to be equal. Default = 0.00000001.
    p_logging
        Log level (see constants of class Log). Default = Log.C_LOG_ALL.

    """

    C_TYPE = 'Action Planner'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_state_thsld=0.00000001, p_logging=Log.C_LOG_ALL):
        super().__init__(p_logging=p_logging)
        self._control_horizon = 0
        self._prediction_horizon = 0
        self._width_limit = 0
        self._policy = None
        self._env_model = None
        self._action_path = None
        self._state_thsld = p_state_thsld


## -------------------------------------------------------------------------------------------------
    def setup(self,
              p_policy: Policy,
              p_envmodel: EnvModel,
              p_prediction_horizon=0,
              p_control_horizon=0,
              p_width_limit=0):
        """
        Setup of action planner object in concrete planning scenario. Must be called before first
        planning. Optional custom method _setup() is called at the end.

        Parameters
        ----------
        p_policy : Policy
            Policy of an agent.
        p_envmodel : EnvModel
            Environment model.
        p_prediction_horizon : int             
            Optional static maximum planning depth (=length of action path to be predicted). Can
            be overridden by method compute_action(). Default=0. 
        p_control_horizon : int             
            The length of predicted action path to be applied. Can be overridden by method
            compute_action(). Default=0.
        p_width_limit : int
            Optional static maximum planning width (=number of alternative actions per planning level).
            Can be overridden by method compute_action(). Default=0. 

        """

        self._policy = p_policy
        self._envmodel = p_envmodel
        self._prediction_horizon = p_prediction_horizon
        self._control_horizon = p_control_horizon
        self._width_limit = p_width_limit
        self._path_id = 0
        self.clear_action_path()
        self._action_path = None
        self._setup()


## -------------------------------------------------------------------------------------------------
    def _setup(self):
        """
        Optional custom setup method.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def compute_action(self,
                       p_obs: State,
                       p_prediction_horizon=0,
                       p_control_horizon=0,
                       p_width_limit=0) -> Action:
        """
        Computes a path of actions with defined length that maximizes the reward of the given 
        environment model. The planning algorithm itself is to be implemented in the custom method
        _plan_action().
        
        Parameters
        ----------
        p_obs : State
            Observation data.
        p_prediction_horizon : int             
            Optional dynamic maximum planning depth (=length of action path to be predicted) that 
            overrides the static limit of method setup(). Default=0 (no override).
        p_control_horizon : int             
            The length of predicted action path to be applied that overrides the static limit of
            method setup(). Default=0 (no override).
        p_width_limit : int
            Optional dynamic maximum planning width (=number of alternative actions per planning level)
            that overrides the static limit of method setup(). Default=0 (no override).

        Returns
        -------
        action : Action
            Best action as result of the planning process.

        """

        if (self._policy is None) or (self._envmodel is None):
            raise RuntimeError('Please call method setup() first')

        if (p_prediction_horizon > 0) and (p_prediction_horizon != self._prediction_horizon):
            self._prediction_horizon = p_prediction_horizon
            self._action_path = None
            
        if (p_control_horizon > 0) and (p_control_horizon != self._control_horizon):
            self._control_horizon = p_control_horizon

        if p_width_limit > 0:
            self._width_limit = p_width_limit

        if (self._prediction_horizon <= 0) or (self._width_limit <= 0) or (self._control_horizon <= 0):
            raise RuntimeError('Please set planning width, prediction horizon, and control horizon.')

        if self._control_horizon > self._prediction_horizon:
            raise RuntimeError('The control horizon must be at least 1 and less than or equal to prediction horizon.')

        # Check: Re-planning required?
        replan = self._action_path is None
        replan = replan or (self._path_id >= self._control_horizon)

        if not replan:
            # Check: Is the next action of action path suitable?
            path_data = self._action_path.get_all()
            obs_buffered = path_data['state'][self._path_id]
            replan = self._policy.get_observation_space().distance(p_obs, obs_buffered) > self._state_thsld

        if replan:
            # (Re-)Planning of action path
            self._path_id = 0
            if self._action_path is not None:
                self._action_path.clear()
            self._action_path = self._plan_action(p_obs)
            if (self._action_path is None) or (len(self._action_path) == 0):
                # Planning returned nothing -> direct action computation as fallback solution
                return self._policy.compute_action(p_obs)

        # Next action of action path can be used
        path_data = self._action_path.get_all()
        action = path_data['action'][self._path_id]
        self._path_id += 1
        return action


## -------------------------------------------------------------------------------------------------
    def _plan_action(self, p_obs: State) -> SARSBuffer:
        """
        Custom planning algorithm to fill the internal action path (self._action_path). Search width
        and depth are restricted by the attributes self._width_limit and self._prediction_horizon.

        Parameters
        ----------
        p_obs : State
            Observation data.

        Returns
        -------
        action_path : SARSBuffer
            Sequence of SARSElement objects with included actions that lead to the best possible reward.

        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def clear_action_path(self):
        if self._action_path is not None:
            self._action_path.clear()





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLScenarioMBInt(RLScenario):
    """
    Internal use in class Agent. Intended for the training of the policy with the environment model of
    a model-based (single) agent.

    """

    C_NAME = 'MB(intern)'

## -------------------------------------------------------------------------------------------------
    def _setup(self, **p_kwargs) -> Model:
        # Pseudo-implementation
        self._env = EnvBase(p_logging=Log.C_LOG_NOTHING)
        return Model(p_logging=Log.C_LOG_NOTHING)


## -------------------------------------------------------------------------------------------------
    def setup_ext(self, p_env: EnvBase, p_policy: Policy, p_logging: Log):
        self._model = Agent(p_policy=p_policy, p_logging=p_logging)
        self._agent = self._model
        self._env = p_env





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Agent (Policy):
    """
    This class represents a single agent model.

    Parameters
    ----------
    p_policy : Policy
        Policy object.
    p_envmodel : EnvModel
        Optional environment model object. Default = None.
    p_em_acc_thsld : float
        Optional threshold for environment model accuracy (whether the envmodel is 'good'
        enough to be used to train the policy). Default = 0.9.
    p_action_planner : ActionPlanner   
        Optional action planner object (obligatory for model based agents). Default = None.
    p_predicting_horizon : int    
        Optional predicting horizon (obligatory for model based agents). Default = 0.
    p_controlling_horizon : int    
        Optional controlling (obligatory for model based agents). Default = 0.
    p_planning_width : int   
        Optional planning width (obligatory for model based agents). Default = 0.
    p_name : str             
        Optional name of agent. Default = ''.
    p_ada : bool               
        Boolean switch for adaptivity. Default = True.
    p_visualize : bool
        Boolean switch for env/agent visualisation. Default = False.
    p_logging          
        Log level (see constants of class mlpro.bf.various.Log). Default = Log.C_LOG_ALL.
    p_mb_training_param : dict
        Optional parameters for internal policy training with environment model (see parameters of
        class RLTraining). Hyperparameter tuning and data logging is not supported here. The suitable
        scenario class is added internally.
    """

    C_TYPE = 'Agent'
    C_NAME = ''

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_policy: Policy,
                 p_envmodel: EnvModel = None,
                 p_em_acc_thsld=0.9,
                 p_action_planner: ActionPlanner = None,
                 p_predicting_horizon=0,
                 p_controlling_horizon=0,
                 p_planning_width=0,
                 p_name='',
                 p_ada=True,
                 p_visualize:bool=True,
                 p_logging=Log.C_LOG_ALL,
                 **p_mb_training_param):

        if p_name != '':
            self.set_name(p_name)
        else:
            self.set_name(self.C_NAME)

        if p_envmodel is not None:
            if len(p_mb_training_param) == 0:
                raise ParamError('Please provide parameters for model-based training in parameter p_mb_training_param')

            self._mb_training_param = p_mb_training_param.copy()
            self._mb_training_param['p_scenario_cls'] = RLScenarioMBInt
            self._mb_training_param['p_visualize'] = False
            self._mb_training_param['p_logging'] = p_logging
            if 'p_collect_states' not in self._mb_training_param:
                self._mb_training_param['p_collect_states'] = False
            if 'p_collect_actions' not in self._mb_training_param:
                self._mb_training_param['p_collect_actions'] = False
            if 'p_collect_rewards' not in self._mb_training_param:
                self._mb_training_param['p_collect_rewards'] = False
            if 'p_collect_eval' not in self._mb_training_param:
                self._mb_training_param['p_collect_eval'] = False

            # Hyperparameter tuning is disabled here
            if 'p_hpt' in self._mb_training_param:
                self._mb_training_param.pop('p_hpt')
            if 'p_hpt_trials' in self._mb_training_param:
                self._mb_training_param.pop('p_hpt_trials')

        if (p_action_planner is not None) and (p_envmodel is None):
            raise ParamError('Agents using an action planner also need an environment model')

        self._previous_observation = None
        self._previous_action = None
        self._policy = p_policy
        self._envmodel = p_envmodel
        self._em_acc_thsld = p_em_acc_thsld
        self._action_planner = p_action_planner
        self._predicting_horizon = p_predicting_horizon
        self._controlling_horizon = p_controlling_horizon
        self._planning_width = p_planning_width

        Policy.__init__( self, 
                         p_observation_space = self._policy.get_observation_space(),
                         p_action_space = self._policy.get_action_space(),
                         p_id = p_policy.get_id(),
                         p_buffer_size = 0,
                         p_ada = p_ada,
                         p_visualize = p_visualize,
                         p_logging = p_logging )

        self.clear_buffer()

        if self._action_planner is not None:
            self._action_planner.setup(p_policy=self._policy,
                                       p_envmodel=self._envmodel,
                                       p_prediction_horizon=self._predicting_horizon,
                                       p_control_horizon=self._controlling_horizon,
                                       p_width_limit=self._planning_width)


## -------------------------------------------------------------------------------------------------
    def _init_hyperparam(self, **p_par):

        # 1 Create a dispatcher hyperparameter tuple for the agent
        self._hyperparam_tuple = HyperParamDispatcher(p_set=self._hyperparam_space)

        # 2 Extend agent's hp space and tuple from policy
        try:
            self._hyperparam_space.append( self._policy.get_hyperparam().get_related_set(), p_new_dim_ids=False)
            self._hyperparam_tuple.add_hp_tuple(self._policy.get_hyperparam())
        except:
            pass

        # 3 Extend agent's hp space and tuple from an optional environment model
        try:
            self._hyperparam_space.append(self._envmodel.get_hyperparam().get_related_set(), p_new_dim_ids=False)
            self._hyperparam_tuple.add_hp_tuple(self._envmodel.get_hyperparam())
        except:
            pass

        
## -------------------------------------------------------------------------------------------------
    def switch_logging(self, p_logging):
        super().switch_logging(p_logging)
        self._policy.switch_logging(p_logging)


## -------------------------------------------------------------------------------------------------
    def switch_adaptivity(self, p_ada: bool):
        super().switch_adaptivity(p_ada)
        try:
            self._policy.switch_adaptivity(p_ada)
        except AttributeError:
            pass


## -------------------------------------------------------------------------------------------------
    def set_log_level(self, p_level):
        super().set_log_level(p_level)
        self._policy.set_log_level(p_level)


## -------------------------------------------------------------------------------------------------
    def get_observation_space(self) -> MSpace:
        return self._policy.get_observation_space()


## -------------------------------------------------------------------------------------------------
    def get_action_space(self) -> MSpace:
        return self._policy.get_action_space()


## -------------------------------------------------------------------------------------------------
    def _extract_observation(self, p_state: State) -> State:
        if p_state.get_related_set() == self.get_observation_space():
            return p_state

        obs_space = self.get_observation_space()
        obs_dim_ids = obs_space.get_dim_ids()
        observation = State(obs_space)

        for dim_id in obs_dim_ids:
            p_state_ids = p_state.get_dim_ids()
            try:
                obs_idx = p_state.get_dim_ids().index(dim_id)
            except:
                obs_idx = obs_space.get_dim_ids().index(dim_id)
            observation.set_value(dim_id, p_state.get_value(p_state_ids[obs_idx]))

        return observation


## -------------------------------------------------------------------------------------------------
    def set_random_seed(self, p_seed=None):
        self._policy.set_random_seed(p_seed)
        if self._envmodel is not None:
            self._envmodel.set_random_seed(p_seed)


## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_state: State) -> Action:
        """
        Default implementation of a single agent.

        Parameters
        ----------
        p_state : State        
            State of the related environment.

        Returns
        -------
        action : Action
            Action object.

        """

        # 0 Intro
        self.log(self.C_LOG_TYPE_I, 'Action computation started')
        observation = self._extract_observation(p_state)

        # 1 Action computation
        if self._action_planner is None:
            # 1.1 W/o action planner
            action = self._policy.compute_action(observation)

        else:
            # 1.2 With action planner
            if self._envmodel.get_accuracy() >= self._em_acc_thsld:
                action = self._action_planner.compute_action(observation)
            else:
                action = self._policy.compute_action(observation)

        # 2 Outro
        self.log(self.C_LOG_TYPE_I, 'Action computation finished')
        self._previous_observation = observation
        self._previous_action = action
        return action


## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_state:State, p_reward:Reward) -> bool:
        """
        Default adaptation implementation of a single agent.

        Parameters
        ----------
        p_state : State       
            State object.
        p_reward : Reward     
            Reward object.
 
        Returns
        -------
        result : bool
            True, if something has been adapted. False otherwise.

        """

        # 1 Check: Adaptation possible?
        if self._previous_observation is None:
            self.log(self.C_LOG_TYPE_I, 'Adaption: previous observation is None -> adaptivity skipped')
            return False

        # 2 Extract agent specific observation data from state
        observation = self._extract_observation(p_state)
        adapted = False

        # 3 Adaptation
        if self._envmodel is None:
            # 3.1 Model-free adaptation
            adapted = self._policy.adapt( p_sars_elem=SARSElement(self._previous_observation, self._previous_action, p_reward, observation) )

        else:
            # 3.2 Model-based adaptation
            adapted = self._envmodel.adapt( p_sars_elem=SARSElement(self._previous_observation, self._previous_action, p_reward, observation) )

            if self._envmodel.get_accuracy() >= self._em_acc_thsld:
                adapted = adapted or self._adapt_policy_by_model()

        return adapted


## -------------------------------------------------------------------------------------------------
    def _adapt_policy_by_model(self):
        self.log(self.C_LOG_TYPE_I, 'Model-based policy training')
        training = RLTraining(**self._mb_training_param)
        training.get_scenario().setup_ext(p_env=self._envmodel,
                                          p_policy=self._policy,
                                          p_logging=self.get_log_level())

        # The RLTraining need to be adjusted again due to setup_ext()
        # And also due to model_train.py line 595 only executed on RLTraining init
        # Not after the setup_ext
        training._env = training.get_scenario().get_env()
        training._agent = training.get_scenario().get_agent()

        return training.run().num_adaptations > 0


## -------------------------------------------------------------------------------------------------
    def clear_buffer(self):
        self._policy.clear_buffer()
        if self._envmodel is not None:
            self._envmodel.clear_buffer()





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MultiAgent (Agent):
    """
    Multi-Agent.

    Parameters
    ----------
    p_name : str
        Name of agent. Default = ''.
    p_ada : bool               
        Boolean switch for adaptivity. Default = True.
    p_visualize : bool
        Boolean switch for env/agent visualisation. Default = False.
    p_logging
        Log level (see constants of class Log). Default = Log.C_LOG_ALL.
    """

    C_TYPE      = 'Multi-Agent'
    C_NAME      = ''

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_name : str = '', 
                  p_ada : bool = True, 
                  p_visualize : bool = False, 
                  p_logging = Log.C_LOG_ALL ):

        self._agents    = []
        self._agent_ids = []

        Model.__init__( self,
                        p_ada = p_ada,
                        p_name = p_name,
                        p_visualize = p_visualize,
                        p_logging = p_logging )


## -------------------------------------------------------------------------------------------------
    def switch_logging(self, p_logging) -> None:
        Log.switch_logging(self, p_logging=p_logging)

        for agent, weight in self._agents:
            agent.switch_logging(p_logging)


## -------------------------------------------------------------------------------------------------
    def switch_adaptivity(self, p_ada: bool):
        super().switch_adaptivity(p_ada)
        
        for agent, weight in self._agents:
            agent.switch_adaptivity(p_ada)

    
## -------------------------------------------------------------------------------------------------
    def set_log_level(self, p_level):
        Log.set_log_level(self, p_level)

        for agent, weight in self._agents:
            agent.set_log_level(p_level)


## -------------------------------------------------------------------------------------------------
    def add_agent(self, p_agent: Agent, p_weight=1.0) -> None:
        """
        Adds agent object to internal list of agents. 

        Parameters
        ----------
        p_agent : Agent
            Agent object to be added.
        p_weight : float         
            Optional weight for the agent. Default = 1.0.

        """

        p_agent.switch_adaptivity(self._adaptivity)
        self._agents.append( (p_agent, p_weight) )
        self._agent_ids.append( p_agent.get_id() )

        self.log(Log.C_LOG_TYPE_I, p_agent.C_TYPE + ' ' + p_agent.get_name() + ' added.')

        if p_agent._policy.get_hyperparam() is not None:
            self._hyperparam_space.append( p_set=p_agent._policy.get_hyperparam().get_related_set(), 
                                           p_new_dim_ids=False,
                                           p_ignore_duplicates=True )
        
        if p_agent._envmodel is not None:
            try:
                self._hyperparam_space.append(p_agent._envmodel.get_hyperparam().get_related_set())
            except:
                pass
 
        if self._hyperparam_tuple is None:
            self._hyperparam_tuple = HyperParamDispatcher(p_set=self._hyperparam_space)
            
        self._hyperparam_tuple.add_hp_tuple(p_agent.get_hyperparam())
                        

## -------------------------------------------------------------------------------------------------
    def get_agents(self):
        return self._agents


## -------------------------------------------------------------------------------------------------
    def get_agent(self, p_agent_id):
        """
        Returns information of a single agent.

        Returns
        -------
        agent_info : tuple
            agent_info[0] is the agent object itself and agent_info[1] it's weight
            
        """

        return self._agents[ self._agent_ids.index(p_agent_id) ]

    
## -------------------------------------------------------------------------------------------------
    def get_observation_space(self) -> MSpace:
        return None

    
## -------------------------------------------------------------------------------------------------
    def get_action_space(self) -> MSpace:
        return None

    
## -------------------------------------------------------------------------------------------------
    def set_random_seed(self, p_seed=None):
        for agent, weight in self._agents:
            agent.set_random_seed(p_seed)

    
## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_state: State) -> Action:
        self.log(self.C_LOG_TYPE_I, 'Start of action computation for all agents...')

        action = Action()

        for agent, weight in self._agents:
            action_agent = agent.compute_action(p_state)
            action_element = action_agent.get_elem( agent.get_id() )
            action_element.set_weight(weight)
            action.add_elem(agent.get_id(), action_element)

        self.log(self.C_LOG_TYPE_I, 'End of action computation for all agents...')
        return action

    
## -------------------------------------------------------------------------------------------------
    def _adapt(self, p_state:State, p_reward:Reward) -> bool:
        self.log(self.C_LOG_TYPE_I, 'Start of adaptation for all agents...')

        adapted = False
        for agent, weight in self._agents:
            if (p_reward.get_type() != Reward.C_TYPE_OVERALL) and not p_reward.is_rewarded(agent.get_id()):
                continue
            self.log(self.C_LOG_TYPE_I, 'Start adaption for agent', agent.get_id())
            adapted = agent.adapt(p_state=p_state, p_reward=p_reward) or adapted

        self.log(self.C_LOG_TYPE_I, 'End of adaptation for all agents...')

        self._set_adapted(adapted)
        return adapted

    
## -------------------------------------------------------------------------------------------------
    def clear_buffer(self):
        for agent, weight in self._agents:
            agent.clear_buffer()

    
## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure: Figure = None, p_plot_settings: list = ..., p_plot_depth: int = 0, p_detail_level: int = 0, p_step_rate: int = 0, **p_kwargs):
        """
        Doesn't support embedded plot of underlying agent hierarchy.
        """

        if not self.get_visualization(): return

        self.log(self.C_LOG_TYPE_I, 'Init visualization for all agents...')

        for agent, weight in self._agents:
            agent.init_plot(p_figure = None)

    
## -------------------------------------------------------------------------------------------------
    def update_plot(self, **p_kwargs):
        if not self.get_visualization(): return

        self.log(self.C_LOG_TYPE_I, 'Start visualization for all agents...')

        for agent, weight in self._agents:
            agent.update_plot()