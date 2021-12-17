## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
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
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.3.2 (2021-12-17)

This module provides model classes for policies, model-free and model-based agents and multi-agents.
"""


from mlpro.bf.data import *
from mlpro.rl.models_sar import *
from mlpro.rl.models_env import *




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

    Furthermore a policy class can compute actions from states.

    Hyperparameters of the policy should be stored in the internal object self._hp_list, so that
    they can be tuned from outside. Optionally a policy-specific callback method can be called on 
    changes. For more information see class HyperParameterList.
    """

    C_TYPE          = 'Policy'
    C_NAME          = '????'
    C_BUFFER_CLS    = SARSBuffer

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

        super().__init__(p_buffer_size=p_buffer_size, p_ada=p_ada, p_logging=p_logging)
        self._observation_space = p_observation_space
        self._action_space      = p_action_space
        self.set_id(0)


## -------------------------------------------------------------------------------------------------
    def get_observation_space(self) -> MSpace:
        return self._observation_space


## -------------------------------------------------------------------------------------------------
    def get_action_space(self) -> MSpace:
        return self._action_space


## -------------------------------------------------------------------------------------------------
    def get_id(self):
        return self._id


## -------------------------------------------------------------------------------------------------
    def set_id(self, p_id):
        self._id = p_id


## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_obs:State) -> Action:
        """
        Specific action computation method to be redefined. 

        Parameters:
            p_obs       Observation data

        Returns:
            Action object
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _adapt(self, *p_args) -> bool:
        """
        Adapts the policy based on State-Action-Reward-State (SARS) data.

        Parameters:
            p_arg[0]           Object of type SARSElement
        """

        raise NotImplementedError





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class ActionPlanner (Log):
    """
    Template class for action planning algorithms to be used as part of planning agents.
    """

    C_TYPE          = 'Action Planner'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_logging=True):
        super().__init__(p_logging=p_logging)
        self._action_path = []


## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_state:State, p_policy:Policy, p_envmodel:EnvModel, p_depth, p_width) -> Action:
        """
        Computes a path of actions with defined length that maximizes the reward of the given 
        environment model.
        
        Parameters:
            p_state             Current state of environment
            p_policy            Poliy of an agent
            p_envmodel          Environment model
            p_depth             Planning depth (=length of action path to be predicted)
            p_width             Planning width (=number of alternative actions per planning level)
        """

        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def clear_action_path(self):
        self._action_path.clear()





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Agent(Policy):
    """
    This class represents a single agent model.

    """

    C_TYPE          = 'Agent'
    C_NAME          = ''

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_policy:Policy, p_envmodel:EnvModel=None, p_em_mat_thsld=1, p_action_planner:ActionPlanner=None, p_planning_depth=0, p_planning_width=0, p_name='', p_id=0, p_ada=True, 
                p_logging=True):
        """
        Parameters:
            p_policy            Policy object
            p_envmodel          Optional environment model object
            p_em_mat_thsld      Threshold for environment model maturity (whether or not the envmodel is 'good' enougth to be used to train the policy)
            p_action_planner    Optional action planner object (obligatory for model based agents)
            p_planning_depth    Optional planning depth (obligatory for model based agents)
            p_planning_width    Optional planning width (obligatory for model based agents)
            p_name              Optional name of agent
            p_id                Unique agent id (especially important for multi-agent scenarios)
            p_ada               Boolean switch for adaptivity
            p_logging           Boolean switch for logging functionality
        """

        if p_name != '': 
            self.set_name(p_name)
        else:
            self.set_name(self.C_NAME)

        if   ( p_action_planner is not None ) and ( p_envmodel is None ):
           raise ParamError('Agents using an action planner also need an environment model')
           
        self._previous_observation  = None
        self._previous_action       = None
        self._policy                = p_policy
        self._action_space          = self._policy.get_action_space()
        self._observation_space     = self._policy.get_observation_space()
        self._envmodel              = p_envmodel
        self._em_mat_thsld          = p_em_mat_thsld
        self._action_planner        = p_action_planner
        self._planning_depth        = p_planning_depth
        self._planning_width        = p_planning_width

        self._set_id(p_id)

        Log.__init__(self, p_logging)
        self.switch_logging(p_logging)
        self.switch_adaptivity(p_ada)

        self.clear_buffer()


## -------------------------------------------------------------------------------------------------
    def _set_id(self, p_id): 
        super().set_id(p_id)
        self._policy.set_id(p_id)


## -------------------------------------------------------------------------------------------------
    def set_id(self, p_id):
        """
        The unique agent id will be defined while instantiation and can't be changed any more.
        """

        pass


## -------------------------------------------------------------------------------------------------
    def get_name(self):
        return self._name


## -------------------------------------------------------------------------------------------------
    def set_name(self, p_name):
        self._name   = p_name
        self.C_NAME  = p_name


## -------------------------------------------------------------------------------------------------
    def switch_logging(self, p_logging: bool):
        super().switch_logging(p_logging)
        self._policy.switch_logging(p_logging)


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
    def _extract_observation(self, p_state:State) -> State:
        if p_state.get_related_set() == self.get_observation_space(): return p_state

        obs_space   = self.get_observation_space()
        obs_dim_ids = obs_space.get_dim_ids()
        observation = State(obs_space)

        for dim_id in obs_dim_ids:
            observation.set_value(dim_id, p_state.get_value(dim_id))

        return observation


## -------------------------------------------------------------------------------------------------
    def set_random_seed(self, p_seed=None):
        self._policy.set_random_seed(p_seed)
        if self._envmodel is not None:
            self._envmodel.set_random_seed(p_seed)


## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_state:State) -> Action:
        """
        Default implementation of a single agent.

        Parameters:
            p_state         State of the related environment

        Returns:
            Action object
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
            action = self._action_planner.compute_action(observation, self._policy, self._envmodel, self._planning_depth, self._planning_width)


        # 2 Outro
        self.log(self.C_LOG_TYPE_I, 'Action computation finished')
        self._previous_observation  = observation
        self._previous_action       = action
        return action


## -------------------------------------------------------------------------------------------------
    def _adapt(self, *p_args) -> bool:
        """
        Default adaptation implementation of a single agent.

        Parameters:
            p_args[0]       State object (see class State)
            p_args[1]       Reward object (see class Reward)
 
        Returns:
            True, if something has beed adapted
        """

        # 1 Check: Adaptation possible?
        if self._previous_observation is None:
            self.log(self.C_LOG_TYPE_I, 'Adaption: previous observation is None -> adaptivity skipped')
            return False


        # 2 Extract agent specific observation data from state
        state       = p_args[0]
        reward      = p_args[1]
        observation = self._extract_observation(state)
        adapted     = False


        # 3 Adaptation
        if self._envmodel is None:
            # 3.1 Model-free adaptation
            adapted = self._policy.adapt(SARSElement(self._previous_observation, self._previous_action, reward, observation))

        else:
            # 3.2 Model-based adaptation
            adapted = self._envmodel.adapt(SARSElement(self._previous_observation, self._previous_action, reward, observation))

            if self._envmodel.get_maturity() >= self._em_mat_thsld:
                adapted = adapted or self._adapt_policy_by_model()

        return adapted


    def _adapt_policy_by_model(self):
        # 1 Instantiate a Scenario object
        # 2 Instantiate a Training object
        # 3 Execute episodical training
        
        return True


## -------------------------------------------------------------------------------------------------
    def clear_buffer(self):
        self._policy.clear_buffer()
        if self._envmodel is not None: self._envmodel.clear_buffer()





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MultiAgent(Agent):
    """
    This class implements a reinforcement learning multi-agent model.
    """

    C_TYPE          = 'Multi-Agent'
    C_NAME          = ''
    C_SUFFIX        = '.cfg'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_name='', p_ada=True, p_logging=True):
        """
        Parameters:
            p_name              Name of agent
            p_ada               Boolean switch for adaptivity
            p_logging           Boolean switch for logging functionality
        """

        self._agents    = []
        self._agent_ids = []
        self.set_name(p_name)

        Log.__init__(self, p_logging)
        self.switch_logging(p_logging)
        self.switch_adaptivity(p_ada)
        self._set_adapted(False)
        

## -------------------------------------------------------------------------------------------------
    def switch_logging(self, p_logging:bool) -> None: 
        Log.switch_logging(self, p_logging=p_logging)

        for agent_entry in self._agents:
            agent_entry[0].switch_logging(p_logging)


## -------------------------------------------------------------------------------------------------
    def set_log_level(self, p_level):
        Log.set_log_level(self, p_level)

        for agent_entry in self._agents:
            agent_entry[0].set_log_level(p_level)


## -------------------------------------------------------------------------------------------------
    def get_filename(self) -> str:
        return self.C_TYPE + ' ' + self.C_NAME + self.C_SUFFIX


## -------------------------------------------------------------------------------------------------
    def load(self, p_path, p_filename=None) -> bool:
        # load all subagents
        for i, agent_entry in enumerate(self._agents):
            agent       = agent_entry[0]
            agent_name  = agent.C_TYPE + ' ' + agent.C_NAME + '(' + str(i) + ')'

            if agent.load(p_path, agent_name + agent.C_SUFFIX):
                self.log(Log.C_LOG_TYPE_I, agent_name + ' loaded')
            else:
                self.log(Log.C_LOG_TYPE_E, agent_name + ' not loaded')
                return False

        return True


## -------------------------------------------------------------------------------------------------
    def save(self, p_path, p_filename=None) -> bool:
        # save all subagents
        for i, agent_entry in enumerate(self._agents):
            agent       = agent_entry[0]
            agent_name  = agent.C_TYPE + ' ' + agent.C_NAME + '(' + str(i) + ')'

            if agent.save(p_path, agent_name + agent.C_SUFFIX):
                self.log(Log.C_LOG_TYPE_I, agent_name + ' saved')
            else:
                self.log(Log.C_LOG_TYPE_E, agent_name + ' not saved')
                return False

        return True


## -------------------------------------------------------------------------------------------------
    def add_agent(self, p_agent:Agent, p_weight=1.0) -> None:
        """
        Adds agent object to internal list of agents. 

        Parameters:
            p_agent           Agent object
            p_weight          Optional weight for the agent

        Returns:
            Nothing
        """

        p_agent.switch_adaptivity(self._adaptivity)
        self._agent_ids.append(p_agent.get_id())
        self._agents.append([p_agent, p_weight])
        p_agent.set_name(str(p_agent.get_id()) + ' ' + p_agent.get_name())
        self.log(Log.C_LOG_TYPE_I, p_agent.C_TYPE + ' ' + p_agent.get_name() + ' added.')


## -------------------------------------------------------------------------------------------------
    def get_agents(self):
        return self._agents


## -------------------------------------------------------------------------------------------------
    def get_agent(self, p_agent_id):
        """
        Returns informations of a single agent.

        Returns
        -------
        agent_info : list
            agent_info[0] is the agent object itself and agent_info[1] it's weight
            
        """

        return self._agents[self._agent_ids.index(p_agent_id)]


## -------------------------------------------------------------------------------------------------
    def get_observation_space(self) -> MSpace:
        return None


## -------------------------------------------------------------------------------------------------
    def get_action_space(self) -> MSpace:
        return None


## -------------------------------------------------------------------------------------------------
    def set_random_seed(self, p_seed=None):
        for i, agent_entry in enumerate(self._agents):
            agent       = agent_entry[0]
            agent.set_random_seed(p_seed)


## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_state:State) -> Action:
        self.log(self.C_LOG_TYPE_I, 'Start of action computation for all agents...')      

        action = Action()

        for agent, weight in self._agents:
            action_agent    = agent.compute_action(p_state)
            action_element  = action_agent.get_elem(agent.get_id())
            action_element.set_weight(weight)
            action.add_elem(agent.get_id(), action_element)

        self.log(self.C_LOG_TYPE_I, 'End of action computation for all agents...')  
        return action      


## -------------------------------------------------------------------------------------------------
    def _adapt(self, *p_args) -> bool:
        state     = p_args[0]
        reward    = p_args[1]

        self.log(self.C_LOG_TYPE_I, 'Start of adaptation for all agents...')      

        adapted = False
        for agent_entry in self._agents:
            agent = agent_entry[0]
            if ( reward.get_type() != Reward.C_TYPE_OVERALL ) and not reward.is_rewarded(agent.get_id()): continue
            self.log(self.C_LOG_TYPE_I, 'Start adaption for agent', agent.get_id())
            adapted = adapted or agent.adapt(state,reward)

        self.log(self.C_LOG_TYPE_I, 'End of adaptation for all agents...')        

        self._set_adapted(adapted)
        return adapted


## -------------------------------------------------------------------------------------------------
    def clear_buffer(self):
        for agent_entry in self._agents:
            agent_entry[0].clear_buffer()


## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure=None):
        """
        Doesn't support embedded plot of underlying agent hierarchy.
        """

        self.log(self.C_LOG_TYPE_I, 'Init vizualization for all agents...')      

        for agent_entry in self._agents: agent_entry[0].init_plot(None)


## -------------------------------------------------------------------------------------------------
    def update_plot(self):
        self.log(self.C_LOG_TYPE_I, 'Start vizualization for all agents...')      

        for agent_entry in self._agents: agent_entry[0].update_plot()
