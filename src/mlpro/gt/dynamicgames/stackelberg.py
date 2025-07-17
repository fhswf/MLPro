## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.gt.dynamicgames
## -- Module  : stackelberg.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-04-03  0.0.0     SY       Creation
## -- 2023-04-12  1.0.0     SY       Release of first version
## -- 2023-05-11  1.1.0     SY       Refactoring
## -- 2021-08-22  1.1.1     SY       Refactoring due to compatibility in mlpro.gt.dynamicsgames
## -- 2025-07-17  1.2.0     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.2.0 (2025-07-17) 

This module provides model classes for Stackelberg Games in dynamic programming.
"""


from mlpro.bf import Log
from mlpro.bf.systems import State, Action
from mlpro.rl import ActionPlanner, Policy, EnvModel, SARSElement, Reward
from mlpro.gt.dynamicgames.basics import *



# Export list for public API
__all__ = [ 'GTPlayer_SG', 
            'GTMultiPlayer_SG' ]




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GTPlayer_SG (Player):
    """
    This class implements a game theoretical player model in a stackelberg game mode, in which
    there is a possibility to assign the role of the player as a leader or follower.

    The leader(s) has a priority to compute actions and adapt policies over the followers.
    Then, the followers can react according to the selected actions by the leaders, while computing
    their actions and adapting their policies. Thus, as followers, the selected actions by the
    leaders will assign as one of the inputs on both _adapt and compute_action methods.

    Parameters
    ----------
    p_role          
        Role of the player. Default = C_PLAYER_LEADER.
    """

    C_TYPE              = 'GT Player SG'
    C_PLAYER_LEADER     = 0
    C_PLAYER_FOLLOWER   = 1


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_policy:Policy,
                 p_envmodel:EnvModel = None,
                 p_em_acc_thsld=0.9,
                 p_action_planner:ActionPlanner = None,
                 p_predicting_horizon=0,
                 p_controlling_horizon=0,
                 p_planning_width=0,
                 p_name='',
                 p_ada=True,
                 p_visualize:bool=True,
                 p_logging=Log.C_LOG_ALL,
                 p_role:int=0,
                 **p_mb_training_param):
        
        super().__init__(p_policy=p_policy,
                        p_envmodel=p_envmodel,
                        p_em_acc_thsld=p_em_acc_thsld,
                        p_action_planner=p_action_planner,
                        p_predicting_horizon=p_predicting_horizon,
                        p_controlling_horizon=p_controlling_horizon,
                        p_planning_width=p_planning_width,
                        p_name=p_name,
                        p_ada=p_ada,
                        p_visualize=p_visualize,
                        p_logging=p_logging,
                        **p_mb_training_param)
        
        self._role = p_role
    
    
## -------------------------------------------------------------------------------------------------
    def _adapt(self, **p_args) -> bool:
        
        # 1 Check: Adaptation possible?
        if self._previous_observation is None:
            self.log(self.C_LOG_TYPE_I, 'Adaption: previous observation is None -> adaptivity skipped')
            return False

        # 2 Extract agent specific observation data from state
        state = p_args['p_state']
        reward = p_args['p_reward']

        if self._role == self.C_PLAYER_FOLLOWER:
            action_leaders = p_args['p_action_leaders']

        observation = self._extract_observation(state)
        adapted = False

        # 3 Adaptation
        if self._envmodel is None:
            # 3.1 Model-free adaptation
            if self._role == self.C_PLAYER_FOLLOWER:
                adapted = self._policy.adapt(
                    p_sars_element=SARSElement(self._previous_observation,
                                               self._previous_action,
                                               reward,
                                               observation),
                    p_action_leaders=action_leaders)
            else:
                adapted = self._policy.adapt(
                    p_sars_element=SARSElement(self._previous_observation,
                                               self._previous_action,
                                               reward,
                                               observation))

        else:
            # 3.2 Model-based adaptation
            adapted = self._envmodel.adapt(
                p_sars_element=SARSElement(self._previous_observation,
                                           self._previous_action,
                                           reward,
                                           observation))

            if self._envmodel.get_maturity() >= self._em_mat_thsld:
                adapted = adapted or self._adapt_policy_by_model()

        return adapted
    
    
## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_state: State, p_action_leaders=False) -> Action:

        # 0 Intro
        self.log(self.C_LOG_TYPE_I, 'Action computation started')
        observation = self._extract_observation(p_state)

        # 1 Action computation
        if self._action_planner is None:
            # 1.1 W/o action planner
            action = self._policy.compute_action(observation, p_action_leaders)

        else:
            # 1.2 With action planner
            action = self._action_planner.compute_action(observation, p_action_leaders)

        # 2 Outro
        self.log(self.C_LOG_TYPE_I, 'Action computation finished')
        self._previous_observation = observation
        self._previous_action = action
        return action





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GTMultiPlayer_SG (MultiPlayer):
    """
    This class implements a game theoretical multi-player model in a stackelberg game mode.
    """

    C_TYPE = 'GT Multi-Player SG'

    
## -------------------------------------------------------------------------------------------------
    def _adapt(self, **p_args) -> bool:

        state = p_args['p_state']
        reward = p_args['p_reward']
        adapted = False

        self.log(self.C_LOG_TYPE_I, 'Start of adaptation for all agents...')
        
        ## leaders make initial adaptations ##
        for agent_entry in self._agents:
            agent = agent_entry[0]
            if agent._role == GTPlayer_SG.C_PLAYER_LEADER:
                if (reward.get_type() != Reward.C_TYPE_OVERALL) and not reward.is_rewarded(agent.get_id()):
                    continue
                self.log(self.C_LOG_TYPE_I, 'Start adaption for agent', agent.get_id())
                adapted = agent.adapt(p_state=state, p_reward=reward) or adapted
                
        ## followers make adaptations ## 
        for agent_entry in self._agents:
            agent = agent_entry[0]
            if agent._role == GTPlayer_SG.C_PLAYER_FOLLOWER:
                action_leaders = []
                if (reward.get_type() != Reward.C_TYPE_OVERALL) and not reward.is_rewarded(agent.get_id()):
                    continue
                for ag in self._agents:
                    potential_leader = ag[0]
                    if potential_leader._role == GTPlayer_SG.C_PLAYER_LEADER:
                        action_leaders.append(potential_leader._previous_action)
                self.log(self.C_LOG_TYPE_I, 'Start adaption for agent', agent.get_id())
                adapted = agent.adapt(p_state=state,
                                      p_reward=reward,
                                      p_action_leaders=action_leaders) or adapted

        self.log(self.C_LOG_TYPE_I, 'End of adaptation for all agents...')

        self._set_adapted(adapted)
        return adapted

    
## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_state: State) -> Action:

        self.log(self.C_LOG_TYPE_I, 'Start of action computation for all agents...')

        action = Action()

        ## leaders makes initial moves ##
        for agent, weight in self._agents:
            if agent._role == GTPlayer_SG.C_PLAYER_LEADER:
                action_agent = agent.compute_action(p_state)
                action_element = action_agent.get_elem(agent.get_id())
                action_element.set_weight(weight)
                action.add_elem(agent.get_id(), action_element)

        ## followers makes moves ##
        for agent, weight in self._agents:
            if agent._role == GTPlayer_SG.C_PLAYER_FOLLOWER:
                action_leaders = []
                for ag in self._agents:
                    potential_leader = ag[0]
                    if potential_leader._role == GTPlayer_SG.C_PLAYER_LEADER:
                        action_leaders.append(potential_leader._previous_action)
                action_agent = agent.compute_action(p_state, action_leaders)
                action_element = action_agent.get_elem(agent.get_id())
                action_element.set_weight(weight)
                action.add_elem(agent.get_id(), action_element)

        self.log(self.C_LOG_TYPE_I, 'End of action computation for all agents...')
        return action
