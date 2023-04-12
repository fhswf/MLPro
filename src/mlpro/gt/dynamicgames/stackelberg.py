## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.gt.dp
## -- Module  : stackelberg.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-04-03  0.0.0     SY       Creation
## -- 2023-04-12  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-04-12)

This module provides model classes for Stackelberg Games in dynamic programming.
"""

from mlpro.gt.dynamicgames.basics import *





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class GTPlayer_SG (GTPlayer_DG):
    """
    This class implements a game theoretical player model in a stackelberg game mode, in which
    there is a possibility to assign the role of the player as a leader or follower.

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
    def _adapt(self, *p_args) -> bool:
        
        # 1 Check: Adaptation possible?
        if self._previous_observation is None:
            self.log(self.C_LOG_TYPE_I, 'Adaption: previous observation is None -> adaptivity skipped')
            return False

        # 2 Extract agent specific observation data from state
        state = p_args[0]
        reward = p_args[1]

        if self.role == self.C_PLAYER_FOLLOWER:
            action_leaders = p_args[2]

        observation = self._extract_observation(state)
        adapted = False

        # 3 Adaptation
        if self._envmodel is None:
            # 3.1 Model-free adaptation
            if self.role == self.C_PLAYER_FOLLOWER:
                adapted = self._policy.adapt(
                    SARSElement(self._previous_observation, self._previous_action, reward, observation),
                    action_leaders)
            else:
                adapted = self._policy.adapt(
                    SARSElement(self._previous_observation, self._previous_action, reward, observation))

        else:
            # 3.2 Model-based adaptation
            adapted = self._envmodel.adapt(
                SARSElement(self._previous_observation, self._previous_action, reward, observation))

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



