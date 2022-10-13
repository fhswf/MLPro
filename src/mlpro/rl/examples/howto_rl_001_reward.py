## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : howto_rl_001_types_of_reward.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-05-30  0.0.0     DA       Creation
## -- 2021-05-31  1.0.0     DA       Release of first version
## -- 2021-09-11  1.0.1     MRD      Change Header information to match our new library name
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2021-09-11)

This module shows how to create and interprete reward objects in own projects.
"""


from mlpro.bf.various import Log
from mlpro.rl.models import Reward



class MyLog(Log):
    C_TYPE      = 'Reward Demo'
    C_NAME      = ''


# 1 Some initial stuff
my_log = MyLog()

# 1.1 Unique agent ids
C_AGENT_1       = 1
C_AGENT_2       = 2
C_AGENT_3       = 3

# 1.2 Unique action ids
C_AGENT_1_ACT_1 = 1
C_AGENT_1_ACT_2 = 2
C_AGENT_1_ACT_3 = 3
C_AGENT_2_ACT_1 = 4


# 2 Rewards as single overall scalar values (independent from agents and actions)
my_log.log(Log.C_LOG_TYPE_I, 'Example for reward type C_TYPE_OVERALL:')
reward = Reward(p_type=Reward.C_TYPE_OVERALL)
reward.set_overall_reward(4.77)
my_log.log(Log.C_LOG_TYPE_I, 'Reward is just a scalar...', reward.get_agent_reward(0), '\n')


# 3 Rewards as scalar values for every agent
my_log.log(Log.C_LOG_TYPE_I, 'Example for reward type C_TYPE_EVERY_AGENT')
reward = Reward(p_type=Reward.C_TYPE_EVERY_AGENT)
my_log.log(Log.C_LOG_TYPE_I, 'Reward is a list with entries for each agent...')
reward.add_agent_reward(C_AGENT_1, 4.77)
my_log.log(Log.C_LOG_TYPE_I, 'Reward for agent 1 added:', reward.get_agent_reward(C_AGENT_1))
reward.add_agent_reward(C_AGENT_2, 5.19)
my_log.log(Log.C_LOG_TYPE_I, 'Reward for agent 2 added:', reward.get_agent_reward(C_AGENT_2))
reward.add_agent_reward(C_AGENT_3, 0.23)
my_log.log(Log.C_LOG_TYPE_I, 'Reward for agent 3 added:', reward.get_agent_reward(C_AGENT_3), '\n')

# 4 Rewards as scalar values for every agent and it's actions
my_log.log(Log.C_LOG_TYPE_I, 'Example for reward type C_TYPE_EVERY_ACTION')
reward = Reward(p_type=Reward.C_TYPE_EVERY_ACTION)
my_log.log(Log.C_LOG_TYPE_I, 'Reward is a list with entries for each agent and its action components...')
reward.add_action_reward(C_AGENT_1, C_AGENT_1_ACT_1, 1.23)
my_log.log(Log.C_LOG_TYPE_I, 'Reward for agent 1, action 1 added:', reward.get_action_reward(C_AGENT_1, C_AGENT_1_ACT_1))
reward.add_action_reward(C_AGENT_1, C_AGENT_1_ACT_2, 0.47)
my_log.log(Log.C_LOG_TYPE_I, 'Reward for agent 1, action 2 added:', reward.get_action_reward(C_AGENT_1, C_AGENT_1_ACT_2))
reward.add_action_reward(C_AGENT_1, C_AGENT_1_ACT_3, 1.63)
my_log.log(Log.C_LOG_TYPE_I, 'Reward for agent 1, action 3 added:', reward.get_action_reward(C_AGENT_1, C_AGENT_1_ACT_3))
reward.add_action_reward(C_AGENT_2, C_AGENT_2_ACT_1, 4.23)
my_log.log(Log.C_LOG_TYPE_I, 'Reward for agent 2, action 4 added:', reward.get_action_reward(C_AGENT_2, C_AGENT_2_ACT_1))
