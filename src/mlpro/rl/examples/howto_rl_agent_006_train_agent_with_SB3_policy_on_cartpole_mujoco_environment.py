## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl.examples
## -- Module  : howto_rl_agent_006_train_agent_with_SB3_policy_on_cartpole_mujoco_environment.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-02-23  0.0.0     MRD      Creation
## -- 2022-02-23  1.0.0     MRD      Released first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-02-23)

This module shows how to train a single agent with SB3 Policy on Cartpole MuJoCo Environment.
"""


import mlpro
from stable_baselines3 import PPO
from mlpro.rl import *
from mlpro.wrappers.sb3 import WrPolicySB32MLPro
from pathlib import Path




# 1 Create Cartpole Environment
class CartpoleEnvironment(Environment):
    
    C_NAME          = 'CartpoleEnvironment'
    C_REWARD_TYPE   = Reward.C_TYPE_OVERALL
    C_CYCLE_LIMIT   = 400

    def __init__(self, 
                p_mode=Mode.C_MODE_SIM, 
                p_mujoco_file=None, 
                p_frame_skip: int = 1, 
                p_state_mapping=None, 
                p_action_mapping=None,
                p_camera_conf: tuple = (None, None, None), 
                p_visualize: bool = False, 
                p_logging=Log.C_LOG_ALL):

        super().__init__(p_mode=p_mode, 
                        p_mujoco_file=p_mujoco_file, 
                        p_frame_skip=p_frame_skip, 
                        p_state_mapping=p_state_mapping, 
                        p_action_mapping=p_action_mapping,
                        p_camera_conf=p_camera_conf, 
                        p_visualize=p_visualize, 
                        p_logging=p_logging)

        
        self._state = State(self._state_space)
        self.reset()


    def _compute_reward(self, p_state_old: State = None, p_state_new: State = None) -> Reward:
        reward = Reward(self.C_REWARD_TYPE)
        reward.set_overall_reward(1)
        
        return reward
    
    def _compute_broken(self, p_state: State) -> bool:
        state_value = p_state.get_values()
        
        slide_pos_joint_thresh = 3
        hinge_pos_joint_thresh = 0.3
        
        slide_pos_joint = state_value[0]
        hinge_pos_joint = state_value[3]
        
        terminated = bool(
            slide_pos_joint < -slide_pos_joint_thresh
            or slide_pos_joint > slide_pos_joint_thresh
            or hinge_pos_joint < -hinge_pos_joint_thresh
            or hinge_pos_joint > hinge_pos_joint_thresh
        )
        
        self._state.set_terminal(terminated)
        return terminated

    def _reset(self, p_seed=None) -> None:
        pass
    


class MyScenario (RLScenario):
    C_NAME = 'Matrix'

    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging) -> Model:
        # 1.1 Setup environment
        model_file = os.path.join(os.path.dirname(mlpro.__file__), "bf/systems/pool/mujoco", "cartpole.xml")
        self._env = CartpoleEnvironment(p_logging=logging, p_mujoco_file=model_file, p_visualize=visualize)

        # 1.2 Setup Policy From SB3
        policy_sb3 = PPO(
            policy="MlpPolicy",
            n_steps=10,
            env=None,
            _init_setup_model=False,
            device="cpu",
            seed=1)

        # 1.3 Wrap the policy
        policy_wrapped = WrPolicySB32MLPro(
            p_sb3_policy=policy_sb3,
            p_cycle_limit=self._cycle_limit,
            p_observation_space=self._env.get_state_space(),
            p_action_space=self._env.get_action_space(),
            p_ada=p_ada,
            p_visualize=p_visualize,
            p_logging=p_logging)

        # 1.4 Setup standard single-agent with own policy
        return Agent(
            p_policy=policy_wrapped,
            p_envmodel=None,
            p_name='Smith',
            p_ada=p_ada,
            p_visualize=p_visualize,
            p_logging=p_logging
        )
        

# 3 Create scenario and run some cycles
if __name__ == '__main__':
    # Parameters for demo mode
    cycle_limit = 10000
    adaptation_limit = 0
    stagnation_limit = 0
    eval_frequency = 0
    eval_grp_size = 0
    logging = Log.C_LOG_WE
    visualize = True
    path = str(Path.home())

else:
    # Parameters for internal unit test
    cycle_limit = 50
    adaptation_limit = 5
    stagnation_limit = 5
    eval_frequency = 2
    eval_grp_size = 1
    logging = Log.C_LOG_NOTHING
    visualize = False
    path = str(Path.home())
 

# 2 Create scenario and start training
training = RLTraining(
    p_scenario_cls=MyScenario,
    p_cycle_limit=cycle_limit,
    p_adaptation_limit=adaptation_limit,
    p_stagnation_limit=stagnation_limit,
    p_eval_frequency=eval_frequency,
    p_eval_grp_size=eval_grp_size,
    p_path=path,
    p_visualize=visualize,
    p_logging=logging )



# 3 Training
training.run()
