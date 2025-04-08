## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - The integrative middleware framework for standardized machine learning
## -- Package : mlpro.rl.examples
## -- Module  : howto_rl_env_003_train_agent_with_minGRPO_policy_on_2D_collision_avoidance_environment.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2025-04-08  0.0.0     SY       Creation
## -- 2025-04-08  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2025-04-08)

This module shows how to train a minimal GRPO policy on MLPro's native 2D collision avoidance
environment.

You will learn:
    
1) How to set up a reward function for 2D collision avoidance environment

2) How to set up a scenario for 2D collision avoidance environment and also with SB3 wrapper

3) How to run the scenario and train the agent
    
4) How to plot from the generated results
    
"""


from mlpro.bf.plot import DataPlotting
from mlpro.rl import *
from mlpro.rl.pool.envs.collisionavoidance_2D import DynamicTrajectoryPlanner
from stable_baselines3 import PPO, DDPG, A2C
from stable_baselines3.common.noise import NormalActionNoise
from mlpro_int_sb3.wrappers import WrPolicySB32MLPro
from mlpro_int_gymnasium.wrappers import WrEnvMLPro2GYM
from pathlib import Path



# 1 Set up your own reward function
class MyDynamicTrajectoryPlanner(DynamicTrajectoryPlanner):
    
    def _compute_reward(self, p_state_old:State, p_state_new:State) -> Reward:
        
        number_of_collide_points = 0
        number_of_collide_lines = 0
        for _ in self.collide_point_list:
            number_of_collide_points += 1
        for _ in self.collide_line_list:
            number_of_collide_lines += 1

        distance = self._calc_distance()
        
        total_rewards = 0
        
        if number_of_collide_lines != 0:
            total_rewards -= number_of_collide_lines
            
        if number_of_collide_points != 0:
            total_rewards -= number_of_collide_points
        
        if (number_of_collide_lines == 0) and (number_of_collide_points == 0):
            total_rewards = 10
            
        total_rewards -= distance
            
        reward = Reward()
        reward.set_overall_reward(total_rewards)
        
        return reward



# 2 Implement your own RL scenario
class ScenarioTrajectoryPlanning(RLScenario):
    C_NAME = 'Trajectory Planning'

    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging) -> Model:
        # 1.1 Setup environment
        self._env = MyDynamicTrajectoryPlanner(
            p_visualize=p_visualize,
            p_logging=p_logging,
            p_num_point=4,
            p_action_boundaries=[-0.05,0.05],
            p_dt=0.01,
            p_cycle_limit=500
            )
        
        # Algorithm : PPO
        # policy_kwargs = dict(activation_fn=torch.nn.Tanh,
        #                       net_arch=dict(pi=[128, 128], vf=[128, 128]))
        # policy_sb3 = PPO(
        #     policy="MlpPolicy",
        #     n_steps=100,
        #     env=None,
        #     _init_setup_model=False,
        #     policy_kwargs=policy_kwargs,
        #     device="cpu",
        #     seed=2)
        
        # Algorithm : A2C
        # policy_kwargs = dict(activation_fn=torch.nn.Tanh,
        #                       net_arch=dict(pi=[128, 128], vf=[128, 128]))
        # policy_sb3 = A2C(
        #     policy="MlpPolicy",
        #     learning_rate=3e-4,
        #     n_steps=100,
        #     policy_kwargs=policy_kwargs,
        #     env=None,
        #     _init_setup_model=False,
        #     device="cpu",
        #     seed=2)
        
        
        # Algorithm : DDPG
        action_space = WrEnvMLPro2GYM.recognize_space(self._env.get_action_space())
        n_actions = action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        policy_kwargs = dict(net_arch=dict(pi=[128, 128], qf=[128, 128]))
        policy_sb3 = DDPG(
            policy="MlpPolicy",
            learning_rate=3e-4,
            buffer_size=1000,
            batch_size=128,
            learning_starts=1001,
            action_noise=action_noise,
            policy_kwargs=policy_kwargs,
            env=None,
            _init_setup_model=False,
            device="cpu",
            seed=3)

        policy_wrapped = WrPolicySB32MLPro(
            p_sb3_policy=policy_sb3,
            p_cycle_limit=self._cycle_limit,
            p_observation_space=self._env.get_state_space(),
            p_action_space=self._env.get_action_space(),
            p_ada=p_ada,
            p_visualize=p_visualize,
            p_logging=p_logging)

        # 2.2 Setup standard single-agent with own policy
        return Agent(
            p_policy=policy_wrapped,
            p_envmodel=None,
            p_name='Smith',
            p_ada=p_ada,
            p_visualize=p_visualize,
            p_logging=p_logging
        )



# 3 Create scenario and start training
if __name__ == "__main__":
    # 3.1 Parameters for demo mode
    cycle_limit             = 10000
    cycles_per_epi_limit    = 500
    logging                 = Log.C_LOG_WE
    visualize               = True
    path                    = str(Path.home())
    plotting                = True

else:
    # 3.2 Parameters for internal unit test
    cycle_limit             = 50
    cycles_per_epi_limit    = 5
    logging                 = Log.C_LOG_NOTHING
    visualize               = False
    path                    = None
    plotting                = False


# 4 Train agent in scenario 
training = RLTraining(
    p_scenario_cls=ScenarioTrajectoryPlanning,
    p_cycle_limit=cycle_limit,
    p_cycles_per_epi_limit=cycles_per_epi_limit,
    p_path=path,
    p_visualize=visualize,
    p_logging=logging
)

training.run()


# 5 Plotting with MLPro rewards
if __name__ == "__main__":  
    mem = training.get_results().ds_rewards
    data_printing = {mem.names[0]: [False],
                     mem.names[1]: [False],
                     mem.names[2]: [False],
                     mem.names[3]: [False],
                     mem.names[4]: [True, 0, -1]}
    mem_plot = DataPlotting(mem,
                            p_showing=plotting,
                            p_printing=data_printing,
                            p_type=DataPlotting.C_PLOT_TYPE_EP,
                            p_window=100)
    mem_plot.get_plots()