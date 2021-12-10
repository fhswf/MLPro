from mlpro.bf.ml import *
from mlpro.rl.models import *
from mlpro.rl.pool.envs.robotinhtm import RobotHTM
from stable_baselines3 import PPO
from mlpro.wrappers.sb3 import WrPolicySB32MLPro
from mlpro.rl.pool.envmodels.htmenvmodel import HTMEnvModel

class SimulatedTraining(RLTraining):
    C_NAME = "Simulated"


class ActualTraining(RLTraining):
    C_NAME = "Actual"

# Implement model based agent
class MBAgent(Agent):
    def _adapt_policy_by_model(self):
        env_ext = self._envmodel
        pol_ext = self._policy
        class ScenarioRobotHTMSimulated(RLScenario):
            def _setup(self, p_mode, p_ada: bool, p_logging: bool) -> Model:
                self._env = env_ext
                return Agent(
                    p_policy=pol_ext,
                    p_envmodel=None,
                    p_name="Smith2",
                    p_ada=p_ada,
                    p_logging=p_logging,
                )

        # Instantiate training
        simulated_training = SimulatedTraining(
            p_scenario_cls=ScenarioRobotHTMSimulated,
            p_cycle_limit=100,
            p_max_cycles_per_episode=100,
            p_max_stagnations=0,
            p_collect_states=False,
            p_collect_actions=False,
            p_collect_rewards=False,
            p_collect_training=False,
            p_logging=True,
        )

        # Run Training
        simulated_training.run()

        # Save Policy to be used for actual
        self._policy = simulated_training.get_scenario().get_agent()._policy

        return True


# Implement RL Scenario for the actual environment to train the environment model
class ScenarioRobotHTMActual(RLScenario):

    C_NAME = "Matrix1"

    def _setup(self, p_mode, p_ada, p_logging):
        # 1 Setup environment
        self._env = RobotHTM(p_logging=True)

        policy_sb3 = PPO(
            policy="MlpPolicy", n_steps=100, env=None, _init_setup_model=False
        )

        policy_wrapped = WrPolicySB32MLPro(
            p_sb3_policy=policy_sb3,
            p_observation_space=self._env.get_state_space(),
            p_action_space=self._env.get_action_space(),
            p_ada=p_ada,
            p_logging=p_logging,
        )

        # 2 Setup standard single-agent with own policy
        return MBAgent(
            p_policy=policy_wrapped,
            p_envmodel=HTMEnvModel(),
            p_em_mat_thsld=0,
            p_name="Smith1",
            p_ada=p_ada,
            p_logging=p_logging,
        )

# 4 Train agent in scenario
now = datetime.now()


training = ActualTraining(
    p_scenario_cls=ScenarioRobotHTMActual,
    p_cycle_limit=300000,
    p_max_cycles_per_episode=100,
    p_max_stagnations=0,
    p_collect_states=True,
    p_collect_actions=True,
    p_collect_rewards=True,
    p_collect_training=True,
    p_logging=True,
)

training.run()

# 6 Create Plotting Class
class MyDataPlotting(DataPlotting):
    def get_plots(self):
        """
        A function to plot data
        """
        for name in self.data.names:
            maxval = 0
            minval = 0
            if self.printing[name][0]:
                fig = plt.figure(figsize=(7, 7))
                raw = []
                label = []
                ax = fig.subplots(1, 1)
                ax.set_title(name)
                ax.grid(True, which="both", axis="both")
                for fr_id in self.data.frame_id[name]:
                    raw.append(np.sum(self.data.get_values(name, fr_id)))
                    if self.printing[name][1] == -1:
                        maxval = max(raw)
                        minval = min(raw)
                    else:
                        maxval = self.printing[name][2]
                        minval = self.printing[name][1]

                    label.append("%s" % fr_id)
                ax.plot(raw)
                ax.set_ylim(minval - (abs(minval) * 0.1), maxval + (abs(maxval) * 0.1))
                ax.set_xlabel("Episode")
                ax.legend(label, bbox_to_anchor=(1, 0.5), loc="center left")
                self.plots[0].append(name)
                self.plots[1].append(ax)
                if self.showing:
                    plt.show()
                else:
                    plt.close(fig)


# 7 Plotting 1 MLpro
data_printing = {
    "Cycle": [False],
    "Day": [False],
    "Second": [False],
    "Microsecond": [False],
    "Smith1": [True, -1],
}


mem = training.get_results().ds_rewards
mem_plot = MyDataPlotting(mem, p_showing=True, p_printing=data_printing)
mem_plot.get_plots()
