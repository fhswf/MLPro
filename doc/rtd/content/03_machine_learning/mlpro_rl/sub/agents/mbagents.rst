.. _target_agents_MBRL:
Model-based agents
==================

Model-based agents differ from model-free agents in that they require learning an environment model, which is unnecessary in model-free RL.
In MLPro, this environment model is represented as **EnvModel**, which learns the behavior and dynamics of the environment.

Once trained, the environment model predicts output states, rewards, or status changes based on input actions.
If the predictions significantly deviate from the actual environment values, the environment model is retrained and incorporated into the agent's adaptation process.
This ensures that the policy remains optimized and accurate.

A key component of model-based RL is internal episodic training, where the policy interacts with the learned environment model instead of the real environment to speed up learning.

**Action Planning in Model-Based RL**

Once a reliable environment model is obtained, the agent can optionally be extended into an action planner.
An action planner helps the environment plan future actions, such as using Model Predictive Control (MPC).

MLPro-RL provides a base class for ActionPlanner, which users can extend by defining:

    - **_plan_action** → The core planning algorithm

    - **_setup (optional)** → Custom setup steps

Here is an example of creating a custom action planner:

.. code-block:: python

    from mlpro.rl.models import *
    
    class MyActionPlanner (ActionPlanner):
        """
        Creates an action planner that satisfies mlpro interface.
        """

        C_NAME      = 'MyActionPlanner'
    
        def _setup(self):
            """
            Optional custom setup method.
            """

            pass
    
    
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
 

**Developing an Environment Model (EnvModel)**

To create an environment model, follow these steps:

    - Implement an adaptive function that predicts the next environment state given an action.

    - Extend the **EnvModel** class from MLPro while inheriting from the actual environment.

    - Use the state transition model from the adaptive function while relying on the original environment for reward, success, and failure calculations.

Here is an example of creating an environment model:

.. code-block:: python

    from mlpro.rl.model_env import EnvModel
    from mlpro.rl.pool.envs.robotinhtm import RobotHTM

    class OurEnvModel(RobotHTM, EnvModel):
        C_NAME = "Our Env Model"

        # Put necessary input argument in initialization
        def __init__(
            self,
            p_num_joints=4,
            p_target_mode="Random",
            p_ada=True,
            p_logging=False,
        ):

            # Initialize the actual environment to get all environment functionalities, such as
            # _simulate_reaction, _reset, _compute_reward, _compute_broken and _compute_success
            RobotHTM.__init__(self, p_num_joints=p_num_joints, p_target_mode=p_target_mode)
            
            # Setup Adaptive Function
            afct_strans = AFctSTrans(
                OurStatePredictor,
                p_state_space=self._state_space,
                p_action_space=self._action_space,
                p_threshold=1.8,
                p_buffer_size=20000,
                p_ada=p_ada,
                p_logging=p_logging,
            )

            # In this case set only p_afct_strans, which tells the module to use
            # _simulate_reaction from the adaptive function instead of from the actual environment
            # Set to None to use function such as compute_reward, compute_broken and compute_success
            # from the actual environment
            EnvModel.__init__(
                self,
                p_observation_space=self._state_space,
                p_action_space=self._action_space,
                p_latency=timedelta(seconds=self.dt),
                p_afct_strans=afct_strans,
                p_afct_reward=None,
                p_afct_success=None,
                p_afct_broken=None,
                p_ada=p_ada,
                p_logging=p_logging,
            )

            self.reset()


**Cross reference**

    - `Howto RL-AGENT-001: Train and reload single agent (Gymnasium) <https://mlpro-int-sb3.readthedocs.io/en/latest/content/01_example_pool/01_howtos_agent/howto_rl_agent_001_train_and_reload_single_agent_gym.html>`_
    - :ref:`Howto RL-MB-001: MBRL with MPC on Grid World environment <Howto MB RL 001>`
    - `Howto RL-MB-002: MBRL on RobotHTM environment <https://mlpro-int-sb3.readthedocs.io/en/latest/content/01_example_pool/04_howtos_mb/howto_rl_mb_002_robothtm_environment.html>`_
    - :ref:`MLPro-SL <target_bf_sl_afct>`

