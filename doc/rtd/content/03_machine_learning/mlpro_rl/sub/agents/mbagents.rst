.. _target_agents_MBRL:
Model-Based Agents
==================

Model-Based Agents have a dissimilar learning target as Model-Free Agents, whereas learning the environment model is not required in the model-free RL.
An environment model can be incorporated into a single agent, see :ref:`EnvModel <customEnvModel>` for an overview.
Then, this model learns the behaviour and dynamics of the environment.
After learning the environment, the model is optimized to be able to accurately predict the output states, rewards, or status of the environment with respect to the calculated actions.
As a result, if the predictions of the subsequent state and reward diverge too far from the actual values of the environment, the environment model itself is incorporated into the agent's adaptation process and is always retrained.
An adaptation in the environment model necessitates an adaptation in the policy.
The foundation for this is an internal episodic training of the policy in interaction with the environment model.

After having a model that can accurately predict the behaviour of the environment, the single agent is optionally extended as an action planner.
The action planner can be used by the environment to plan the next actions, e.g. using Model Predictive Control.
In MLPro-RL, we have also provided a base class for ActionPlanner, where only the action planner method (**_plan_action**) and an optional custom setup method (**_setup**) are needed to be adjusted, as shown below:

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
 
**Environment Model (EnvModel)**

To set up environment model, the adaptive function needs to created first. In this case, our adaptive function will
predict the next state of the environment based on provided action.
After that, we need to create another class that is inherited from the actual environment module and **EnvModel**, in this case
RobotHTM. For now, we only use the state transition model. The reward, success and broken model are taken from
the original environment module.  


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


**Cross Reference**

- :ref:`Howto RL-MB-001: MBRL on RobotHTM Environment <Howto MB RL 001>`
- :ref:`Howto RL-MB-002: MBRL with MPC on Grid World Environment <Howto MB RL 002>`
- :ref:`MLPro-SL <target_bf_sl_afct>`

