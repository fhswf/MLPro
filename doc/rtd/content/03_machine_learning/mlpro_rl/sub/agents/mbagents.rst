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


**Cross Reference**

- :ref:`Howto RL-MB-001: MBRL on RobotHTM Environment <Howto MB RL 001>`
- :ref:`Howto RL-MB-002: MBRL with MPC on Grid World Environment <Howto MB RL 002>`
- :ref:`BF-SL: Adaptive Functions <target_bf_sl_afct>`

