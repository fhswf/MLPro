.. _target_scenario_RL:
Scenarios
------------

In RL, a scenario refers to a specific problem or task that an agent is trying to learn how to solve. It defines the environment in which the agent operates, including:

  - State Space – All possible states the environment can be in

  - Action Space – The set of actions the agent can take

  - Reward Function – A signal that quantifies the agent’s performance
  
  - Transition Dynamics – The rules governing how the environment evolves

**How an RL Scenario Works**

  (1) Interaction Over Time → The agent interacts with the environment step by step
  
  (2) State Observation → The agent receives information about the environment’s current state
  
  (3) Action Selection → The agent chooses an action based on its policy
  
  (4) State Transition → The environment updates its state based on the agent’s action
  
  (5) Reward Signal → The agent receives a reward based on its performance

The goal? Learn an optimal policy that maps states to actions in a way that maximizes cumulative rewards over time.

**RL Scenarios in MLPro**

In MLPro-RL, the **RLScenario** class inherits from **Scenario** (a basic function-level class), combining agents and environments into an executable unit.

Here are some of their key features:

  - Template-Based Setup – Users can easily define RL scenarios

  - Single & Multi-Agent Support – Easily switch between single-agent and multi-agent RL

  - Customizable Setup – Inherit RLScenario and redefine the **_setup** function

This flexibility allows users to implement and experiment with different RL problems effortlessly.

**Further Exploration**

MLPro-RL makes it easy to structure and run custom RL scenarios. You can:

  - Define an RL scenario using the RLScenario base class

  - Customize environments and agents to fit your task

  - Experiment with different policies and learning algorithms

Want to get started? Try defining your own RL scenario by inheriting RLScenario and implementing its **_setup** function!


**Cross reference**

  - :ref:`MLPro-RL: Training <target_training_RL>`
  - `Howto RL-AGENT-001: Run an agent with own policy <https://mlpro-int-gymnasium.readthedocs.io/en/latest/content/01_example_pool/01_howtos_rl/howto_rl_agent_001_run_agent_with_own_policy_on_gym_environment.html>`_
  - `Howto RL-AGENT-003: Run multi-agent with own policy <https://mlpro-int-gymnasium.readthedocs.io/en/latest/content/01_example_pool/01_howtos_rl/howto_rl_agent_003_run_multiagent_with_own_policy_on_multicartpole_environment.html>`_