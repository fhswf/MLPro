Multi-agents
------------

Multi-agent reinforcement learning (MARL) extends RL to scenarios where multiple independent agents interact with each other and their environment to achieve a common goal or optimize their own individual rewards.

Unlike single-agent RL, where an agent's decisions are based solely on its own observations and actions, multi-agent interactions introduce complexity, as each agent’s behavior depends on:

    - Its own actions and observations

    - The actions and observations of other agents

This dynamic interdependence makes cooperation, competition, and adaptation key challenges in multi-agent RL.

**Multi-Agent RL in MLPro**

MLPro-RL supports both single-agent and multi-agent RL, providing a structured approach to managing multiple agents within an environment.

Here are some key characteristics of multi-agent RL in MLPro:

    - Multi-Agent Model → Combines multiple single agents into a cohesive system

    - Independent Agent Policies → Each agent can have its own policy

    - Separate Observation & Action Spaces → Each agent operates within a unique portion of the multi-agent environment

    - Scalar Reward Per Agent → Each agent receives individual feedback on performance

    - Native & Third-Party Environments → Compatible with MLPro environments and PettingZoo environments (via :ref:`wrapper class<target_extension_hub>`)

**Cross reference**
    - `Howto RL-AGENT-004: Train multi-agent with own policy <https://mlpro-int-gymnasium.readthedocs.io/en/latest/content/01_example_pool/01_howtos_rl/howto_rl_agent_004_train_multiagent_with_own_policy_on_multicartpole_environment.html>`_
    - :ref:`MLPro-RL: Training <target_training_RL>`