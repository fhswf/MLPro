.. _target_getstarted_RL:

Getting started
---------------

This guide provides a structured introduction to MLPro-RL, catering to both newcomers and experienced MLPro users.

If you are new to MLPro, please refer to the :ref:`Getting Started page of MLPro <target_mlpro_getstarted>` to gain foundational knowledge before proceeding.

By following the step-by-step guidelines below, users will develop a practical understanding of MLPro-RL and learn to use it effectively.

**1. What is reinforcement learning?**
   Reinforcement Learning (RL) is a branch of machine learning where an agent learns optimal decision-making by interacting with an environment.
   The agent receives feedback in the form of rewards or penalties, guiding it toward maximizing cumulative rewards.

   Unlike supervised learning, which relies on labeled data, RL involves the agent exploring different actions and learning from their consequences.
   Key RL concepts include:

   - Exploration: Trying new actions to discover better strategies.

   - Exploitation: Choosing the best-known actions based on prior learning.

   Common applications of RL include robotics, game playing, and autonomous systems.
   For an in-depth understanding, we recommend reading the book by Sutton and Barto, titled: `Reinforcement Learning: An Introduction <https://dl.acm.org/doi/10.5555/3312046>`_.

**2. What is MLPro-RL?**
   If you are already familiar with MLPro and RL, the next step is understanding MLPro-RL. Start with:

   (a) :ref:`MLPro-RL introduction page <target_overview_RL>`

   (b) `Section 4 of MLPro 1.0 paper <https://doi.org/10.1016/j.mlwa.2022.100341>`_

**3. Understanding environments in MLPro-RL**
   The environment is a crucial component of RL. Begin by learning its structure in MLPro:
   
   - :ref:`Understanding environments in MLPro-RL <target_env_RL>`.

   For practical examples, refer to the following guides:

   (a) :ref:`Howto RL-001: Reward <Howto RL 001>`

   (b) `Howto RL-AGENT-001: Run an Agent with Own Policy <https://mlpro-int-gymnasium.readthedocs.io/en/latest/content/01_example_pool/01_howtos_rl/howto_rl_agent_001_run_agent_with_own_policy_on_gym_environment.html>`_

**4. Understanding agents in MLPro-RL**
   MLPro-RL supports both single-agent and multi-agent RL. Learn more here:
   
   - :ref:`Understanding agents in MLPro-RL <target_agents_RL>`.

   Then, explore how to set up different agent types:

   (a) `Howto RL-AGENT-001: Run an agent with own policy <https://mlpro-int-gymnasium.readthedocs.io/en/latest/content/01_example_pool/01_howtos_rl/howto_rl_agent_001_run_agent_with_own_policy_on_gym_environment.html>`_

   (b) `Howto RL-AGENT-003: Run multi-agent with own policy <https://mlpro-int-gymnasium.readthedocs.io/en/latest/content/01_example_pool/01_howtos_rl/howto_rl_agent_003_run_multiagent_with_own_policy_on_multicartpole_environment.html>`_

**5. Selecting between model-Free and model-based RL**
   Decide on your RL training approach by first reviewing these pages:
   
   - :ref:`RL scenario <target_scenario_RL>`

   - :ref:`training <target_training_RL>`

   * Model-free reinforcement rearning

      For hands-on experience with model-free RL in MLPro-RL, explore these resources:

      (a) `A sample application video of MLPro-RL on a UR5 robot <https://ars.els-cdn.com/content/image/1-s2.0-S2665963822001051-mmc2.mp4>`_

      (b) `Howto RL-AGENT-002: Train an Agent with Own Policy <https://mlpro-int-gymnasium.readthedocs.io/en/latest/content/01_example_pool/01_howtos_rl/howto_rl_agent_002_train_agent_with_own_policy_on_gym_environment.html>`_

      (c) `Howto RL-AGENT-004: Train Multi-Agent with Own Policy <https://mlpro-int-gymnasium.readthedocs.io/en/latest/content/01_example_pool/01_howtos_rl/howto_rl_agent_004_train_multiagent_with_own_policy_on_multicartpole_environment.html>`_
   
   * Model-based reinforcement learning

      Model-based RL involves learning the environment model and leveraging it for planning or decision-making. Explore these how-to guides:

      (a) `Howto RL-MB-001: Train and Reload Model Based Agent (Gymnasium) <https://mlpro-int-sb3.readthedocs.io/en/latest/content/01_example_pool/04_howtos_mb/howto_rl_mb_001_train_and_reload_model_based_agent_gym%20copy.html>`_

      (b) :ref:`Howto RL-MB-001: MBRL with MPC on Grid World Environment <Howto MB RL 001>`

      For more advanced MBRL techniques, such as using a native MBRL network, refer to the following example:
      
      (c) `Howto RL-MB-002: MBRL on RobotHTM Environment <https://mlpro-int-sb3.readthedocs.io/en/latest/content/01_example_pool/04_howtos_mb/howto_rl_mb_002_robothtm_environment.html>`_


**6. Additional guidance**
   After completing the above steps, you should be comfortable working with MLPro-RL. For further learning, consider these advanced topics:

   (a) `Howto RL-AGENT-001: Train and reload single agent (Gymnasium) <https://mlpro-int-sb3.readthedocs.io/en/latest/content/01_example_pool/01_howtos_agent/howto_rl_agent_001_train_and_reload_single_agent_gym.html>`_

   (b) `Howto RL-HT-001: Hyperparameter tuning using Hyperopt <https://mlpro-int-hyperopt.readthedocs.io/en/latest/content/01_examples_pool/howto.rl.ht.001.html>`_

   (c) `Howto RL-HT-001: Hyperparameter tuning using Optuna <https://mlpro-int-optuna.readthedocs.io/en/latest/content/01_examples_pool/howto.rl.ht.002.html>`_

   (d) `Howto RL-ATT-001: Train and reload single agent using stagnation detection (Gymnasium) <https://mlpro-int-sb3.readthedocs.io/en/latest/content/01_example_pool/03_howtos_att/howto_rl_att_001_train_and_reload_single_agent_gym_sd.html>`_

By following this guide, you will be well-equipped to integrate MLPro-RL into your reinforcement learning projects.