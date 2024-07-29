.. _target_getstarted_RL:
Getting Started
---------------

Here is a concise series designed to introduce all users to MLPro-RL in a practical manner, whether you are new to it or an experienced MLPro user.

No experience with MLPro? To learn more about MLPro, please refer to the :ref:`Getting Started page of MLPro <target_mlpro_getstarted>`.

By following the step-by-step guidelines below, we expect users to gain practical understanding of MLPro-RL and begin using it effectively.

**1. What is Reinforcement Learning?**
   If you are unfamiliar with reinforcement learning, we recommend starting with an understanding of its basic concepts.
   There are many references, articles, papers, books, and videos available online that explain reinforcement learning.
   For a deeper understanding, we recommend reading the book by Sutton and Barto, titled: `Reinforcement Learning: An Introduction <https://dl.acm.org/doi/10.5555/3312046>`_.

**2. What is MLPro-RL?**
   We assume you have a basic understanding of MLPro and reinforcement learning.
   Therefore, you should familiarize yourself with the overview of MLPro-RL by following these steps:

   (a) :ref:`MLPro-RL introduction page <target_overview_RL>`

   (b) `Section 4 of MLPro 1.0 paper <https://doi.org/10.1016/j.mlwa.2022.100341>`_

**3. Understanding Environment in MLPro-RL**
   Firstly, it is crucial to understand the structure of an environment in MLPro, which can be found on  :ref:`this page <target_env_RL>`.

   Next, you can refer to our how-to files related to the environment in MLPro-RL, listed below:

   (a) :ref:`Howto RL-001: Reward <Howto RL 001>`

   (b) `Howto RL-AGENT-001: Run an Agent with Own Policy <https://mlpro-int-gymnasium.readthedocs.io/en/latest/content/01_example_pool/01_howtos_rl/howto_rl_agent_001_run_agent_with_own_policy_on_gym_environment.html>`_

**4. Understanding Agent in MLPro-RL**
   In reinforcement learning, there are two types of agents: single-agent RL and multi-agent RL. Both types are supported by MLPro-RL.
   To explore the various possibilities for an agent in MLPro, you can visit: :ref:`this page <target_agents_RL>`.

   Next, you need to learn how to set up both single-agent and multi-agent RL in MLPro-RL by following these examples:

   (a) `Howto RL-AGENT-001: Run an Agent with Own Policy <https://mlpro-int-gymnasium.readthedocs.io/en/latest/content/01_example_pool/01_howtos_rl/howto_rl_agent_001_run_agent_with_own_policy_on_gym_environment.html>`_

   (b) `Howto RL-AGENT-003: Run Multi-Agent with Own Policy <https://mlpro-int-gymnasium.readthedocs.io/en/latest/content/01_example_pool/01_howtos_rl/howto_rl_agent_003_run_multiagent_with_own_policy_on_multicartpole_environment.html>`_

**5. Selecting between Model-Free and Model-Based RL**
   In this section, you need to choose your approach for RL training, deciding between model-free RL and model-based RL.
   However, before choosing between the options, please review these two pages: :ref:`RL scenario <target_scenario_RL>` and :ref:`training <target_training_RL>`, before selecting either of the paths below.

   * Model-Free Reinforcement Learning

      To practice model-free RL with the MLPro-RL package, you can refer to the following video and ready-to-use how-to files:

      (a) `A sample application video of MLPro-RL on a UR5 robot <https://ars.els-cdn.com/content/image/1-s2.0-S2665963822001051-mmc2.mp4>`_

      (b) `Howto RL-AGENT-002: Train an Agent with Own Policy <https://mlpro-int-gymnasium.readthedocs.io/en/latest/content/01_example_pool/01_howtos_rl/howto_rl_agent_002_train_agent_with_own_policy_on_gym_environment.html>`_

      (c) `Howto RL-AGENT-004: Train Multi-Agent with Own Policy <https://mlpro-int-gymnasium.readthedocs.io/en/latest/content/01_example_pool/01_howtos_rl/howto_rl_agent_004_train_multiagent_with_own_policy_on_multicartpole_environment.html>`_
   
   * Model-Based Reinforcement Learning

      Model-based RL involves two learning paradigms: learning the environment (model-based learning) and utilizing the model (e.g., as an action planner).
      To practice model-based RL with the MLPro-RL package, refer to the following how-to file:

      (a) `Howto RL-MB-001: Train and Reload Model Based Agent (Gymnasium) <https://mlpro-int-sb3.readthedocs.io/en/latest/content/01_example_pool/04_howtos_mb/howto_rl_mb_001_train_and_reload_model_based_agent_gym%20copy.html>`_

      (b) :ref:`Howto RL-MB-001: MBRL with MPC on Grid World Environment <Howto MB RL 001>`

      For more advanced MBRL techniques, such as using a native MBRL network, refer to the following example:
      
      (c) `Howto RL-MB-002: MBRL on RobotHTM Environment <https://mlpro-int-sb3.readthedocs.io/en/latest/content/01_example_pool/04_howtos_mb/howto_rl_mb_002_robothtm_environment.html>`_


**6. Additional Guidance**
   After completing the previous steps, we hope you will be able to practice with MLPro-RL and begin utilizing this subpackage for your RL-related activities.
   For more advanced features, we strongly recommend reviewing the following how-to files:

   (a) `Howto RL-AGENT-001: Train and Reload Single Agent (Gymnasium) <https://mlpro-int-sb3.readthedocs.io/en/latest/content/01_example_pool/01_howtos_agent/howto_rl_agent_001_train_and_reload_single_agent_gym.html>`_

   (b) `Howto RL-HT-001: Hyperparameter Tuning using Hyperopt <https://mlpro-int-hyperopt.readthedocs.io/en/latest/content/01_examples_pool/howto.rl.ht.001.html>`_

   (c) `Howto RL-HT-001: Hyperparameter Tuning using Optuna <https://mlpro-int-optuna.readthedocs.io/en/latest/content/01_examples_pool/howto.rl.ht.002.html>`_

   (d) `Howto RL-ATT-001: Train and Reload Single Agent using Stagnation Detection (Gymnasium) <https://mlpro-int-sb3.readthedocs.io/en/latest/content/01_example_pool/03_howtos_att/howto_rl_att_001_train_and_reload_single_agent_gym_sd.html>`_
