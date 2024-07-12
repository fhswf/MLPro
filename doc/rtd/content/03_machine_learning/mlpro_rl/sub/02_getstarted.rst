.. _target_getstarted_RL:
Getting Started
---------------

Here is a concise series to introduce all users to the MLPro-RL in a practical way, whether you are a first-timer or an experienced MLPro user.

If you are a first-timer, then you can begin with **Section (1) What is MLPro?**.

If you have understood MLPro but not reinforcement learning, then you can jump to **Section (2) What is Reinforcement Learning?**.

If you have experience in both MLPro and reinforcement learning, then you can directly start with **Section (3) What is MLPro-RL?**.

After following the below step-by-step guideline, we expect the user understands the MLPro-RL in practice and starts using MLPro-RL.

**1. What is MLPro?**
   If you are a first-time user of MLPro, you might wonder what is MLPro.
   Therefore, we recommend initially start with understanding MLPro by checking out the following steps:

   (a) :ref:`MLPro: An Introduction <target_mlpro_introduction>`

   (b) `introduction video of MLPro <https://ars.els-cdn.com/content/image/1-s2.0-S2665963822001051-mmc1.mp4>`_

   (c) :ref:`installing and getting started with MLPro <target_mlpro_getstarted>`

   (d) `MLPro paper in Software Impact journal <https://doi.org/10.1016/j.simpa.2022.100421>`_

**2. What is Reinforcement Learning?**
   If you have not dealt with reinforcement learning, we recommend starting to understand at least the basic concept of reinforcement learning.
   There are plenty of references, articles, papers, books, or videos on the internet that explains reinforcement learning.
   But, for deep understanding, we recommend you to read the book from Sutton and Barto, which is `Reinforcement Learning: An Introduction <https://dl.acm.org/doi/10.5555/3312046>`_.

**3. What is MLPro-RL?**
   We expect that you have a basic knowledge of MLPro and reinforcement learning.
   Therefore, you need to understand the overview of MLPro-RL by following the steps below:

   (a) :ref:`MLPro-RL introduction page <target_overview_RL>`

   (b) `Section 4 of MLPro 1.0 paper <https://doi.org/10.1016/j.mlwa.2022.100341>`_

**4. Understanding Environment in MLPro-RL**
   First of all, it is important to understand the structure of an environment in MLPro, which can be found on :ref:`this page <target_env_RL>`.

   Then, you can start following some of our howto files related to the environment in MLPro-RL, as follows:

   (a) :ref:`Howto RL-001: Reward <Howto RL 001>`

   (b) `Howto RL-AGENT-001: Run an Agent with Own Policy <https://mlpro-int-gymnasium.readthedocs.io/en/latest/content/01_example_pool/01_howtos_rl/howto_rl_agent_001_run_agent_with_own_policy_on_gym_environment.html>`_

**5. Understanding Agent in MLPro-RL**
   In reinforcement learning, we have two types of agents, such as a single-agent RL or a multi-agent RL. Both of the types are covered by MLPro-RL.
   To understand the different possibilities of an agent in MLPro, you can visit :ref:`this page <target_agents_RL>`.

   Then, you need to understand how to set up a single-agent and a multi-agent RL in MLPro-RL by following these examples:

   (a) `Howto RL-AGENT-001: Run an Agent with Own Policy <https://mlpro-int-gymnasium.readthedocs.io/en/latest/content/01_example_pool/01_howtos_rl/howto_rl_agent_001_run_agent_with_own_policy_on_gym_environment.html>`_

   (b) `Howto RL-AGENT-003: Run Multi-Agent with Own Policy <https://mlpro-int-gymnasium.readthedocs.io/en/latest/content/01_example_pool/01_howtos_rl/howto_rl_agent_003_run_multiagent_with_own_policy_on_multicartpole_environment.html>`_

**6. Selecting between Model-Free and Model-Based RL**
   In this section, you need to select your direction of the RL training, whether it is a model-free RL or a model-based RL.
   However, firstly, you can pay attention to these two pages, which are :ref:`RL scenario <target_scenario_RL>` and :ref:`training <target_training_RL>`, before selecting either of the paths below.

   * Model-Free Reinforcement Learning

      To practice model-free RL in the MLPro-RL package, here are a video and some ready-to-use howto files that can be followed:

      (a) `A sample application video of MLPro-RL on a UR5 robot <https://ars.els-cdn.com/content/image/1-s2.0-S2665963822001051-mmc2.mp4>`_

      (b) `Howto RL-AGENT-002: Train an Agent with Own Policy <https://mlpro-int-gymnasium.readthedocs.io/en/latest/content/01_example_pool/01_howtos_rl/howto_rl_agent_002_train_agent_with_own_policy_on_gym_environment.html>`_

      (c) `Howto RL-AGENT-004: Train Multi-Agent with Own Policy <https://mlpro-int-gymnasium.readthedocs.io/en/latest/content/01_example_pool/01_howtos_rl/howto_rl_agent_004_train_multiagent_with_own_policy_on_multicartpole_environment.html>`_
   
   * Model-Based Reinforcement Learning

      Model-based RL contains two learning paradigms, such as learning the environment (model-based learning) and utilizing the model (e.g. as an action planner).
      To practice model-based RL in the MLPro-RL package, here are a howto file that can be followed:

      (a) `Howto RL-MB-001: Train and Reload Model Based Agent (Gymnasium) <https://mlpro-int-sb3.readthedocs.io/en/latest/content/01_example_pool/04_howtos_mb/howto_rl_mb_001_train_and_reload_model_based_agent_gym%20copy.html>`_

      (b) :ref:`Howto RL-MB-001: MBRL with MPC on Grid World Environment <Howto MB RL 001>`

      For more advanced MBRL technique, e.g. applying a native MBRL network, here is an example that can be used as a reference:
      
      (c) `Howto RL-MB-002: MBRL on RobotHTM Environment <https://mlpro-int-sb3.readthedocs.io/en/latest/content/01_example_pool/04_howtos_mb/howto_rl_mb_002_robothtm_environment.html>`_


**7. Additional Guidance**
   After following the previous steps, we hope that you could practice MLPro-RL and start using this subpackage for your RL-related activities.
   For more advanced features, we highly recommend you to check out the following howto files:

   (a) `Howto RL-AGENT-001: Train and Reload Single Agent (Gymnasium) <https://mlpro-int-sb3.readthedocs.io/en/latest/content/01_example_pool/01_howtos_agent/howto_rl_agent_001_train_and_reload_single_agent_gym.html>`_

   (b) `Howto RL-HT-001: Hyperparameter Tuning using Hyperopt <https://mlpro-int-hyperopt.readthedocs.io/en/latest/content/01_examples_pool/howto.rl.ht.001.html>`_

   (c) `Howto RL-HT-001: Hyperparameter Tuning using Optuna <https://mlpro-int-optuna.readthedocs.io/en/latest/content/01_examples_pool/howto.rl.ht.002.html>`_

   (d) `Howto RL-ATT-001: Train and Reload Single Agent using Stagnation Detection (Gymnasium) <https://mlpro-int-sb3.readthedocs.io/en/latest/content/01_example_pool/03_howtos_att/howto_rl_att_001_train_and_reload_single_agent_gym_sd.html>`_
