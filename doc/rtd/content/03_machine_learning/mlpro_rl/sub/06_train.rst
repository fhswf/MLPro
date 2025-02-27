.. _target_training_RL:
Training and tuning
===================

In RL, the agent and the environment interact over a sequence of time steps.
At each time step, the agent receives an observation of the current state of the environment and selects an action.
The environment then transitions to a new state and returns a reward signal to the agent.
This process continues until some terminal state is reached.

The agent uses the observed state-action-reward sequences to update its policy,
either through model-based methods that estimate the underlying dynamics of the environment,
or model-free methods that directly estimate the value or the policy.
The policy is used to select actions in subsequent interactions with the environment, allowing the agent to learn from its mistakes and improve over time.

In MLPro-RL, a class **RLTraining** inherits the functionality from class **Training** in the basic function level, where the **RLTraining** class are used for training and hyperparameter tuning of RL agents.
We implement episodic training algorithms and make the corresponding extended training data and results as well as the trained agents available in the file system.
In this RL training, we always start with a defined random initial state of the environment and evaluate at each time step whether one of the following three categories is satisfied,

    (1) **Event Success**: This means that the defined target state is reached and the actual episode is ended.

    (2) **Event Broken**: This means that the defined target state is no longer reachable and the actual episode is ended.

    (3) **Event Timeout**: This means that the maximum training cycles for an episode are reached and the actual episode is ended.

If none of the events is satisfied, then the training continues. The goal of the training is to maximize the score of the repetitive evaluations.
In this case, a `stagnation detection functionality <https://mlpro-int-sb3.readthedocs.io/en/latest/content/01_example_pool/03_howtos_att/howto_rl_att_001_train_and_reload_single_agent_gym_sd.html>`_ can be incorporated to avoid a long training time without any more improvements.
The training can be ended, once the stagnation is detected. For more information, you can read `Section 4.3 of MLPro 1.0 paper <https://doi.org/10.1016/j.mlwa.2022.100341>`_.

In MLPro-RL, we simplify the process of setting up an RL scenario and training for both single-agent and multi-agent RL, as shown below:

- **Single-agent scenario creation**

    .. code-block:: python
        
        from mlpro.rl.models import *
        
        class MyScenario(Scenario):

            C_NAME      = 'MyScenario'
            
            def _setup(self, p_mode, p_ada:bool, p_logging:bool):
                """
                Here's the place to explicitely setup the entire rl scenario. Please bind your env to
                self._env and your agent to self._agent. 
        
                Parameters:
                    p_mode              Operation mode of environment (see Environment.C_MODE_*)
                    p_ada               Boolean switch for adaptivity of agent
                    p_logging           Boolean switch for logging functionality
               """
        
               # Setup environment
               self._env    = MyEnvironment(....)
               
               # Setup an agent with selected policy
               self._agent = Agent(
                   p_policy=MyPolicy(
                    p_state_space=self._env.get_state_space(),
                    p_action_space=self._env.get_action_space(),
                    ....
                    ),
                    ....
                )
        
        # Instantiate scenario
        myscenario  = MyScenario(p_scenario=myscenario, ....)
        
        # Train agent in scenario
        training    = Training(....)
        training.run()

- **Multi-agent scenario creation**

    .. code-block:: python
        
        from mlpro.rl.models import *
        
        class MyScenario(Scenario):

            C_NAME      = 'MyScenario'
            
            def _setup(self, p_mode, p_ada:bool, p_logging:bool):
                """
                Here's the place to explicitely setup the entire rl scenario. Please bind your env to
                self._env and your agent to self._agent. 
        
                Parameters:
                    p_mode              Operation mode of environment (see Environment.C_MODE_*)
                    p_ada               Boolean switch for adaptivity of agent
                    p_logging           Boolean switch for logging functionality
               """
        
               # Setup environment
               self._env    = MyEnvironment(....)
               
               # Create an empty mult-agent
               self._agent     = MultiAgent(....)
               
               # Add Single-Agent #1 with own policy (controlling sub-environment #1)
               self._agent.add_agent = Agent(
                   self._agent = Agent(
                       p_policy=MyPolicy(
                        p_state_space=self._env.get_state_space().spawn[....],
                        p_action_space=self._env.get_action_space().spawn[....],
                        ....
                        ),
                        ....
                    ),
                    ....
                )
               
               # Add Single-Agent #2 with own policy (controlling sub-environment #2)
               self._agent.add_agent = Agent(....)
               
               ....
        
        # Instantiate scenario
        myscenario  = MyScenario(p_scenario=myscenario, ....)
        
        # Train agent in scenario
        training    = Training(....)
        training.run()


**Cross reference**

    - `A sample application video of MLPro-RL on a UR5 robot <https://ars.els-cdn.com/content/image/1-s2.0-S2665963822001051-mmc2.mp4>`_
    - `Howto RL-AGENT-002: Train an agent with own policy <https://mlpro-int-gymnasium.readthedocs.io/en/latest/content/01_example_pool/01_howtos_rl/howto_rl_agent_002_train_agent_with_own_policy_on_gym_environment.html>`_
    - `Howto RL-AGENT-004: Train multi-agent with own policy <https://mlpro-int-gymnasium.readthedocs.io/en/latest/content/01_example_pool/01_howtos_rl/howto_rl_agent_004_train_multiagent_with_own_policy_on_multicartpole_environment.html>`_
    - `Howto RL-AGENT-001: Train and reload single agent (Gymnasium) <https://mlpro-int-sb3.readthedocs.io/en/latest/content/01_example_pool/01_howtos_agent/howto_rl_agent_001_train_and_reload_single_agent_gym.html>`_
    - `Howto RL-ATT-001: Train and reload single agent using stagnation detection (Gymnasium) <https://mlpro-int-sb3.readthedocs.io/en/latest/content/01_example_pool/03_howtos_att/howto_rl_att_001_train_and_reload_single_agent_gym_sd.html>`_
    - `Howto RL-MB-001: Train and reload model-based agent (Gymnasium) <https://mlpro-int-sb3.readthedocs.io/en/latest/content/01_example_pool/04_howtos_mb/howto_rl_mb_001_train_and_reload_model_based_agent_gym%20copy.html>`_
    - :ref:`Howto RL-MB-001: MBRL with MPC on Grid World environment <Howto MB RL 001>`
    - :ref:`MLPro-BF-ML: Training and tuning <target_bf_ml_train_and_tune>`