.. _target_training_RL:
Training and Tuning
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
In this case, a :ref:`stagnation detection functionality <Howto RL ATT 001>` can be incorporated to avoid a long training time without any more improvements.
The training can be ended, once the stagnation is detected. For more information, you can read `Section 4.3 of MLPro 1.0 paper <https://doi.org/10.1016/j.mlwa.2022.100341>`_.

In MLPro-RL, we simplify the process of setting up an RL scenario and training for both single-agent and multi-agent RL, as shown below:

- **Single-Agent Scenario Creation**

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

- **Multi-Agent Scenario Creation**

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


**Cross Reference**

- `A sample application video of MLPro-RL on a UR5 robot <https://ars.els-cdn.com/content/image/1-s2.0-S2665963822001051-mmc2.mp4>`_
- :ref:`Howto RL-AGENT-002: Train an Agent with Own Policy <Howto Agent RL 002>`
- :ref:`Howto RL-AGENT-004: Train Multi-Agent with Own Policy <Howto Agent RL 004>`
- :ref:`Howto RL-AGENT-011: Train and Reload Single Agent (Gym) <Howto Agent RL 011>`
- :ref:`Howto RL-AGENT-021: Train and Reload Single Agent (MuJoCo) <Howto Agent RL 021>`
- :ref:`Howto RL-ATT-001: Train and Reload Single Agent using Stagnation Detection (Gym) <Howto RL ATT 001>`
- :ref:`Howto RL-ATT-002: Train and Reload Single Agent using Stagnation Detection (MuJoCo) <Howto RL ATT 002>`
- :ref:`Howto RL-MB-001: MBRL on RobotHTM Environment <Howto MB RL 001>`
- :ref:`Howto RL-MB-002: MBRL with MPC on Grid World Environment <Howto MB RL 002>`
- :ref:`MLPro-BF-ML: Training and Tuning <target_bf_ml_train_and_tune>`