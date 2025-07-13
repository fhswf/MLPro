.. _target_training_RL:

Training and tuning
===================

In RL, training involves repeated interactions between an agent and an environment over multiple time steps.

**How RL Training Works**

    (1) Agent Observes the State → The agent receives information about the current state of the environment.
    
    (2) Agent Selects an Action → The agent chooses an action using its policy.
    
    (3) Environment Updates State → The environment transitions to a new state based on the action.
    
    (4) Agent Receives Reward → The environment returns a reward signal.
    
    (5) Policy Updates → The agent updates its policy using either:

        - Model-based learning (estimates environment dynamics)

        - Model-free learning (directly optimizes policy/value function)

    (6) Repeat Until Terminal State → This loop continues until an episode ends.

**Training in MLPro-RL**

In MLPro-RL, the **RLTraining** class inherits from the Training class at the basic function level.
This class is used for training RL agents and hyperparameter tuning.

Key Features of **RLTraining**:

    - Episodic Training → Training progresses through multiple episodes

    - Training Data Storage → Extended training data and results are stored in the file system

    - Support for Single & Multi-Agent RL → Easily train different types of agents
    
    - Stagnation Detection → Prevents unnecessary long training times without improvement

**Training Termination Conditions**

An RL training session in MLPro-RL continues until one of the following events occurs:

    (1) Event Success
        
        - The agent reaches the defined target state → Episode ends

    (2) Event Broken

        - The target state is no longer reachable → Episode ends

    (3) Event Timeout

        - The maximum training cycles are reached → Episode ends

If none of these events occur, training continues to maximize the score over repeated evaluations.


**Stagnation Detection in Training**

To prevent unnecessary long training sessions, MLPro-RL provides a `stagnation detection functionality <https://mlpro-int-sb3.readthedocs.io/en/latest/content/01_example_pool/03_howtos_att/howto_rl_att_001_train_and_reload_single_agent_gym_sd.html>`_.

If no further improvements are detected over time, training can be terminated early.

For more information, you can read `Section 4.3 of MLPro 1.0 paper <https://doi.org/10.1016/j.mlwa.2022.100341>`_.


**Simplifying RL Training with MLPro-RL**

MLPro-RL makes it easy to set up and train RL agents by automating the process.
Whether you are working with single-agent or multi-agent RL, MLPro-RL provides a structured and efficient training framework.

Next Step: Define your own RL scenario and start training your agent!
Here is an example for doing it:

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