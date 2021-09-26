Reinforcement Learning
========

RL Environment
-----------------------------------

- **Environment Creation for Simulation Mode**

    To create an environment that satisfies MLPro interface is immensly simple and straigtforward.
    Basically a MLPro environment is a class with 5 main functions. Each environment must apply the
    following mlpro functions:
    
    .. code-block:: python
        
        from mlpro.rl.models import *
        
        class MyEnvironment(Environment):
            """
            Custom Environment that satisfies mlpro interface.
            """
            C_NAME          = 'MyEnvironment'
            C_LATENCY       = timedelta(0,1,0)         # Default latency 1s
            C_REWARD_TYPE   = Reward.C_TYPE_OVERALL    # Default reward type
            
            def __init__(self, p_mode=C_MODE_SIM, p_latency:timedelta=None, p_logging=True):
                """
                Parameters:
                    p_mode              Mode of environment (simulation/real)
                    p_latency           Optional: latency of environment. If not provided
                                        internal value C_LATENCY will be used by default
                    p_logging           Boolean switch for logging
                """
        
                super().__init__(p_latency=p_latency, p_logging=p_logging)
                self._setup_spaces()
                self.set_mode(p_mode)
            
            def _setup_spaces(self):
                """
                Implement this method to enrich the state and action space with specific 
                dimensions. 
                """
        
                # Setup state space example
                # self.state_space.add_dim(Dimension(0, 'Pos', 'Position', '', 'm', 'm', [-50,50]))
                # self.state_space.add_dim(Dimension(1, 'Vel', 'Velocity', '', 'm/sec', '\frac{m}{sec}', [-50,50]))
        
                # Setup action space example
                # self.action_space.add_dim(Dimension(0, 'Rot', 'Rotation', '', '1/sec', '\frac{1}{sec}', [-50,50]))
                ....
            
            def _simulate_reaction(self, p_action:Action) -> None:
                """
                Simulates a state transition of the environment based on a new action.
                Please use method set_state() for internal update.
        
                Parameters:
                    p_action      Action to be processed
                """
                ....
                
            def reset(self) -> None:
                """
                Resets environment to initial state.
                """
                ....
                
            def compute_reward(self) -> Reward:
                """
                Computes a reward.
        
                Returns:
                  Reward object
                """
                ....
            
            def _evaluate_state(self) -> None:
                """
                Updates the goal achievement value in [0,1] and the flags done and broken
                based on the current state.
                """
                
                # state evaluations example
                # if self.done:
                #     self.goal_achievement = 1.0
                # else:
                #     self.goal_achievement = 0.0
                ....
    
    One of the benefits for MLPro users is the variety of reward structures, which is useful for Multi-Agent RL
    and Game Theoretical approach. Three types of reward structures are supported in this framework, such as:
    
    1. **C_TYPE_OVERALL** as the default type and is a scalar overall value
    
    2. **C_TYPE_EVERY_AGENT** is a scalar for every agent
    
    3. **C_TYPE_EVERY_ACTION** is a scalar for every agent and action.
    
    To set up state- and action-spaces using our basic functionalities, please refer to our :ref:`how to File 02<target-howto-bf>`
    or `here <https://github.com/fhswf/MLPro/blob/main/examples/bf/Howto%2002%20-%20(Math)%20Spaces%2C%20subspaces%20and%20elements.py>`_.
    Dimension class is currently improved and we will provide the explanation afterwards!
    
    We highly recommend you to check out our :ref:`how to files<target-howto-rl>` and our
    :ref:`pre-built environments<target-env-pool>`.

- **Environment Creation for Real Hardware Mode**

    In MLPro, we can choose simulation mode or real hardward mode. For real hardware mode, the creation of
    an environment is very similar to simulation mode. You do not need to define **_simulate_reaction**, but you
    need to replace it with **_export_action** and **_import_state** as it is shown in the following:
    
    .. code-block:: python
        
        from mlpro.rl.models import *
        
        class MyEnvironment(Environment):
            """
            Custom Environment that satisfies mlpro interface.
            """
            C_NAME          = 'MyEnvironment'
            C_LATENCY       = timedelta(0,1,0)         # Default latency 1s
            C_REWARD_TYPE   = Reward.C_TYPE_OVERALL    # Default reward type
            
            def __init__(self, p_mode=C_MODE_REAL, p_latency:timedelta=None, p_logging=True):
                """
                Parameters:
                    p_mode              Mode of environment (simulation/real)
                    p_latency           Optional: latency of environment. If not provided
                                        internal value C_LATENCY will be used by default
                    p_logging           Boolean switch for logging
                """
        
                super().__init__(p_latency=p_latency, p_logging=p_logging)
                self._setup_spaces()
                self.set_mode(p_mode)
            
            def _setup_spaces(self):
                """
                Implement this method to enrich the state and action space with specific 
                dimensions. 
                """
        
                # Setup state space example
                # self.state_space.add_dim(Dimension(0, 'Pos', 'Position', '', 'm', 'm', [-50,50]))
                # self.state_space.add_dim(Dimension(1, 'Vel', 'Velocity', '', 'm/sec', '\frac{m}{sec}', [-50,50]))
        
                # Setup action space example
                # self.action_space.add_dim(Dimension(0, 'Rot', 'Rotation', '', '1/sec', '\frac{1}{sec}', [-50,50]))
                ....
    
            def _export_action(self, p_action:Action) -> bool:
                """
                Exports given action to be processed externally (for instance by a real hardware).
        
                Parameters:
                    p_action      Action to be exported
        
                Returns:
                    True, if action export was successful. False otherwise.
                """
                ....

            def _import_state(self) -> bool:
                """
                Imports state from an external system (for instance a real hardware). 
                Please use method set_state() for internal update.
        
                Returns:
                  True, if state import was successful. False otherwise.
                """
                ....
                
            def reset(self) -> None:
                """
                Resets environment to initial state.
                """
                ....
    
            def compute_reward(self) -> Reward:
                """
                Computes a reward.
        
                Returns:
                  Reward object
                """
                ....
            
            def _evaluate_state(self) -> None:
                """
                Updates the goal achievement value in [0,1] and the flags done and broken
                based on the current state.
                """
                
                # state evaluations example
                # if self.done:
                #     self.goal_achievement = 1.0
                # else:
                #     self.goal_achievement = 0.0
                ....

- **Environment from Third Party Packages**

    Alternatively, if your environment follows Gym or PettingZoo interface, you can apply our
    relevant useful wrappers for the integration between third party packages and MLPro. For more
    information, please click :ref:`here<target-package>`.

- **Environment Checker**

    To check whether your developed environment is compatible to MLPro interface, we provide a test script
    using unittest. At the moment, you can find the source code `here <https://github.com/fhswf/MLPro/blob/main/test/test_environment.py>`_.
    We will prepare a built-in testing module in MLPro, show you how to excecute the testing soon and provides an example as well.

RL Algorithm
-----------------------------------

- **Policy Creation**

    To create a RL policy that satisfies MLPro interface is pretty direct.
    You just require to assure that the RL policy consists at least these following 3 main functions:

    .. code-block:: python
        
        from mlpro.rl.models import *
        
        class MyPolicy(Policy):
            """
            Creates a policy that satisfies mlpro interface.
            """
            C_NAME          = 'MyPolicy'
            
            def __init__(self, p_state_space:MSpace, p_action_space:MSpace, p_ada=True, p_logging=True):
                """
                 Parameters:
                    p_state_space       State space object
                    p_action_space      Action space object
                    p_ada               Boolean switch for adaptivity
                    p_logging           Boolean switch for logging functionality
                """
        
                super().__init__(p_ada=p_ada, p_logging=p_logging)
                self._state_space   = p_state_space
                self._action_space  = p_action_space
                self.set_id(0)
                
            def adapt(self, *p_args) -> bool:
                """
                Adapts the policy based on State-Action-Reward (SAR) data that will be expected as a SAR
                buffer object. Please call super-method at the beginning of your own implementation and
                adapt only if it returns True.
        
                Parameters:
                    p_arg[0]            SAR Buffer object
                """
        
                if not super().adapt(*p_args): return False
                
                ....
                return True
            
            def clear_buffer(self):
                """
                Intended to clear internal temporary attributes, buffers, ... Can be used while training
                to prepare the next episode.
                """
                ....
                
            def compute_action(self, p_state:State) -> Action:
                """
                Specific action computation method to be redefined. 
        
                Parameters:
                    p_state       State of environment
        
                Returns:
                    Action object
                """
                ....
    
    This class represents the policy of a single-agent. It is adaptive and can be trained with
    State-Action-Reward (SAR) data that will be expected as a SAR buffer object. 
    
    The three main learning paradigms of machine learning to train a policy are supported:

    1. Training by Supervised Learning: The entire SAR data set inside the SAR buffer shall be adapted.

    2. Training by Reinforcement Learning: The latest SAR data record inside the SAR buffer shall be adapted.

    3. Training by Unsupervised Learning: All state data inside the SAR buffer shall be adapted.

    Furthermore a policy class can compute actions from states.

    Hyperparameters of the policy should be stored in the internal object **self._hp_list**, so that
    they can be tuned from outside. Optionally a policy-specific callback method can be called on 
    changes. For more information see class HyperParameterList.
    
    To set up a hyperparameter space, please refer to our :ref:`how to File 04<target-howto-bf>`
    or `here <https://github.com/fhswf/MLPro/blob/main/examples/bf/Howto%2004%20-%20(ML)%20Hyperparameters%20setup.py>`_.

- **Policy from Third Party Packages**

    In addition, we are planning to reuse Ray RLlib in the near future. For more updates,
    please click :ref:`here<target-package>`.

- **Algorithm Checker**

    A test script using unittest to check the develop policies will be available soon!


RL Scenario
-----------------------------------

Scenario is where the interaction between RL agent(s) and an environment with a unique
specific settings takes place. One of the MLPro's features is enabling the user to apply
a template class for an RL scenario consisting of an environment and an agent/agents.
Moreover, you can create eihter single-agent scenario or multi-agent scenario in a simple
manner.

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