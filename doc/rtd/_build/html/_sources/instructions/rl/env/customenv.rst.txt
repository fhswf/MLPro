Custom Environments
--------------

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
    
    To set up state- and action-spaces using our basic functionalities, please refer to our :ref:`how to File 02<target-howto>`
    or `here <https://github.com/fhswf/MLPro/blob/main/examples/bf/Howto%2002%20-%20(Math)%20Spaces%2C%20subspaces%20and%20elements.py>`_.
    Dimension class is currently improved and we will provide the explanation afterwards!

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
