4.5 Training
--------------

Add text here!

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