Custom Policies
-------------------

- **Policy Creation**


    .. image:: images/MLPro-RL-Agent_Class_Policy_Commented.png
    
    Creating a GT policy that satisfies the MLPro interface is pretty straightforward.
    You just require to assure that the GT policy consists of at least the following 3 main functions:

    .. code-block:: python
        
        from mlpro.rl.models import *
        
        class MyPolicy(Policy):
            """
            Creates a policy that satisfies the mlpro interface.
            """
            C_NAME          = 'MyPolicy'

            def _init_hyperparam(self, **p_par):
                """
                Implementation-specific hyperparameters can be added here. Please follow these steps:
                a) Add each hyperparameter as an object of type HyperParam to the internal hyperparameter
                   space object self._hyperparam_space
                b) Create a hyperparameter tuple and bind it to self._hyperparam_tuple
                c) Set the default value for each hyperparameter
        
                Parameters
                ----------
                p_par : Dict
                    Further model-specific parameters, that are passed through the constructor.
        
                """
                ....

            def compute_action(self, p_obs: State) -> Action:
                """
                Specific action computation method to be redefined. 
    
                Parameters
                ----------
                p_obs : State
                    Observation data.
    
                Returns
                -------
                action : Action
                    Action object.
    
                """
                ....
    
            def _adapt(self, *p_args) -> bool:
                """
                Adapts the policy based on State-Action-Reward-State (SARS) data.
    
                Parameters
                ----------
                p_arg[0] : SARSElement
                    Object of type SARSElement.
    
                Returns
                -------
                adapted : bool
                    True, if something has been adapted. False otherwise.
    
                """
                ....
    
    To set up a hyperparameter space, please refer to :ref:`Howto BF-ML-010: Hyperparameters <Howto BF ML 010>`.

- **Algorithm Checker**

    A test script using the unit test to check the developed policies will be available soon!