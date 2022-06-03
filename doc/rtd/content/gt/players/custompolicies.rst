Custom Policies
-------------------

- **Policy Creation**


    .. image:: images/MLPro-RL-Agent_Class_Policy_Commented.png
    
    To create a GT policy that satisfies MLPro interface is pretty straight forward.
    You just require to assure that the GT policy consists at least the following 3 main functions:

    .. code-block:: python
        
        from mlpro.rl.models import *
        
        class MyPolicy(Policy):
            """
            Creates a policy that satisfies mlpro interface.
            """
            C_NAME          = 'MyPolicy'

            def _init_hyperparam(self, **p_par):
                """
                Implementation specific hyperparameters can be added here. Please follow these steps:
                a) Add each hyperparameter as an object of type HyperParam to the internal hyperparameter
                   space object self._hyperparam_space
                b) Create hyperparameter tuple and bind to self._hyperparam_tuple
                c) Set default value for each hyperparameter
        
                Parameters
                ----------
                p_par : Dict
                    Futher model specific parameters, that are passed through constructor.
        
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
    
    To set up a hyperparameter space, please refer `this how to file <https://github.com/fhswf/MLPro/blob/main/examples/bf/Howto%2004%20-%20(Data)%20Store%2C%20plot%2C%20and%20save%20variables.py>`_.

- **Algorithm Checker**

    A test script using unittest to check the develop policies will be available soon!