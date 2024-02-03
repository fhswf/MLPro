Custom Solvers
""""""""""""""""""""""""""

- **Setup a Custom Solver**

    Setting up a custom solver in MLPro-GT-Native is very simple.
    There are one method to be defined and other two methods to be optionally defined, as following:

    .. code-block:: python
        
        from mlpro.gt.native.basics import *

        class MySolver(GTSolver):

            C_NAME      = 'MySolver'

            # Optional
            def _init_hyperparam(self, **p_param):
                """
                Implementation specific hyperparameters can be added here. Please follow these steps:
                a) Add each hyperparameter as an object of type HyperParam to the internal hyperparameter
                space object self._hyperparam_space
                b) Create hyperparameter tuple and bind to self._hyperparam_tuple
                c) Set default value for each hyperparameter

                Parameters
                ----------
                p_param : Dict
                    Further model specific hyperparameters, that are passed through constructor.
                """

                ...


            # Optional
            def _setup_solver(self):
                """
                A method to setup a solver. This needs to be redefined based on each policy, but remains
                optional.

                """

                ...

            # Mandatory
                def _compute_strategy(self, p_payoff:GTPayoffMatrix) -> GTStrategy:
                    """
                    A method to compute a strategy from the solver. This method needs to be redefined.

                    Parameters
                    ----------
                    p_payoff : GTPayoffMatrix
                        Payoff matrix of a specific player.

                    Returns
                    -------
                    GTStrategy
                        The computed strategy.

                    """
                    
                    ...

- **Algorithm Checker**

    A test script using a unit test to check the developed solvers will be available soon!

