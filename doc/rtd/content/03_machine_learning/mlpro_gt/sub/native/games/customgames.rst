Custom Games
""""""""""""""""""""""""""

- **Prerequisites**
    
    - To set up a player, coalition, and competition, please refer to :ref:`player, coalition, and competition page <target_native_pl_page>`.
    
    - To set up a payoff function or matrix, please refer to :ref:`payoff page <target_native_payoff_page>`.
    
    - To set up a solver, please refer to :ref:`solvers page <target_native_solvers_page>`.

- **Setup a Custom Game**

    If you already know how to set up coalition/competition, payoff funtion/matrix, and solver. 
    Then, setting up a custom game in MLPro-GT-Native is very simple, where you only need to bring everything together in one method, as follows:

    .. code-block:: python
        
        from mlpro.gt.native.basics import *

        class MyGame(GTGame):

            C_NAME  = 'MyGame'

            def _setup(self, p_mode, p_ada:bool, p_visualize:bool, p_logging) -> Model:
                
                # Setup strategy space
                ...
                
                # Setup a solver, e.g. Random Solver
                ...

                # Setup a set of players
                ...

                # Setup a set of coalitions
                ...

                # Setup a competition (if required)
                ...

                # Setup the payoff matrix or function
                self._payoff = ...

                return Model
