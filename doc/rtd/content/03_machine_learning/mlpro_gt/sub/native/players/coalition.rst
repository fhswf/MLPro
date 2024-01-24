.. _target_native_gt_setup_coal:
Coalition
""""""""""""""""""""""""""

- **Setup a Coalition**

    Setting up a colation for a game in MLPro-GT-Native is very simple, as following:

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
                p1 = GTPlayer(...)
                p2 = GTPlayer(...)
                ...

                # Setup a coalition
                coal1 = GTCoalition(
                    p_name="Coalition 1",
                    p_coalition_type=GTCoalition.C_COALITION_SUM
                )
                coal1.add_player(p1)
                coal1.add_player(p2)
                ...

- **Prerequisites**
    
    - To set up a game, please refer to :ref:`games page <target_native_games_page>`.
    
    - To set up a solver, please refer to :ref:`solvers page <target_native_solvers_page>`.
    
    - To set up a player, please refer to :ref:`setup player page <target_native_gt_setup_player>`.