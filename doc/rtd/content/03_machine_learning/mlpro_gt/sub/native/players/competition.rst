Competition
""""""""""""""""""""""""""

- **Setup a Competition**

    Setting up a competition for a game in MLPro-GT-Native is very simple, as following:

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
                coal1 = GTCoalition(...)
                coal2 = GTCoalition(...)
                ...

                # Setup a competition
                competition = GTCompetition(
                    p_name="Prisoner's Dilemma Competition",
                    p_logging=p_logging
                    )
                competition.add_coalition(coal1)
                competition.add_coalition(coal2)
                ...

                ...

    If you only have a competition between individual players, you can simply set up a set of coalitions with one player in each coalition.
    This configuration also allows a competition between players and coalitions.

- **Prerequisites**
    
    - To set up a game, please refer to :ref:`games page <target_native_games_page>`.
    
    - To set up a solver, please refer to :ref:`solvers page <target_native_solvers_page>`.
    
    - To set up a player, please refer to :ref:`setup player page <target_native_gt_setup_player>`.
    
    - To set up a coalition, please refer to :ref:`setup coalition page <target_native_gt_setup_coal>`.
