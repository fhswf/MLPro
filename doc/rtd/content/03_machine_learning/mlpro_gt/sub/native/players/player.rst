.. _target_native_gt_setup_player:
Player
""""""""""""""""""""""""""

- **Setup a Player**

    Setting up a player for a game in MLPro-GT-Native is very simple, as following:

    .. code-block:: python
        
        from mlpro.gt.native.basics import *
        from mlpro.gt.pool.native.solvers.randomsolver import RandomSolver # Optional

        class MyGame(GTGame):

        C_NAME  = 'MyGame'

            def _setup(self, p_mode, p_ada:bool, p_visualize:bool, p_logging) -> Model:
                
                # Setup strategy space
                _strategy_space = MSpace()
                _strategy_space.add_dim(Dimension('RStr','Z','Random Strategy','','','',[0,1]))
                
                # Setup a solver, e.g. Random Solver
                solver1 = RandomSolver(
                    p_strategy_space=_strategy_space,
                    p_id=1,
                    p_name="Random Solver",
                    p_visualize=p_visualize,
                    p_logging=p_logging
                )

                # Setup a player
                p1 = GTPlayer(
                    p_solver=solver1,
                    p_name="Player of Prisoner 1",
                    p_visualize=p_visualize,
                    p_logging=p_logging,
                    p_random_solver=False
                )

                ...

- **Prerequisite**
    
    To set up a solver, please refer to :ref:`solvers page in MLPro-GT-Native section <target_native_solvers_page>`.