Payoff Matrix
""""""""""""""""""""""""""

- **Setup a Payoff Matrix**

    Setting up a payoff matrix for a game in MLPro-GT-Native is very simple.
    There are three main steps, as following:

    .. code-block:: python

        from mlpro.gt.native.basics import *

        class MyPayoff(GTFunction):

            def _setup_mapping_matrix(self) -> np.ndarray:
                # Step 1
                ...

            def _setup_payoff_matrix(self):
                # Step 2
                ...

        class MyGame(GTGame):

            C_NAME  = 'MyGame'

            def _setup(self, p_mode, p_ada:bool, p_visualize:bool, p_logging) -> Model:
                
                ...

                # Step 3
                self._payoff = GTPayoffMatrix(
                    p_function=MyPayoff(
                        p_func_type=GTFunction.C_FUNC_PAYOFF_MATRIX,
                        ...
                        ),
                    p_player_ids=coal_ids
                )

                ...

- **Prerequisites**
    
    - To set up a game, please refer to :ref:`games page <target_native_games_page>`.
    
    - To set up players, coalitions, and a competition, please refer to :ref:`player, coalition, competition page <target_native_pl_page>`.