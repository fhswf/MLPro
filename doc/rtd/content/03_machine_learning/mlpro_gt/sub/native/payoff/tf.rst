Transfer Function
""""""""""""""""""""""""""

- **Setup a Transfer Function for Payoffs**

    Setting up a transfer function for payoffs in a game in MLPro-GT-Native is very simple.
    There are four main steps, as following:

    .. code-block:: python

        from mlpro.gt.native.basics import *
        from mlpro.bf.physics.basics import *

        class MyTransferFunction(TransferFunction):

            def _set_function_parameters(self, p_args) -> bool:
                # Step 1.1
                ...

            def _custom_function(self, p_input, p_range=None):
                # Step 1.2 (Optional)
                ...

        class MyPayoff(GTFunction):

            def _setup_transfer_functions(self):
                # Step 2
                TF = MyTransferFunction(...)

                self._add_transfer_function(p_idx=0, p_transfer_fct=TF)
                self._add_transfer_function(p_idx=1, p_transfer_fct=TF)
                ...

        class MyPayoffMatrix(GTPayoffMatrix):

            def _call_mapping(self, p_input:str, p_strategies:GTStrategy) -> float:
                # Step 3.1
                ...

            def _call_best_response(self, p_element_id:str) -> float:
                # Step 3.2
                ...

            def _call_zero_sum(self) -> bool:
                # Step 3.3
                ...


        class MyGame(GTGame):

            C_NAME  = 'MyGame'

            def _setup(self, p_mode, p_ada:bool, p_visualize:bool, p_logging) -> Model:
                
                ...

                # Step 4
                self._payoff = MyPayoffMatrix(
                    p_function=MyPayoff(
                        p_func_type=GTFunction.C_FUNC_TRANSFER_FCTS
                        ),
                    p_player_ids=coal_ids
                )

                ...

- **Prerequisites**
    
    - To set up a transfer function, please refer to :ref:`transfer function page <target_basics_physics_tf>`.
    
    - To set up a game, please refer to :ref:`games page <target_native_games_page>`.
    
    - To set up players, coalitions, and a competition, please refer to :ref:`player, coalition, competition page <target_native_pl_page>`.