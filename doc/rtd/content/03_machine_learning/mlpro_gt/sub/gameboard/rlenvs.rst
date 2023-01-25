Reusing RL Environments
--------------------------

- **Transfering RL Environment to GT Game Board**

    In MLPro, we can simply transfer an RL environment to a GT game board by inheriting GameBoard functionality,
    as it is shown in the following:
    
    .. code-block:: python
    
        from mlpro.gt.models import *
        from mlpro.rl.pool.envs.dummy_environment import DummyEnv
        
        class MyGameBoard_GT(DummyEnv, GameBoard):
            C_NAME          = 'MyGameBoard_GT'

            def __init__(self, p_logging=True):
                DummyEnv.__init__(self, p_reward_type=Reward.C_TYPE_EVERY_AGENT)

- **Game board from Third Party Packages**

    Alternatively, if your environment follows Gym or PettingZoo interface, you can apply our
    relevant useful wrappers for the integration between third-party packages and MLPro. For more
    information, please click :ref:`here<target-package-third>`.
    Then, you need to transfer the wrapped RL environment to a GT Game Board.