4.1 Overview
================

Game Theory (GT) is well-known in economic studies as a theoretical approach to model the strategic
interaction between multiple individuals or players in a specific situation. Game Theory
approach can also be adopted in the science area to optimize decision-making processes in a
strategic setting and often use to solve Multi-Agent RL (MARL) problems.

You can easily access the GT module, as follows:

    .. code-block:: python

        from fhswf_at_ml.gt.models import *

Some of developed RL frameworks in MLPro can also be reuse in the GT approach.
Thus we can just simply inherit some classes from RL frameworks, such as:

1. **GameBoard(rl.Environent)**
    Since you need a unique utility function for each specific player in the GT approach.
    A local utility function can be defined as below:

    .. code-block:: python
        
        import fhswf_at_ml.rl.models as rl
        
        class GameBoard(rl.Environment):
            """
            Model class for a game theoretical game board. See super class for more information.
            """
        
            C_TYPE              = 'Game Board'
            C_REWARD_TYPE       = rl.Reward.C_TYPE_EVERY_AGENT
            
            def compute_reward(self) -> rl.Reward:
                if self._last_action is None: return None
        
                reward = rl.Reward(self.get_reward_type())
        
                for player_id in self.last_action.get_agent_ids():
                    reward.add_agent_reward(player_id, self._utility_fct(player_id))
        
                return reward
        
            def _utility_fct(self, p_player_id):
                """
                Computes utility of given player. To be redefined.
                """
                
                return 0

2. **Player(rl.Agent)**

    .. code-block:: python
        
        import fhswf_at_ml.rl.models as rl
        
        class Player(rl.Agent):
            """
            This class implements a game theoretical player model. See super class for more information.
            """
        
            C_TYPE      = 'Player'

3. **Game(rl.Scenario)**

    .. code-block:: python
        
        import fhswf_at_ml.rl.models as rl
        
        class Game(rl.Scenario):
            """
            This class implements a game consisting of a game board and a (multi-)player. See super class for 
            more information.
            """
        
            C_TYPE      = 'Game'

4. **MultiPlayer(rl.MultiAgent)**

    .. code-block:: python
        
        import fhswf_at_ml.rl.models as rl
        
        class MultiPlayer(rl.MultiAgent):
            """
            This class implements a game theoretical model for a team of players. See super class for more 
            information.
            """
        
            C_TYPE      = 'Multi-Player'
        
            def add_player(self, p_player:Player, p_weight=1.0) -> None:
                super().add_agent(p_agent=p_player, p_weight=p_weight)

5. **Training(rl.Training)**

    .. code-block:: python
        
        import fhswf_at_ml.rl.models as rl
        
        class Training(rl.Training): 
            """
            This class implements a standardized episodical training process. See super class for more information.
            """
        
            C_NAME      = 'GT'

You can check out some of the examples on our :ref:`how to files<target-howto>`
or `here <https://github.com/fhswf/MLPro/tree/main/examples/gt>`_.