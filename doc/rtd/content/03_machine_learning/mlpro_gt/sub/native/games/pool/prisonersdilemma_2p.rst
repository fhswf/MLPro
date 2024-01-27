2P Prisoners’ Dilemma
^^^^^^^^^^^^^^^^^^^^^^^^^

**General Information**

The Prisoner's Dilemma is a classic concept in game theory that describes a situation where two players must decide whether to cooperate or betray each other, with the outcomes determined by the combined choices made by both players.
In a two-player Prisoner's Dilemma game, the players face a payoff matrix showing possible outcomes and associated payoffs for each player based on their choices.

Each player has two possible choices: "Confess" or "Not Confess".
The players make their decisions independently without knowing the other player's choice.
The payoffs for each player depend on both their choice and the choice made by the other player, as outlined in the payoff matrix.

The four possible outcomes in a two-player Prisoner's Dilemma game are (Confess, Confess), (Confess, Not Confess), (Not Confess, Confess), and (Not Confess, Not Confess).
The dilemma arises because, individually, each player has an incentive to betray the other to achieve a higher payoff, regardless of the other player's choice.

The Nash equilibrium in this game is typically the outcome where both players betray each other, as neither player can unilaterally change their strategy to improve their payoff.
However, from a collective perspective, both players cooperating would lead to a better overall outcome.
The tension lies in the fact that betraying is the dominant strategy for each player, even though the jointly optimal outcome would be cooperation.

The Prisoner's Dilemma is used to illustrate the challenges of cooperation in situations where individual incentives may lead to suboptimal collective outcomes.
It has applications in various fields to analyze situations involving cooperation and competition.

This game can be imported, as follows:

.. code-block:: python

    import mlpro.rl.pool.envs.bglp

**Player, Coalition, and Competition**

In the context of the Prisoner's Dilemma, there are two players, often referred to as Player 1 and Player 2.
The situation involves two individuals who have been accused of committing a crime together and are now being held in separate cells, unable to communicate with each other.
The choices available to each player are either to "Confess" or "Not Confess".
The outcomes and associated payoffs for each player depend on the combination of choices made by both individuals. 

**Payoff Matrix**

+------------------------------------+-------------------------------------------------------+-------------------------------------------------------+
|                                    |                Player 2: Confess                      |                Player 2: Not Confess                  |
+====================================+=======================================================+=======================================================+
|      **Player 1: Confess**         |            (5, 5)                                     |                  (8, 1)                               |
+------------------------------------+-------------------------------------------------------+-------------------------------------------------------+
|      **Player 2: Not Confess**     |            (1, 8)                                     |                  (2, 2)                               |
+------------------------------------+-------------------------------------------------------+-------------------------------------------------------+

**Solvers**

+------------------------------------+-------------------------------------------------------+
|           Player                   |                         Solvers                       |
+====================================+=======================================================+
| 1                                  | Random Solver                                         |
+------------------------------------+-------------------------------------------------------+
| 2                                  | Random Solver                                         |
+------------------------------------+-------------------------------------------------------+

**Cross References**

    + :ref:`API Reference <target_api_gt_pool_2pprisoners>`
    + :ref:`Howto GT-Native-001: 2P Prisoners Dilemma <Howto GTN 001>`

**Citation**

If you apply this game in your research or work, do not forget to :ref:`cite us <target_publications>`.




