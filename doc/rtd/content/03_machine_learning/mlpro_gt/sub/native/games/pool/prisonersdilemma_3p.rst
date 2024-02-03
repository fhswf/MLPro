3P Prisoners’ Dilemma
^^^^^^^^^^^^^^^^^^^^^^^^^

**General Information**

The Prisoner's Dilemma can be extended to involve three players, creating what is known as a 3-player Prisoner's Dilemma game.
In this scenario, three individuals face a situation where they must decide whether to cooperate or betray each other, with the outcomes determined by the combined choices made by all three players.
The setup involves a payoff matrix that outlines the possible outcomes and associated payoffs for each player based on their choices.

Each player has two possible choices: "Confess" or "Not Confess".
The outcomes are determined based on the combination of choices made by all three players.
The challenge is that, similarly to the 2-player version, there is a conflict between individual rationality and collective optimality.

The players individually have an incentive to not confess, as not confessing typically yields a higher payoff for the individual, regardless of the choices made by others.
However, if all players not confessing, the collective outcome may be suboptimal compared to a scenario where all players cooperate.

Analyzing and solving a 3-player Prisoner's Dilemma involves considering the strategic interactions among all three players, understanding the incentives for cooperation and betrayal, and exploring whether there are any stable strategies or equilibria in the game.
The extension to more players adds complexity to the decision-making dynamics and strategic considerations.

This game can be imported, as follows:

.. code-block:: python

    from mlpro.gt.pool.native.games.prisonersdilemma_3p import PrisonersDilemma3PGame

**Player, Coalition, and Competition**

In the context of the Prisoner's Dilemma, there are threee players, often referred to as Player 1, Player 2, and Player 3.
The situation involves three individuals who have been accused of committing a crime together and are now being held in separate cells, unable to communicate with each other.
The choices available to each player are either to "Confess" or "Not Confess".
The outcomes and associated payoffs for each player depend on the combination of choices made by all individuals. 

**Payoff Matrix**

+------------------------------+-----------------------------------------+----------------------------------------+----------------------------------------+----------------------------------------+
|           (P1, P2, P3)       |         P2: Confess, P3: Confess        |         P2: Confess, P3: Not Confess   |         P2: Not Confess, P3: Confess   |   P2: Not Confess, P3: Not Confess     |
+==============================+=========================================+========================================+========================================+========================================+
|      **P1: Confess**         |            (2, 2, 2)                    |            (5, 5, 1)                   |            (5, 1, 5)                   |            (10, 1, 1)                  |
+------------------------------+-----------------------------------------+----------------------------------------+----------------------------------------+----------------------------------------+
|      **P2: Not Confess**     |            (1, 5, 5)                    |            (1, 10, 1)                  |            (1, 1, 10)                  |            (15, 15, 15)                |
+------------------------------+-----------------------------------------+----------------------------------------+----------------------------------------+----------------------------------------+


**Solvers**

+------------------------------------+-------------------------------------------------------+
|           Player                   |                         Solvers                       |
+====================================+=======================================================+
| 1                                  | Random Solver, Min Greedy Solver                      |
+------------------------------------+-------------------------------------------------------+
| 2                                  | Min Greedy Solver                                     |
+------------------------------------+-------------------------------------------------------+
| 3                                  | Random Solver, Min Greedy Solver                      |
+------------------------------------+-------------------------------------------------------+

**Cross References**

    + :ref:`Howto GT-Native-002: 3P Prisoners Dilemma <Howto GTN 002>`
    + :ref:`API Reference <target_api_gt_pool_3pprisoners>`

**Citation**

If you apply this game in your research or work, do not forget to :ref:`cite us <target_publications>`.
