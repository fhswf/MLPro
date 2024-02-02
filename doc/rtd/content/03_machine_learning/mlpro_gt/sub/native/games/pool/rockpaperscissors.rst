Rock, Paper, Scissors
^^^^^^^^^^^^^^^^^^^^^^^^^

**General Information**

Rock, Paper, Scissors is a simple hand game often used as a playful decision-making tool.
In game theory, it serves as a basic example of a non-cooperative, simultaneous-move, zero-sum game.
The game is typically played between two people who simultaneously form one of three shapes with their hands: a rock, paper, or scissors.

The basic rules are as follows:

    + Rock crushes Scissors,
    + Scissors cuts Paper, and
    + Paper covers Rock.

The game has a cyclic nature, with no single option dominating the others.
Each choice has a clear advantage over one option and a disadvantage against another.

The game is designed such that the payoffs balance out to zero in each possible outcome, making it a zero-sum game.
In a theoretical sense, neither player has a dominant strategy, and the optimal strategy involves randomizing between Rock, Paper, and Scissors to prevent the opponent from predicting their moves.

Rock, Paper, Scissors is often used as a teaching tool in introductory game theory to illustrate concepts such as mixed strategies, Nash equilibrium, and the nature of zero-sum games.
Despite its simplicity, it provides insights into strategic thinking and decision-making in competitive situations.

This game can be imported, as follows:

.. code-block:: python

    from mlpro.gt.pool.native.games.rockpaperscissors import RockPaperScissors

**Player, Coalition, and Competition**

In the context of Rock, Paper, Scissors, there are usually two players, often referred to as Player 1 and Player 2.
However, in this game, we setup a coalition against a coaltion Rock, Paper, Scissors games.
The game consists of two coaltions, where each coalition makes a decision based on the colllaborative approach between the coalitions.
Each coalition consists of 5 members, the most voted decision of the 5 members represents the final decision of the coalition. 

**Payoff Matrix**

The outcomes are often represented in a payoff matrix, where each coaltion's payoff depends on the combination of choices made by both coalitions:

+------------------------------+-----------------------------------------+----------------------------------------+----------------------------------------+
|                              |         Rock                            |         Paper                          |         Scissors                       |
+==============================+=========================================+========================================+========================================+
|      **Rock**                |            (0, 0)                       |            (0, 1)                      |            (1, 0)                      |
+------------------------------+-----------------------------------------+----------------------------------------+----------------------------------------+
|      **Paper**               |            (1, 0)                       |            (0, 0)                      |            (0, 1)                      |
+------------------------------+-----------------------------------------+----------------------------------------+----------------------------------------+
|      **Scissors**            |            (0, 1)                       |            (1, 0)                      |            (0, 0)                      |
+------------------------------+-----------------------------------------+----------------------------------------+----------------------------------------+

Here, the first value in each pair represents the payoff to the row coalition, and the second value represents the payoff to the column coalition.

**Solvers**

+------------------------------------+-------------------------------------------------------+
|           Coalition                |                         Solvers                       |
+====================================+=======================================================+
| 1                                  | Random solvers for all players                        |
+------------------------------------+-------------------------------------------------------+
| 2                                  | Random solvers for all players                        |
+------------------------------------+-------------------------------------------------------+

**Cross References**

    + :ref:`Howto GT-Native-003: Rock, Paper, Scissors <Howto GTN 003>`
    + :ref:`API Reference <target_api_gt_pool_rps>`

**Citation**

If you apply this game in your research or work, do not forget to :ref:`cite us <target_publications>`.
