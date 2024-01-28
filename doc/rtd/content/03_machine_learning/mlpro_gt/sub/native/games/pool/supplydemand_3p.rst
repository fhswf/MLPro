3P Supply and Demand
^^^^^^^^^^^^^^^^^^^^^^^^^

**General Information**

In a 3-seller supply and demand game within the framework of game theory, three sellers, denoted as Seller 1, Seller 2, and Seller 3, engage in strategic decision-making alongside a group of buyers in a market where goods or services are exchanged.
Each seller must decide on the quantity of goods they wish to supply/produce to the market, while buyers independently determine the quantity they want to demand.
The payoff for sellers is typically associated with their revenue, contingent on the quantity they sell and the price they set, while buyers' payoffs may relate to the utility they derive from purchased goods and the corresponding prices.

The interaction between the supply decisions of sellers and the demand decisions of buyers is crucial in determining the market dynamics.
Sellers strive to maximize their revenue, considering factors like quantity and pricing, while buyers aim to maximize their utility by deciding on the quantity they wish to purchase and the price they are willing to pay.
The equilibrium, a fundamental concept in competitive markets, occurs when the quantity supplied equals the quantity demanded, establishing a market-clearing price.

Strategic considerations play a pivotal role as sellers may take into account the decisions of other sellers when determining their own supply/production quantities and pricing strategies.
Similarly, buyers may consider the prices set by sellers and the available quantities when deciding on their demand quantities.
The Nash equilibrium, a key concept in game theory, is reached when each participant's strategy is optimal given the strategies chosen by others.

In this market setting, competition arises as sellers may compete against each other to attract buyers, adjusting prices or quantities to gain a competitive edge.
Alternatively, sellers may engage in strategic cooperation, such as forming a cartel, to collectively influence prices and quantities in the market.
The game dynamics involve repeated interactions between sellers and buyers, leading to adjustments in strategies based on past outcomes and market conditions.

Analyzing this type of game requires an examination of the strategies and payoffs for both sellers and buyers, identification of potential equilibria, and an understanding of how the strategic decisions of each participant influence the overall market dynamics.
Game theory provides a valuable framework for studying the strategic interactions and decision-making processes in complex market scenarios involving multiple sellers and buyers.

This game can be imported, as follows:

.. code-block:: python

    from mlpro.gt.pool.native.games.supplydemand_3p import SupplyDemand_3P

**Player, Coalition, and Competition**

In the 3P Supply and Demand games in MLPro-GT, we set up three players with a constant market demand of the products is 10 products/day, where the buyer will always firstly buy the products with less prices.
Each player in the game represent each seller. They need to compete or cooperate with each other to maximize their indvidual profit.

**Payoff Function**

The tables below show the sales price per product based on the quantity produced by each seller:

+------------------------------+-----------------------------------------+----------------------------------------+----------------------------------------+----------------------------------------+----------------------------------------+
|                                       Seller 1                         |                             Seller 2                                            |                             Seller 3                                            |
+==============================+=========================================+========================================+========================================+========================================+========================================+
|      **Price (€)**           |            **Quantity**                 |            **Price (€)**               |            **Quantity**                |            **Price (€)**               |            **Quantity**                |
+------------------------------+-----------------------------------------+----------------------------------------+----------------------------------------+----------------------------------------+----------------------------------------+
|      15                      |            1                            |            10                          |            1                           |            8                           |            1                           |
+------------------------------+-----------------------------------------+----------------------------------------+----------------------------------------+----------------------------------------+----------------------------------------+
|      12                      |            2                            |            8                           |            2                           |            7                           |            2                           |
+------------------------------+-----------------------------------------+----------------------------------------+----------------------------------------+----------------------------------------+----------------------------------------+
|      9                       |            3                            |            6                           |            3                           |            6                           |            3                           |
+------------------------------+-----------------------------------------+----------------------------------------+----------------------------------------+----------------------------------------+----------------------------------------+
|      6                       |            4                            |            4                           |            4                           |            5                           |            4                           |
+------------------------------+-----------------------------------------+----------------------------------------+----------------------------------------+----------------------------------------+----------------------------------------+
|     3                        |            5                            |            2                           |            5                           |            4                           |            5                           |
+------------------------------+-----------------------------------------+----------------------------------------+----------------------------------------+----------------------------------------+----------------------------------------+

**Solvers**

+------------------------------------+-------------------------------------------------------+
|           Seller                   |                         Solvers                       |
+====================================+=======================================================+
| 1                                  | Max Greedy Solver                                     |
+------------------------------------+-------------------------------------------------------+
| 2                                  | Max Greedy Solver                                     |
+------------------------------------+-------------------------------------------------------+
| 3                                  | Random Solver                                         |
+------------------------------------+-------------------------------------------------------+

**Cross References**

    + :ref:`API Reference <target_api_gt_pool_3psd>`
    + :ref:`Howto GT-Native-004: 3P Supply and Demand <Howto GTN 004>`

**Citation**

If you apply this game in your research or work, do not forget to :ref:`cite us <target_publications>`.
