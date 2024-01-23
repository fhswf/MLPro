.. _target_players_GT:
Players
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the game theoretical approach, the player takes decisions based on the actual states in a game. 
A game contains two or more players that can be working cooperatively or compete with each other.
Each player is supplied with a policy, where the policy can be used for the decision-making process and also optimized.
The decision-making of the player is in the form of the next action.

The three main tasks of the player are as follows,

   (1) Compute new action based on the current state.

   (2) Calculate local utility value based on the previous state and next state.

   (3) Optimize their policy based on the state and the selected action.

In MLPro, you can customize your GT-based policy or import the provided policies in the pool of objects (unavailable at the moment).

.. toctree::
   :maxdepth: 1
   
   players/custompolicies