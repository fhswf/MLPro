.. _target_player_sbpg:
State-Based Potential Games (SbPG)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: mlpro.gt.pool.policies.sbpg


**Prerequisites**

    - `PyTorch <https://pypi.org/project/torch/>`_



**General information**

State-Based Potential Games (SbPG) are a class of multi-agent learning frameworks that extend the concept of potential games to environments with explicitly modeled states.
In traditional potential games, each agent’s incentive aligns with a global potential function, meaning any improvement in an individual agent’s utility corresponds to an improvement in the shared objective.
SbPG adapts this principle by introducing a state space that discretizes the environment, allowing agents to adapt their behavior not just based on actions, but also on spatial or situational context.
This is particularly useful in complex dynamic environments like manufacturing systems, smart grids, or logistics networks, where the state of the system plays a crucial role in decision-making.

In an SbPG setup, the environment is divided into a dsicrete grid of states, and agents maintain a performance map that records the best-known action and corresponding utility for each state.
Over time, agents use reinforcement learning techniques to update this map, either by using Best Response learning or by estimating the gradient of the utility landscape (Gradient-Based learning).
This process allows agents to refine their strategies iteratively, seeking actions that maximize their individual payoff while collectively steering the system toward more optimal global behavior.

The SbPG framework supports several learning algorithms to update policies, notably Best Response (BR), Gradient-Based (GB), and Gradient-Based with Momentum (GB_MOM).
BR is a simpler approach where agents always sample random actions during exploring.
GB uses the utility gradient to suggest better actions over time, while GB_MOM adds a momentum term, smoothing learning and improving convergence stability.

The SbPG can be imported via:

.. code-block:: python

    from mlpro.gt.pool.policies.sbpg import SbPG



**Cross reference**

    + :ref:`Howto GT-DG-003: Train Multi-Player with SbPG on the BGLP <Howto GT 003>`
    + :ref:`API reference <target_pool_dg_sbpg>`

**Citation**

If you apply this policy in your research or work, please :ref:`cite <target_publications>` us and
the original papers: `for gradient-based learning <https://ieeexplore.ieee.org/document/10905619>`_ and `for best response learning <https://ieeexplore.ieee.org/document/9152119>`_.

