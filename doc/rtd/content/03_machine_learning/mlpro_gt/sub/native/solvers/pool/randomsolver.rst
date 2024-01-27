Random Solvers
^^^^^^^^^^^^^^^^^^^^^^^^^
  
**General Information**

A random solver for a specific strategy space with defined boundaries is an algorithm that generates random values within a specific range or set of constraints.
This type of random solver is often used for testing environments, generate random behaviour of competitiors or alliances, and many more.
The random generator can be attach to an GT player/coalition and will detect the strategy space of the player/coalition as well as the boundaries.
The strategy space refers to the set of possible strategies of the agent, and the boundaries define the range of values within which the strategy must lie.
Then, a strategy is randomly computed by the generator using uniform distribution.  

The random solvers can be imported via:

.. code-block:: python

    from mlpro.gt.pool.native.solvers.randomsolver import RandomSolver
    

**Cross Reference**
    + :ref:`API Reference <target_native_pool_gt_solvers_random>`
    + :ref:`Howto GT-Native-001: 2P Prisoners Dilemma <Howto GTN 001>`


**Citation**

If you apply this solver in your research or work, please :ref:`cite <target_publications>` us.

