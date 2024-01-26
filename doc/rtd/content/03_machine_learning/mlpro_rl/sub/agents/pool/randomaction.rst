Random Action Generator
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: mlpro.rl.pool.policies.randomgenerator
    
    
**General Information**

A random generator for a specific action space with defined boundaries is an algorithm that generates random values within a specific range or set of constraints.
This type of random generator is often used for testing environments, generate sample data for model-based learning, and many more.
The random generator can be attach to an RL agent and will detect the action space of the agent as well as the boundaries.
The action space refers to the set of possible actions of the agent, and the boundaries define the range of values within which the actions must lie.
Then, an action is randomly computed by the generator using uniform distribution.  

For example, if the action space of an agent is defined as a joint velocity of a robot, the boundaries may be defined as the minimum and maximum velocity of the joint.
The random generator would then generate random values within these boundaries that represent the possible velocity of the robot.
The specific method used to generate random values within the action space boundaries depends on the type of data being generated and the desired properties of the generated data.
    
This Random Action Generator policy can be imported via:

.. code-block:: python

    from mlpro.rl.pool.policies.randomgenerator import RandomGenerator
    

**Cross Reference**
    + :ref:`API Reference <target_pol_randoms>`


**Citation**

If you apply this policy in your research or work, please :ref:`cite <target_publications>` us.

