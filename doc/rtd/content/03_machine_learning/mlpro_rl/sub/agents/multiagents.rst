Multi-Agents
--------------

MLPro-RL is not only compatible with single-agent RL but also multi-agent RL, where the extent of the agent landscape is completed by the multi-agent model.
It is compatible with single-agent but does not have its own policy.
Instead, it is utilized to combine and control any quantity of single agents that together control the action calculation.
Every single agent in this situation interacts with a separate portion of the surrounding multi-observation agents and action space.
Multi-agent interactions take place in appropriate contexts that support the scalar reward per agent reward type. 
These are native applications that incorporate the MLPro environment template or PettingZoo environments that may be incorporated using the corresponding :ref:`wrapper class<target-package-third>` offered by MLPro.

For setting-up multi-agents in MLPro-RL, you can refer to :ref:`this page<target_training_RL>` or :ref:`this howto file <Howto Agent RL 004>`.