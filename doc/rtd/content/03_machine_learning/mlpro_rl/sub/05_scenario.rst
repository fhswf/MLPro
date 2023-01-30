.. _target_scenario_RL:
Scenarios
------------

A scenario in reinforcement learning (RL) refers to a specific problem or task that the agent is trying to learn how to solve.
A scenario defines the environment in which the agent operates, including the state space, the action space, the reward function, and the transition dynamics.

In RL, the agent interacts with the environment over a sequence of time steps.
At each time step, the agent receives an observation of the current state of the environment and selects an action.
The environment then transitions to a new state and returns a reward signal to the agent.

The scenario provides the agent with a set of goals to be achieved, and the reward function quantifies how well the agent is doing in terms of achieving these goals.
The reward function can be designed to encourage certain behaviors, such as reaching a specific target state, or penalize certain behaviors, such as taking actions that lead to a state of low reward.

The scenario also defines the state and action spaces, which are the sets of all possible states and actions that the agent can experience and take, respectively.
The transition dynamics describe how the environment changes in response to the agent's actions.

Overall, a scenario in RL defines the problem that the agent is trying to solve, and provides the necessary information for the agent to learn a policy that maps states to actions and maximizes the cumulative reward signal.

In MLPro-RL, a class **RLScenario** inherits the functionality from class **Scenario** in the basic function level, where the **RLScenario** class combines RL agents and an environment into an executable unit.

One of the MLPro's features is enabling the user to apply a template class for an RL scenario consisting of an environment and agents.
Moreover, you can create either a single-agent scenario or a multi-agent scenario in a simple manner.
Here is the commented class diagram for creating a custom scenario in MLPro-RL:

.. image:: images/MLPro-RL-Scenario_Class_Commented.png

For setting-up a scenarion for single-agent and multi-agent in MLPro-RL, you can refer to :ref:`this page<target_training_RL>`.