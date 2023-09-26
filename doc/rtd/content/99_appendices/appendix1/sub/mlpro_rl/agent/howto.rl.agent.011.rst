.. _Howto Agent RL 011:
Howto RL-AGENT-011: Train and Reload Single Agent (Gym)
=======================================================

**Prerequisites**

Please install the following packages to run this examples properly:

    - `OpenAI Gym <https://pypi.org/project/gym/>`_
    - `Stable-Baselines3 <https://pypi.org/project/stable-baselines3/>`_



**Executable code**

.. literalinclude:: ../../../../../../../../test/howtos/rl/howto_rl_agent_011_train_and_reload_single_agent_gym.py
	:language: python



**Results**

The Gym Cartpole environment window appears. Afterwards, the training runs 
for a few episodes before terminating and printing the result. 

After termination the local result folders contain the training result files:
    - agent_actions.csv
    - env_rewards.csv
    - env_states.csv
    - evaluation.csv
    - summary.csv
    - trained model.pkl

Both training results are from the same agent.



**Cross Reference**

    - :ref:`MLPro-RL: Training <target_training_RL>`
    - :ref:`API Reference <target_api_rl_run_train>`
