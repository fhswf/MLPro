.. _Howto RL HT 002:
Howto RL-HT-002: Hyperparameter Tuning using Optuna
===================================================

.. automodule:: mlpro.rl.examples.howto_rl_ht_002_optuna



**Prerequisites**

Please install the following packages to run this examples properly:
    - `Optuna <https://pypi.org/project/optuna/>`_



**Executable code**

.. literalinclude:: ../../../../../../../../src/mlpro/rl/examples/howto_rl_ht_002_optuna.py
	:language: python


**Results**

.. code-block:: bashh

    2023-02-12  16:50:55.790961  I  Wrapper "Optuna": Instantiated
    2023-02-12  16:50:56.033963  I  Wrapper "Optuna": Wrapped package optuna installed in version 3.1.0
    2023-02-12  16:50:56.033963  I  Wrapper "Optuna": Optuna configuration is successful
    2023-02-12  16:50:56.033963  I  Training "RL": Instantiated
    2023-02-12  16:50:56.033963  I  Training "RL": Training started (with hyperparameter tuning)
    2023-02-12  16:50:56.035961  I  Environment "BGLP": Instantiated
    2023-02-12  16:50:56.035961  I  Environment "BGLP": Reset
    2023-02-12  16:50:56.037961  I  Policy "MyPolicy 41746e2c-045e-485e-9dbf-7cbfc1acaddb": Instantiated
    2023-02-12  16:50:56.037961  I  Policy "MyPolicy 41746e2c-045e-485e-9dbf-7cbfc1acaddb": Adaptivity switched on
    2023-02-12  16:50:56.037961  I  Agent "BELT_CONVEYOR_A": Instantiated
    2023-02-12  16:50:56.037961  I  Agent "BELT_CONVEYOR_A": Adaptivity switched on
    2023-02-12  16:50:56.037961  I  Policy "MyPolicy 41746e2c-045e-485e-9dbf-7cbfc1acaddb": Adaptivity switched on
    2023-02-12  16:50:56.037961  I  Agent "BELT_CONVEYOR_A": Adaptivity switched on
    2023-02-12  16:50:56.037961  I  Policy "MyPolicy 41746e2c-045e-485e-9dbf-7cbfc1acaddb": Adaptivity switched on
    2023-02-12  16:50:56.038960  I  Policy "MyPolicy 4d6eff7c-d873-4806-ad4d-b02cc70bb689": Instantiated
    2023-02-12  16:50:56.038960  I  Policy "MyPolicy 4d6eff7c-d873-4806-ad4d-b02cc70bb689": Adaptivity switched on
    2023-02-12  16:50:56.038960  I  Agent "VACUUM_PUMP_B": Instantiated
    2023-02-12  16:50:56.039961  I  Agent "VACUUM_PUMP_B": Adaptivity switched on
    2023-02-12  16:50:56.039961  I  Policy "MyPolicy 4d6eff7c-d873-4806-ad4d-b02cc70bb689": Adaptivity switched on
    2023-02-12  16:50:56.039961  I  Agent "VACUUM_PUMP_B": Adaptivity switched on
    2023-02-12  16:50:56.039961  I  Policy "MyPolicy 4d6eff7c-d873-4806-ad4d-b02cc70bb689": Adaptivity switched on
    2023-02-12  16:50:56.039961  I  Policy "MyPolicy 47697bb1-df82-412e-b11d-da9be4d71fbb": Instantiated
    2023-02-12  16:50:56.039961  I  Policy "MyPolicy 47697bb1-df82-412e-b11d-da9be4d71fbb": Adaptivity switched on
    2023-02-12  16:50:56.040959  I  Agent "VIBRATORY_CONVEYOR_B": Instantiated
    2023-02-12  16:50:56.040959  I  Agent "VIBRATORY_CONVEYOR_B": Adaptivity switched on
    2023-02-12  16:50:56.040959  I  Policy "MyPolicy 47697bb1-df82-412e-b11d-da9be4d71fbb": Adaptivity switched on
    2023-02-12  16:50:56.040959  I  Agent "VIBRATORY_CONVEYOR_B": Adaptivity switched on
    2023-02-12  16:50:56.040959  I  Policy "MyPolicy 47697bb1-df82-412e-b11d-da9be4d71fbb": Adaptivity switched on
    2023-02-12  16:50:56.040959  I  Policy "MyPolicy 23203733-d4bc-44c4-a154-cad82824f622": Instantiated
    2023-02-12  16:50:56.040959  I  Policy "MyPolicy 23203733-d4bc-44c4-a154-cad82824f622": Adaptivity switched on
    2023-02-12  16:50:56.041961  I  Agent "VACUUM_PUMP_C": Instantiated
    2023-02-12  16:50:56.041961  I  Agent "VACUUM_PUMP_C": Adaptivity switched on
    2023-02-12  16:50:56.041961  I  Policy "MyPolicy 23203733-d4bc-44c4-a154-cad82824f622": Adaptivity switched on
    2023-02-12  16:50:56.041961  I  Agent "VACUUM_PUMP_C": Adaptivity switched on
    2023-02-12  16:50:56.041961  I  Policy "MyPolicy 23203733-d4bc-44c4-a154-cad82824f622": Adaptivity switched on
    2023-02-12  16:50:56.041961  I  Policy "MyPolicy b821f2cc-2add-42e3-9326-7a37bef1ba3d": Instantiated
    2023-02-12  16:50:56.041961  I  Policy "MyPolicy b821f2cc-2add-42e3-9326-7a37bef1ba3d": Adaptivity switched on
    2023-02-12  16:50:56.042961  I  Agent "ROTARY_FEEDER_C": Instantiated
    2023-02-12  16:50:56.042961  I  Agent "ROTARY_FEEDER_C": Adaptivity switched on
    2023-02-12  16:50:56.042961  I  Policy "MyPolicy b821f2cc-2add-42e3-9326-7a37bef1ba3d": Adaptivity switched on
    2023-02-12  16:50:56.042961  I  Agent "ROTARY_FEEDER_C": Adaptivity switched on
    2023-02-12  16:50:56.042961  I  Policy "MyPolicy b821f2cc-2add-42e3-9326-7a37bef1ba3d": Adaptivity switched on
    [I 2023-02-12 16:50:56,044] A new study created in memory with name: no-name-92b887d1-0699-42f7-8d74-ddc065e13b15
    C:\MLPro\MLPro\src\mlpro\wrappers\optuna.py:245: FutureWarning: suggest_uniform has been deprecated in v3.0.0. This feature will be removed in v6.0.0. See https://github.com/optuna/optuna/releases/tag/v3.0.0. Use :func:`~optuna.trial.Trial.suggest_float` instead.
      parameters.append(trial.suggest_uniform(hp_object.get_name_short()+'_'+str(x),hp_low,hp_high))
    2023-02-12  16:50:56.046964  I  Wrapper "Optuna": Trial number 0 has started
    2023-02-12  16:50:56.046964  I  Wrapper "Optuna": ------------------------------------------------------------------------------

    2023-02-12  16:50:56.047961  I  Environment "BGLP": Instantiated
    2023-02-12  16:50:56.047961  I  Environment "BGLP": Reset
    2023-02-12  16:50:56.048960  I  Policy "MyPolicy 2343b475-8b8a-4fa9-9551-3f91751bf067": Instantiated
    2023-02-12  16:50:56.048960  I  Policy "MyPolicy 2343b475-8b8a-4fa9-9551-3f91751bf067": Adaptivity switched on
    2023-02-12  16:50:56.049965  I  Agent "BELT_CONVEYOR_A": Instantiated
    2023-02-12  16:50:56.049965  I  Agent "BELT_CONVEYOR_A": Adaptivity switched on
    2023-02-12  16:50:56.049965  I  Policy "MyPolicy 2343b475-8b8a-4fa9-9551-3f91751bf067": Adaptivity switched on
    2023-02-12  16:50:56.049965  I  Agent "BELT_CONVEYOR_A": Adaptivity switched on
    2023-02-12  16:50:56.049965  I  Policy "MyPolicy 2343b475-8b8a-4fa9-9551-3f91751bf067": Adaptivity switched on
    2023-02-12  16:50:56.049965  I  Policy "MyPolicy 00d3b281-3e2f-459a-8b96-f5d45b0ef172": Instantiated
    2023-02-12  16:50:56.049965  I  Policy "MyPolicy 00d3b281-3e2f-459a-8b96-f5d45b0ef172": Adaptivity switched on
    2023-02-12  16:50:56.050961  I  Agent "VACUUM_PUMP_B": Instantiated
    2023-02-12  16:50:56.050961  I  Agent "VACUUM_PUMP_B": Adaptivity switched on
    2023-02-12  16:50:56.050961  I  Policy "MyPolicy 00d3b281-3e2f-459a-8b96-f5d45b0ef172": Adaptivity switched on
    2023-02-12  16:50:56.050961  I  Agent "VACUUM_PUMP_B": Adaptivity switched on
    2023-02-12  16:50:56.050961  I  Policy "MyPolicy 00d3b281-3e2f-459a-8b96-f5d45b0ef172": Adaptivity switched on
    2023-02-12  16:50:56.050961  I  Policy "MyPolicy 50ca0db6-d3f7-435b-82db-5273a2df4798": Instantiated
    2023-02-12  16:50:56.050961  I  Policy "MyPolicy 50ca0db6-d3f7-435b-82db-5273a2df4798": Adaptivity switched on
    2023-02-12  16:50:56.051965  I  Agent "VIBRATORY_CONVEYOR_B": Instantiated
    2023-02-12  16:50:56.051965  I  Agent "VIBRATORY_CONVEYOR_B": Adaptivity switched on
    2023-02-12  16:50:56.051965  I  Policy "MyPolicy 50ca0db6-d3f7-435b-82db-5273a2df4798": Adaptivity switched on
    2023-02-12  16:50:56.051965  I  Agent "VIBRATORY_CONVEYOR_B": Adaptivity switched on
    2023-02-12  16:50:56.051965  I  Policy "MyPolicy 50ca0db6-d3f7-435b-82db-5273a2df4798": Adaptivity switched on
    2023-02-12  16:50:56.051965  I  Policy "MyPolicy 2ce4e729-cacd-47e9-9e55-4137858c7467": Instantiated
    2023-02-12  16:50:56.051965  I  Policy "MyPolicy 2ce4e729-cacd-47e9-9e55-4137858c7467": Adaptivity switched on
    2023-02-12  16:50:56.052964  I  Agent "VACUUM_PUMP_C": Instantiated
    2023-02-12  16:50:56.052964  I  Agent "VACUUM_PUMP_C": Adaptivity switched on
    2023-02-12  16:50:56.052964  I  Policy "MyPolicy 2ce4e729-cacd-47e9-9e55-4137858c7467": Adaptivity switched on
    2023-02-12  16:50:56.052964  I  Agent "VACUUM_PUMP_C": Adaptivity switched on
    2023-02-12  16:50:56.052964  I  Policy "MyPolicy 2ce4e729-cacd-47e9-9e55-4137858c7467": Adaptivity switched on
    2023-02-12  16:50:56.052964  I  Policy "MyPolicy 41c9f1a1-4e3f-4819-a768-fd714135f20c": Instantiated
    2023-02-12  16:50:56.052964  I  Policy "MyPolicy 41c9f1a1-4e3f-4819-a768-fd714135f20c": Adaptivity switched on
    2023-02-12  16:50:56.053962  I  Agent "ROTARY_FEEDER_C": Instantiated
    2023-02-12  16:50:56.053962  I  Agent "ROTARY_FEEDER_C": Adaptivity switched on
    2023-02-12  16:50:56.053962  I  Policy "MyPolicy 41c9f1a1-4e3f-4819-a768-fd714135f20c": Adaptivity switched on
    2023-02-12  16:50:56.053962  I  Agent "ROTARY_FEEDER_C": Adaptivity switched on
    2023-02-12  16:50:56.053962  I  Policy "MyPolicy 41c9f1a1-4e3f-4819-a768-fd714135f20c": Adaptivity switched on
    2023-02-12  16:50:56.065960  I  Wrapper "Optuna": New parameters for optuna tuner is ready
    2023-02-12  16:50:56.065960  I  Wrapper "Optuna": ------------------------------------------------------------------------------



**Cross Reference**

    + :ref:`API Reference - RL Agent <target_api_rl_agents>`
    + :ref:`API Reference - RL Environments <target_api_rl_env>`
    + :ref:`API Reference - RL Scenario and Training <target_api_rl_run_train>`
    + :ref:`API Reference - Machine Learning <target_api_bf_ml>`
