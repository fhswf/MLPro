.. _Howto BF SYSTEMS 001:
Howto BF-SYSTEMS-001: Demonstrating Native Systems
====================================================================

.. automodule:: mlpro.bf.examples.howto_bf_systems_001_demonstrating_native_systems

**Prerequisites**

**Executable Code**

.. literalinclude:: ../../../../../../../../../src/mlpro/bf/examples/howto_bf_systems_001_demonstrating_native_systems.py
	:language: python

**Results**

.. code-block:: bashh
    
    2023-05-03  10:50:48.419856  I  System "DoublePendulumSystemS4": Instantiated 
    2023-05-03  10:50:48.419856  I  System "DoublePendulumSystemS4": Reset 
    2023-05-03  10:50:48.420853  I  System "DoublePendulumSystemS7": Instantiated 
    2023-05-03  10:50:48.420853  I  System "DoublePendulumSystemS7": Reset 
    2023-05-03  10:50:48.420853  I  Scenario Base "????": Instantiated 
    2023-05-03  10:50:48.420853  I  Scenario Base "????": Process time 0:00:00 : Scenario reset with seed 1 
    2023-05-03  10:50:48.421853  I  System "DoublePendulumSystemS4": Reset 
    2023-05-03  10:50:48.421853  S  Scenario Base "????": Process time 0:00:00 : Start of processing 
    2023-05-03  10:50:48.421853  S  Scenario Base "????": Process time 0:00:00 : Start of cycle 0 
    2023-05-03  10:50:48.421853  I  Scenario Base "????": Generating new action 
    2023-05-03  10:50:48.421853  I  System "DoublePendulumSystemS4": Start processing action 
    2023-05-03  10:50:48.421853  I  System "DoublePendulumSystemS4": Actions of agent 0 = [-0.18259651632236285] 
    2023-05-03  10:50:48.422850  I  System "DoublePendulumSystemS4": Assessment for success... 
    2023-05-03  10:50:48.423849  I  System "DoublePendulumSystemS4": Assessment for breakdown...
    2023-05-03  10:50:48.423849  I  System "DoublePendulumSystemS4": Action processing finished successfully
    2023-05-03  10:50:48.423849  I  System "DoublePendulumSystemS4": Assessment for breakdown...
    2023-05-03  10:50:48.423849  S  Scenario Base "????": Process time 0:00:00 : End of cycle 0
    2023-05-03  10:50:48.424853  S  Scenario Base "????": Process time 0:00:00.040000 : Start of cycle 1
    2023-05-03  10:50:48.424853  I  Scenario Base "????": Generating new action
    2023-05-03  10:50:48.424853  I  System "DoublePendulumSystemS4": Start processing action
    2023-05-03  10:50:48.424853  I  System "DoublePendulumSystemS4": Actions of agent 0 = [-2.020357408450476]
    2023-05-03  10:50:48.426856  I  System "DoublePendulumSystemS4": Assessment for success...
    2023-05-03  10:50:48.427853  I  System "DoublePendulumSystemS4": Assessment for breakdown...
    2023-05-03  10:50:48.427853  I  System "DoublePendulumSystemS4": Action processing finished successfully
    2023-05-03  10:50:48.427853  I  System "DoublePendulumSystemS4": Assessment for breakdown...
    2023-05-03  10:50:48.428854  S  Scenario Base "????": Process time 0:00:00.040000 : End of cycle 1
    2023-05-03  10:50:48.428854  S  Scenario Base "????": Process time 0:00:00.080000 : Start of cycle 2
    2023-05-03  10:50:48.428854  I  Scenario Base "????": Generating new action
    2023-05-03  10:50:48.428854  I  System "DoublePendulumSystemS4": Start processing action
    2023-05-03  10:50:48.429852  I  System "DoublePendulumSystemS4": Actions of agent 0 = [6.063718908910516]
    2023-05-03  10:50:48.431851  I  System "DoublePendulumSystemS4": Assessment for success...
    2023-05-03  10:50:48.431851  I  System "DoublePendulumSystemS4": Assessment for breakdown...
    2023-05-03  10:50:48.431851  I  System "DoublePendulumSystemS4": Action processing finished successfully
    2023-05-03  10:50:48.432867  I  System "DoublePendulumSystemS4": Assessment for breakdown...
    2023-05-03  10:50:48.432867  S  Scenario Base "????": Process time 0:00:00.080000 : End of cycle 2
    2023-05-03  10:50:48.432867  S  Scenario Base "????": Process time 0:00:00.120000 : Start of cycle 3
    2023-05-03  10:50:48.433854  I  Scenario Base "????": Generating new action
    2023-05-03  10:50:48.433854  I  System "DoublePendulumSystemS4": Start processing action
    2023-05-03  10:50:48.433854  I  System "DoublePendulumSystemS4": Actions of agent 0 = [11.548934045420527]
    2023-05-03  10:50:48.435851  I  System "DoublePendulumSystemS4": Assessment for success...
    2023-05-03  10:50:48.435851  I  System "DoublePendulumSystemS4": Assessment for breakdown...
    2023-05-03  10:50:48.435851  I  System "DoublePendulumSystemS4": Action processing finished successfully
    2023-05-03  10:50:48.436852  I  System "DoublePendulumSystemS4": Assessment for breakdown...
    2023-05-03  10:50:48.436852  S  Scenario Base "????": Process time 0:00:00.120000 : End of cycle 3
    2023-05-03  10:50:48.438163  S  Scenario Base "????": Process time 0:00:00.160000 : Start of cycle 4 
    2023-05-03  10:50:48.438163  I  Scenario Base "????": Generating new action
    2023-05-03  10:50:48.438862  I  System "DoublePendulumSystemS4": Start processing action
    2023-05-03  10:50:48.438862  I  System "DoublePendulumSystemS4": Actions of agent 0 = [-16.245616529030606]
    2023-05-03  10:50:48.440850  I  System "DoublePendulumSystemS4": Assessment for success...
    2023-05-03  10:50:48.440850  I  System "DoublePendulumSystemS4": Assessment for breakdown...
    2023-05-03  10:50:48.440850  I  System "DoublePendulumSystemS4": Action processing finished successfully
    2023-05-03  10:50:48.441852  I  System "DoublePendulumSystemS4": Assessment for breakdown...
    2023-05-03  10:50:48.441852  S  Scenario Base "????": Process time 0:00:00.160000 : End of cycle 4
    2023-05-03  10:50:48.441852  S  Scenario Base "????": Process time 0:00:00.200000 : Start of cycle 5
    2023-05-03  10:50:48.441852  I  Scenario Base "????": Generating new action
    2023-05-03  10:50:48.442852  I  System "DoublePendulumSystemS4": Start processing action
    2023-05-03  10:50:48.442852  I  System "DoublePendulumSystemS4": Actions of agent 0 = [-18.866100939119747]
    2023-05-03  10:50:48.445850  I  System "DoublePendulumSystemS4": Assessment for success...
    2023-05-03  10:50:48.445850  I  System "DoublePendulumSystemS4": Assessment for breakdown...
    2023-05-03  10:50:48.446849  I  System "DoublePendulumSystemS4": Action processing finished successfully
    2023-05-03  10:50:48.446849  I  System "DoublePendulumSystemS4": Assessment for breakdown...
    2023-05-03  10:50:48.446849  S  Scenario Base "????": Process time 0:00:00.200000 : End of cycle 5
    2023-05-03  10:50:48.446849  S  Scenario Base "????": Process time 0:00:00.240000 : Start of cycle 6
    2023-05-03  10:50:48.446849  I  Scenario Base "????": Generating new action
    2023-05-03  10:50:48.446849  I  System "DoublePendulumSystemS4": Start processing action
    2023-05-03  10:50:48.448365  I  System "DoublePendulumSystemS4": Actions of agent 0 = [13.430604156794786]
    2023-05-03  10:50:48.450378  I  System "DoublePendulumSystemS4": Assessment for success...
    2023-05-03  10:50:48.450378  I  System "DoublePendulumSystemS4": Assessment for breakdown...
    2023-05-03  10:50:48.450378  I  System "DoublePendulumSystemS4": Action processing finished successfully
    2023-05-03  10:50:48.451379  I  System "DoublePendulumSystemS4": Assessment for breakdown...
    2023-05-03  10:50:48.451379  S  Scenario Base "????": Process time 0:00:00.240000 : End of cycle 6
    2023-05-03  10:50:48.452385  S  Scenario Base "????": Process time 0:00:00.280000 : Start of cycle 7
    2023-05-03  10:50:48.454383  I  Scenario Base "????": Generating new action 
    2023-05-03  10:50:48.454383  I  System "DoublePendulumSystemS4": Start processing action
    2023-05-03  10:50:48.454383  I  System "DoublePendulumSystemS4": Actions of agent 0 = [-2.689317283797866]
    2023-05-03  10:50:48.457380  I  System "DoublePendulumSystemS4": Assessment for success...
    2023-05-03  10:50:48.458390  I  System "DoublePendulumSystemS4": Assessment for breakdown...
    2023-05-03  10:50:48.458390  I  System "DoublePendulumSystemS4": Action processing finished successfully
    2023-05-03  10:50:48.459386  I  System "DoublePendulumSystemS4": Assessment for breakdown...
    2023-05-03  10:50:48.459386  S  Scenario Base "????": Process time 0:00:00.280000 : End of cycle 7
    2023-05-03  10:50:48.460387  S  Scenario Base "????": Process time 0:00:00.320000 : Start of cycle 8
    2023-05-03  10:50:48.460387  I  Scenario Base "????": Generating new action
    2023-05-03  10:50:48.461382  I  System "DoublePendulumSystemS4": Start processing action
    2023-05-03  10:50:48.462412  I  System "DoublePendulumSystemS4": Actions of agent 0 = [10.491203298317679]
    2023-05-03  10:50:48.465387  I  System "DoublePendulumSystemS4": Assessment for success...
    2023-05-03  10:50:48.466380  I  System "DoublePendulumSystemS4": Assessment for breakdown...
    2023-05-03  10:50:48.466380  I  System "DoublePendulumSystemS4": Action processing finished successfully
    2023-05-03  10:50:48.466380  I  System "DoublePendulumSystemS4": Assessment for breakdown...
    2023-05-03  10:50:48.467403  S  Scenario Base "????": Process time 0:00:00.320000 : End of cycle 8
    2023-05-03  10:50:48.467403  S  Scenario Base "????": Process time 0:00:00.360000 : Start of cycle 9
    2023-05-03  10:50:48.467403  I  Scenario Base "????": Generating new action
    2023-05-03  10:50:48.467403  I  System "DoublePendulumSystemS4": Start processing action
    2023-05-03  10:50:48.468958  I  System "DoublePendulumSystemS4": Actions of agent 0 = [-19.915757865955573] 
    2023-05-03  10:50:48.471972  I  System "DoublePendulumSystemS4": Assessment for success...
    2023-05-03  10:50:48.471972  I  System "DoublePendulumSystemS4": Assessment for breakdown...
    2023-05-03  10:50:48.471972  I  System "DoublePendulumSystemS4": Action processing finished successfully
    2023-05-03  10:50:48.473972  I  System "DoublePendulumSystemS4": Assessment for breakdown...
    2023-05-03  10:50:48.473972  S  Scenario Base "????": Process time 0:00:00.360000 : End of cycle 9
    2023-05-03  10:50:48.474970  S  Scenario Base "????": Process time 0:00:00.400000 : End of processing
    2023-05-03  10:50:48.474970  I  Scenario Base "????": Instantiated
    2023-05-03  10:50:48.475971  I  Scenario Base "????": Process time 0:00:00 : Scenario reset with seed 1
    2023-05-03  10:50:48.475971  I  System "DoublePendulumSystemS7": Reset
    2023-05-03  10:50:48.475971  S  Scenario Base "????": Process time 0:00:00 : Start of processing
    2023-05-03  10:50:48.476972  S  Scenario Base "????": Process time 0:00:00 : Start of cycle 0
    2023-05-03  10:50:48.476972  I  Scenario Base "????": Generating new action
    2023-05-03  10:50:48.476972  I  System "DoublePendulumSystemS7": Start processing action
    2023-05-03  10:50:48.476972  I  System "DoublePendulumSystemS7": Actions of agent 0 = [-0.18259651632236285]
    2023-05-03  10:50:48.478971  I  System "DoublePendulumSystemS7": Assessment for success...
    2023-05-03  10:50:48.478971  I  System "DoublePendulumSystemS7": Assessment for breakdown...
    2023-05-03  10:50:48.478971  I  System "DoublePendulumSystemS7": Action processing finished successfully
    2023-05-03  10:50:48.480331  I  System "DoublePendulumSystemS7": Assessment for breakdown...
    2023-05-03  10:50:48.481324  S  Scenario Base "????": Process time 0:00:00 : End of cycle 0
    2023-05-03  10:50:48.481324  S  Scenario Base "????": Process time 0:00:00.040000 : Start of cycle 1
    2023-05-03  10:50:48.482329  I  Scenario Base "????": Generating new action
    2023-05-03  10:50:48.482329  I  System "DoublePendulumSystemS7": Start processing action
    2023-05-03  10:50:48.482329  I  System "DoublePendulumSystemS7": Actions of agent 0 = [-2.020357408450476]
    2023-05-03  10:50:48.489836  I  System "DoublePendulumSystemS7": Assessment for success... 
    2023-05-03  10:50:48.490859  I  System "DoublePendulumSystemS7": Assessment for breakdown...
    2023-05-03  10:50:48.492870  I  System "DoublePendulumSystemS7": Action processing finished successfully
    2023-05-03  10:50:48.493852  I  System "DoublePendulumSystemS7": Assessment for breakdown...
    2023-05-03  10:50:48.493852  S  Scenario Base "????": Process time 0:00:00.040000 : End of cycle 1
    2023-05-03  10:50:48.493852  S  Scenario Base "????": Process time 0:00:00.080000 : Start of cycle 2
    2023-05-03  10:50:48.493852  I  Scenario Base "????": Generating new action
    2023-05-03  10:50:48.494849  I  System "DoublePendulumSystemS7": Start processing action
    2023-05-03  10:50:48.494849  I  System "DoublePendulumSystemS7": Actions of agent 0 = [6.063718908910516]
    2023-05-03  10:50:48.497848  I  System "DoublePendulumSystemS7": Assessment for success...
    2023-05-03  10:50:48.497848  I  System "DoublePendulumSystemS7": Assessment for breakdown...
    2023-05-03  10:50:48.497848  I  System "DoublePendulumSystemS7": Action processing finished successfully
    2023-05-03  10:50:48.497848  I  System "DoublePendulumSystemS7": Assessment for breakdown...
    2023-05-03  10:50:48.498855  S  Scenario Base "????": Process time 0:00:00.080000 : End of cycle 2
    2023-05-03  10:50:48.498855  S  Scenario Base "????": Process time 0:00:00.120000 : Start of cycle 3
    2023-05-03  10:50:48.498855  I  Scenario Base "????": Generating new action
    2023-05-03  10:50:48.498855  I  System "DoublePendulumSystemS7": Start processing action
    2023-05-03  10:50:48.499854  I  System "DoublePendulumSystemS7": Actions of agent 0 = [11.548934045420527] 
    2023-05-03  10:50:48.501851  I  System "DoublePendulumSystemS7": Assessment for success...
    2023-05-03  10:50:48.502853  I  System "DoublePendulumSystemS7": Assessment for breakdown...
    2023-05-03  10:50:48.502853  I  System "DoublePendulumSystemS7": Action processing finished successfully
    2023-05-03  10:50:48.502853  I  System "DoublePendulumSystemS7": Assessment for breakdown...
    2023-05-03  10:50:48.502853  S  Scenario Base "????": Process time 0:00:00.120000 : End of cycle 3
    2023-05-03  10:50:48.503855  S  Scenario Base "????": Process time 0:00:00.160000 : Start of cycle 4
    2023-05-03  10:50:48.503855  I  Scenario Base "????": Generating new action
    2023-05-03  10:50:48.503855  I  System "DoublePendulumSystemS7": Start processing action
    2023-05-03  10:50:48.504868  I  System "DoublePendulumSystemS7": Actions of agent 0 = [-16.245616529030606]
    2023-05-03  10:50:48.507846  I  System "DoublePendulumSystemS7": Assessment for success...
    2023-05-03  10:50:48.507846  I  System "DoublePendulumSystemS7": Assessment for breakdown...
    2023-05-03  10:50:48.507846  I  System "DoublePendulumSystemS7": Action processing finished successfully
    2023-05-03  10:50:48.508850  I  System "DoublePendulumSystemS7": Assessment for breakdown...
    2023-05-03  10:50:48.508850  S  Scenario Base "????": Process time 0:00:00.160000 : End of cycle 4
    2023-05-03  10:50:48.508850  S  Scenario Base "????": Process time 0:00:00.200000 : Start of cycle 5
    2023-05-03  10:50:48.508850  I  Scenario Base "????": Generating new action
    2023-05-03  10:50:48.508850  I  System "DoublePendulumSystemS7": Start processing action
    2023-05-03  10:50:48.510360  I  System "DoublePendulumSystemS7": Actions of agent 0 = [-18.866100939119747]
    2023-05-03  10:50:48.512369  I  System "DoublePendulumSystemS7": Assessment for success...
    2023-05-03  10:50:48.512369  I  System "DoublePendulumSystemS7": Assessment for breakdown...
    2023-05-03  10:50:48.513373  I  System "DoublePendulumSystemS7": Action processing finished successfully
    2023-05-03  10:50:48.513373  I  System "DoublePendulumSystemS7": Assessment for breakdown...
    2023-05-03  10:50:48.513373  S  Scenario Base "????": Process time 0:00:00.200000 : End of cycle 5
    2023-05-03  10:50:48.514376  S  Scenario Base "????": Process time 0:00:00.240000 : Start of cycle 6
    2023-05-03  10:50:48.514376  I  Scenario Base "????": Generating new action
    2023-05-03  10:50:48.515382  I  System "DoublePendulumSystemS7": Start processing action 
    2023-05-03  10:50:48.515382  I  System "DoublePendulumSystemS7": Actions of agent 0 = [13.430604156794786]
    2023-05-03  10:50:48.519427  I  System "DoublePendulumSystemS7": Assessment for success...
    2023-05-03  10:50:48.520390  I  System "DoublePendulumSystemS7": Assessment for breakdown...
    2023-05-03  10:50:48.521408  I  System "DoublePendulumSystemS7": Action processing finished successfully
    2023-05-03  10:50:48.522373  I  System "DoublePendulumSystemS7": Assessment for breakdown...
    2023-05-03  10:50:48.522373  S  Scenario Base "????": Process time 0:00:00.240000 : End of cycle 6
    2023-05-03  10:50:48.522373  S  Scenario Base "????": Process time 0:00:00.280000 : Start of cycle 7
    2023-05-03  10:50:48.523373  I  Scenario Base "????": Generating new action
    2023-05-03  10:50:48.523373  I  System "DoublePendulumSystemS7": Start processing action
    2023-05-03  10:50:48.524382  I  System "DoublePendulumSystemS7": Actions of agent 0 = [-2.689317283797866]
    2023-05-03  10:50:48.528376  I  System "DoublePendulumSystemS7": Assessment for success...
    2023-05-03  10:50:48.528376  I  System "DoublePendulumSystemS7": Assessment for breakdown...
    2023-05-03  10:50:48.528376  I  System "DoublePendulumSystemS7": Action processing finished successfully
    2023-05-03  10:50:48.529374  I  System "DoublePendulumSystemS7": Assessment for breakdown...
    2023-05-03  10:50:48.529374  S  Scenario Base "????": Process time 0:00:00.280000 : End of cycle 7
    2023-05-03  10:50:48.529374  S  Scenario Base "????": Process time 0:00:00.320000 : Start of cycle 8
    2023-05-03  10:50:48.529374  I  Scenario Base "????": Generating new action
    2023-05-03  10:50:48.529374  I  System "DoublePendulumSystemS7": Start processing action
    2023-05-03  10:50:48.530707  I  System "DoublePendulumSystemS7": Actions of agent 0 = [10.491203298317679] 
    2023-05-03  10:50:48.532373  I  System "DoublePendulumSystemS7": Assessment for success...
    2023-05-03  10:50:48.532373  I  System "DoublePendulumSystemS7": Assessment for breakdown...
    2023-05-03  10:50:48.532373  I  System "DoublePendulumSystemS7": Action processing finished successfully
    2023-05-03  10:50:48.533372  I  System "DoublePendulumSystemS7": Assessment for breakdown...
    2023-05-03  10:50:48.533372  S  Scenario Base "????": Process time 0:00:00.320000 : End of cycle 8
    2023-05-03  10:50:48.533372  S  Scenario Base "????": Process time 0:00:00.360000 : Start of cycle 9
    2023-05-03  10:50:48.533372  I  Scenario Base "????": Generating new action
    2023-05-03  10:50:48.533372  I  System "DoublePendulumSystemS7": Start processing action
    2023-05-03  10:50:48.534372  I  System "DoublePendulumSystemS7": Actions of agent 0 = [-19.915757865955573]
    2023-05-03  10:50:48.536372  I  System "DoublePendulumSystemS7": Assessment for success...
    2023-05-03  10:50:48.537369  I  System "DoublePendulumSystemS7": Assessment for breakdown...
    2023-05-03  10:50:48.537369  I  System "DoublePendulumSystemS7": Action processing finished successfully
    2023-05-03  10:50:48.537369  I  System "DoublePendulumSystemS7": Assessment for breakdown...
    2023-05-03  10:50:48.537369  S  Scenario Base "????": Process time 0:00:00.360000 : End of cycle 9
    2023-05-03  10:50:48.538374  S  Scenario Base "????": Process time 0:00:00.400000 : End of processing


**Cross Reference**

+ :ref:`API Reference: Systems <target_ap_bf_systems>`