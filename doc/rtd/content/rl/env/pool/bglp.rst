`Bulk Good Laboratory Plant (BGLP) <https://github.com/fhswf/MLPro/blob/main/src/mlpro/rl/pool/envs/bglp.py>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    This BGLP environment can be installed via:

    .. code-block:: python
    
        import mlpro.rl.pool.envs.bglp
    
    - **3rd Party Dependencies**
    
    add text here!
    
    - **Overview**
    
    add text here!
      
    - **General information**
    
    +------------------------------------+-------------------------------------------------------+
    |         Parameter                  |                         Value                         |
    +====================================+=======================================================+
    | Agents                             | 5                                                     |
    +------------------------------------+-------------------------------------------------------+
    | Native Source                      | MLPro                                                 |
    +------------------------------------+-------------------------------------------------------+
    | Action Space Dimension             | [5,]                                                  |
    +------------------------------------+-------------------------------------------------------+
    | Action Space Base Set              | Real numbers, except Agent 3 uses Integer             |
    +------------------------------------+-------------------------------------------------------+
    | Action Space Boundaries            | [0,1]                                                 |
    +------------------------------------+-------------------------------------------------------+
    | State Space Dimension              | [6,]                                                  |
    +------------------------------------+-------------------------------------------------------+
    | State Space Base Set               | Real numbers                                          |
    +------------------------------------+-------------------------------------------------------+
    | State Space Boundaries             | [0,1]                                                 |
    +------------------------------------+-------------------------------------------------------+
    | Reward Structure                   | Individual reward for each agent                      |
    +------------------------------------+-------------------------------------------------------+
      
    - **Action space**
    
    add text here!
      
    - **State space**
    
    add text here!
      
    - **Reward structure**
    
    add text here!
      
    - **Version structure**
    
        + Version 1.4.4 : Enhanchement, debug, refactoring, adding batch production scenario in MLPro v. 0.0.0
        + Version 1.0.0 : Initial version release in MLPro v. 0.0.0
        
    If you apply this environment in your research or work, please kindly cite the following related paper:
    
    .. code-block:: bibtex

     @article{Schwung2021,
      title={Decentralized learning of energy optimal production policies using PLC-informed reinforcement learning},
      author={Dorothea Schwung and Steve Yuwono and Andreas Schwung and Steven X. Ding},
      journal={Comput. Chem. Eng.},
      year={2021},
      volume={152},
      pages={107382}
      }