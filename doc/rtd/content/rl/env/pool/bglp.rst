`Bulk Good Laboratory Plant (BGLP) <https://github.com/fhswf/MLPro/blob/main/src/mlpro/rl/pool/envs/bglp.py>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    This BGLP environment can be installed via:

    .. code-block:: python
    
        import mlpro.rl.pool.envs.bglp
    
    - **3rd Party Dependencies**
    
        - NumPy
    
    - **Overview**
    
    text-align:justify
    The BGLP illustrates a smart production system with high flexibility and distributed control to transport bulk raw materials.
    One of the advantages of this laboratory test belt is the modularity in design, as depicted schematically below:
    
    .. image:: images/BGLP_Scheme.jpg
        :width: 400
    
    The BGLP consists of four modules, which are loading, storing, weighing, and filling stations respectively, and has conveying and dosing units as integral parts of the system.
    The interface between the modules is assembled via a mini hopper placed in the prior module. 
    Then, the next module is fed by a vacuum pump, which operates in a discontinuous manner, before the goods are temporary stored in a silo of the next module. 
    The filling station has no silo because the main purpose of the station is to occupy the transport containers.
    
    We utilize dissimilar actuators in modules 1-3 to transport the goods from the silo to the mini hopper. 
    Module 1 utilizes a belt conveyor, that operates between 0 and 1800 rpm. 
    Module 2 uses a vibratory conveyor, which can be completely switched on and off. 
    Lastly, Module 3 utilizes a rotary feeder, that operates between 0 and 1450 rpm.
    
    In RL context, we consider the BGLP as a multi-agents system, where each actuator of the system is pointed as an agent or a player.
    The states information for each agent are the fill level of the prior reservoir and the fill level of the next reservoir.
    
        
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