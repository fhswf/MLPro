`Bulk Good Laboratory Plant (BGLP) <https://github.com/fhswf/MLPro/blob/main/src/mlpro/rl/pool/envs/bglp.py>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    This BGLP environment can be installed via:

    .. code-block:: python
    
        import mlpro.rl.pool.envs.bglp
    
    - **3rd Party Dependency**
    
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
    
    .. note::
    
    	In this simulation, we assume that the actuator in Module D has a constant flow, which automatically match the production demand in L/s.
    	This parameter can be defined, while setting up the BGLP environment.
    	Therefore, 5 actuators are involved in this simulation instead of 6 actuators.
        
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
    
    In this environment, we consider 5 actuators to be controlled. 
    Thus, there are 5 agents and 5 joint actions because each agent requires an action.
    Every action is normalized within a range between 0 and 1, except for Agent 3.
    0 means the minimum possible action and 1 means the maximum possible action.
    For Agent 3, the vibratory conveyor has different characteristic than other actuators, which mostly perform in a continuous manner.
    The vibratory conveyor can only be either fully switched-on or switched-off. Therefore the base set of action for Agent 3 is integer (0/1).
    0 means off and 1 means on.
    
    +-------+-------------------+--------+-------------------+--------------+
    | Agent | Actuator          | Station| Parameter         | Boundaries   |
    +=======+===================+========+===================+==============+
    |   1   | Conveyor Belt     | A      | rpm               | 450 ... 1800 |
    +-------+-------------------+--------+-------------------+--------------+
    |   2   | Vacuum Pump       | B      | on-duration (sec) | 0 ... 4.575  |
    +-------+-------------------+--------+-------------------+--------------+
    |   3   | Vibratory Conveyor| B      | on/off            | 0/1          |
    +-------+-------------------+--------+-------------------+--------------+
    |   4   | Vacuum Pump       | C      | on-duration (sec) | 0 ... 9.5    |
    +-------+-------------------+--------+-------------------+--------------+
    |   5   | Rotary Feeder     | C      | rpm               | 450 ... 1450 |
    +-------+-------------------+--------+-------------------+--------------+
      
    - **State space**
    
    The state information in the BGLP are the fill levels of the reservoirs.
    Each agent is always placed in between two reservoirs, e.g. between a silo and a hopper or vice versa.
    Therefore, each agent has two state information, which are shared with their neighbours.
    Every state is normalized within a range between 0 and 1.
    0 means the minimum fill-level and 1 means the maximum fill-level.
    
    +------+----------+--------+--------+---------------+
    | Agent| State No.| Element| Station| Boundaries    |
    +======+==========+========+========+===============+
    |      | 1        | Silo   | A      | 0 ... 17.42 L |
    + 1    +----------+--------+--------+---------------+
    |      | 2        |        |        |               |
    +------+----------+ Hopper + A      + 0 ... 9.1 L   +
    |      | 1        |        |        |               |
    + 2    +----------+--------+--------+---------------+
    |      | 2        |        |        |               |
    +------+----------+ Silo   + B      + 0 ... 17.42 L +
    |      | 1        |        |        |               |
    + 3    +----------+--------+--------+---------------+
    |      | 2        |        |        |               |
    +------+----------+ Hopper + B      + 0 ... 9.1 L   +
    |      | 1        |        |        |               |
    + 4    +----------+--------+--------+---------------+
    |      | 2        |        |        |               |
    +------+----------+ Silo   + C      + 0 ... 17.42 L +
    |      | 1        |        |        |               |
    + 5    +----------+--------+--------+---------------+
    |      | 2        | Hopper | C      | 0 ... 9.1 L   |
    +------+----------+--------+--------+---------------+
      
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