.. _target_bf_systems_mujoco:
MuJoCo Integration
==================

MuJoCo is a one of well-known physics engine for its fast and accurate simulation. The aim is to facilitate research and development in robotics, biomechanis, graphics
and animation, and other areas. More explanation about MuJoCo can be found in `here <https://mujoco.org/>`_.

In order to use the MuJoCo integration in MLPro, the following steps need to be done:

    * **Create a MuJoCo Model**

        Create a MuJoCo model file accordingly to your design. Some of example model are published by MuJoCo and can be accessed `here <https://mujoco.readthedocs.io/en/latest/models.html>`_.

    * **Create a System**

        Create your own system inherited from the bf.System class. Define the action and the state space. Make sure the action and the state name are matched with the joints and actuator name on the MuJoCo model file.
        The following is an example of naming the action and state space accordingly.

        .. list-table:: State and Action Space MuJoCo and System
            :widths: 25 25 25
            :header-rows: 1

            * - Description
              - System
              - MuJoCo
            * - Joint named "pin1" for position
              - pin1_pos
              - pin1
            * - Joint named "pin1" for velocity
              - pin1_vel
              - pin1


By providing the ``p_mujoco_file`` with the MJCF model file, the base system will use MuJoCo engine as its state-based processor. 


**Cross Reference**

- Please refer to :ref:`Howto Howto BF SYSTEMS 002 <Howto BF SYSTEMS 002>` to know more about MuJoCo integration functionality in MLPro
