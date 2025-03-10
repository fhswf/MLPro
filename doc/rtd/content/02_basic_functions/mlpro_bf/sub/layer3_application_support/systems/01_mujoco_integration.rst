.. _target_bf_systems_mujoco:
MuJoCo integration
==================

|pic1| |pic2|

.. |pic1| image:: images/mujoco_cartpole.gif
   :width: 20%

.. |pic2| image:: images/mujoco_doublependulum.gif
   :width: 20%


MuJoCo is a well-known physics engine for its fast and accurate simulation. The aim is to facilitate research and development in robotics, biomechanics, graphics 
and animation, and other areas. More explanation about MuJoCo can be found in `here <https://mujoco.org/>`_.

In order to use the MuJoCo integration in MLPro, the following steps need to be done:

    * **Install MLPro-Int-MuJoCo**

        `MLPro-Int-MuJoCo <https://mlpro-int-mujoco.readthedocs.io>`_ is a wrapper to integrate the MuJoCo package to MLPro.

        Before starting, please install the latest versions of MLPro and MuJoCo as follows:

        .. code-block:: bash

          pip install mlpro-int-mujoco[full] --upgrade

    * **Create a MuJoCo model**

        Create a MuJoCo model file accordingly to your design. Some example model are published by MuJoCo and can be accessed `here <https://mujoco.readthedocs.io/en/latest/models.html>`_.
        Below is an example of MuJoCo model file.

        .. code-block:: xml

            <mujoco>
              <option timestep="0.05" gravity="0 0 -9.81" integrator="RK4">
                <flag sensornoise="enable" energy="enable"/>
              </option>
              <worldbody>
                    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
                <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>

                <body name="link1" pos="0 0 2" euler="0 0 0">
                  <joint name="pin" type="hinge" axis = "0 -1 0" pos="0 0 0.5"/>
                  <inertial pos="0 0 0" mass="1" diaginertia="1 1 1" />
                  <geom type="cylinder" size="0.05 0.5" rgba="0 .9 0 1"/>
                </body>
                
              </worldbody>

              <actuator>
                <motor joint="pin" name="torque1" gear="1" ctrllimited="true" ctrlrange="-50 50"/>
              </actuator>
            </mujoco>

        The above model simulates one body called ``link1`` which has a cylindrical shape. It is attached to the world body with a joint called ``pin`` 
        with the type of hinge. This joint is controlled by an actuator called ``torque1`` boundaries between -50 and 50.

    * **Create a system**

        When you instantiate the System, put the MuJoCo model file path on ``p_mujoco_file``.
        If the model is correct and the path is correct, then the wrapper will automatically wrap the state and action space based on the MuJoCo model.
        
If you want to view your model only before including it in MLPro, you can use the MuJoCo tool by dragging and dropping the model file into it. The tool 
can be downloaded `here <https://github.com/deepmind/mujoco/releases>`_.


**Cross reference**

- `MLPro-Int-MuJoCo <https://mlpro-int-mujoco.readthedocs.io>`_
- `MuJoCo Tool <https://github.com/deepmind/mujoco/releases>`_
- `MuJoCo XML reference <https://mujoco.readthedocs.io/en/stable/XMLreference.html>`_
- `MuJoCo model samples <https://mujoco.readthedocs.io/en/latest/models.html>`_
- `Unity plug-in for MuJoCo <https://mujoco.readthedocs.io/en/stable/unity.html>`_