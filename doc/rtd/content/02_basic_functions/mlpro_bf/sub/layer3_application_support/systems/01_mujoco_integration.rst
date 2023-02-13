.. _target_bf_systems_mujoco:
MuJoCo Integration
==================

MuJoCo is a well-known physics engine for its fast and accurate simulation. The aim is to facilitate research and development in robotics, biomechanics, graphics 
and animation, and other areas. More explanation about MuJoCo can be found in `here <https://mujoco.org/>`_.

In order to use the MuJoCo integration in MLPro, the following steps need to be done:

    * **Create a MuJoCo Model**

        Create a MuJoCo model file accordingly to your design. Some example model are published by MuJoCo and can be accessed `here <https://mujoco.readthedocs.io/en/latest/models.html>`_.

        Here is an example of MuJoCo model file.

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

    * **Create a System**

        Create your system inherited from the bf.System class. When you instatiated your custom System, put the MuJoCo model file path on ``p_mujoco_file``.
        If the model is correct and the path is correct, then the wrapper will automatically wrap the state and action space based on the MuJoCo model.
        

**Cross Reference**

- `MuJoCo Model Samples <https://mujoco.readthedocs.io/en/latest/models.html>`_
- :ref:`Howto BF System 002 <Howto BF SYSTEMS 002>`
- :ref:`Howto RL Env 007 <Howto Env RL 007>`
- :ref:`MuJoCo Wrapper <Wrapper MuJoCo>`
