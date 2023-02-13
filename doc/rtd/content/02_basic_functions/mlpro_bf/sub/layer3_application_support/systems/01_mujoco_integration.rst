.. _target_bf_systems_mujoco:
MuJoCo Integration
==================

MuJoCo is a well-known physics engine for its fast and accurate simulation. The aim is to facilitate research and development in robotics, biomechanics, graphics 
and animation, and other areas. More explanation about MuJoCo can be found in `here <https://mujoco.org/>`_.

In order to use the MuJoCo integration in MLPro, the following steps need to be done:

    * **Create a MuJoCo Model**

        Create a MuJoCo model file accordingly to your design. Some example model are published by MuJoCo and can be accessed. `here <https://mujoco.readthedocs.io/en/latest/models.html>`_.

    * **Create a System**

        Create your system inherited from the bf. System class. Define the action and the state space. Ensure the action and state names match the joints and actuator names on the MuJoCo model file. 
        The following is an example of naming the action and state space accordingly.

        Here is the MuJoCo model file.

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

        The name of the first joint is "pin" (``<joint name="pin" type="hinge" axis = "0 -1 0" pos="0 0 0.5"/>``). A joint consists of position, velocity, and acceleration data. 
        In the current MuJoCo integration, only position and velocity can be retrieved. This is done by putting an indicator after the short name of a state dimension. 
        ``_vel`` is for velocity and ``_pos`` for position. For the actuator first actuator is named "pin" (``<motor joint="pin" name="torque1" gear="1" ctrllimited="true" ctrlrange="-50 50"/>``).
        So, in this case, the action dimension must be named "pin". An example of using the indicator is shown below code.

        .. code-block:: python

            @staticmethod
            def setup_spaces():
                
                # 1 State space
                state_space = ESpace()
                state_space.add_dim( p_dim = Dimension( p_name_short='pin_pos', p_name_long="Pin 1 Joint Angle") )

                state_space.add_dim( p_dim = Dimension( p_name_short='pin_vel', p_name_long="Pin 1 Angular Velocity") )

                # 2 Action space
                action_space = ESpace()
                action_space.add_dim( p_dim = Dimension( p_name_short='pin') )

                return state_space, action_space

        Failing to follow the above structure will fail to integrate MuJoCo into the System.

**Cross Reference**

- Please refer to :ref:`Howto BF SYSTEMS 002 <Howto BF SYSTEMS 002>` to know more about MuJoCo integration functionality in MLPro
- For further implementation as Environment on Reinforcement Learning, please refer to :ref:`Howto Env RL 007 <Howto Env RL 007>`
