## -------------------------------------------------------------------------------------------------
## -- Project : FH-SWF Automation Technology - Common Code Base (CCB)
## -- Package : mlpro
## -- Module  : mujoco.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-09-17  0.0.0     MRD       Creation
## -- 2022-12-11  0.0.1     MRD       Refactor due to new bf.Systems
## -- 2022-12-11  1.0.0     MRD       First Release
## -- 2023-01-06  1.1.0     MRD       The wrapper now has flexibility. It can now be used for System 
## --                                 and Environment. Can now be used for visualization only to
## --                                 visualize current state. The camera configuration now can be
## --                                 configured by the user.
## -- 2023-01-07  1.1.1     MRD       Wrap original reset to preserve custom reset from the
## --                                 orignal reset. Add functionality to call a function when
## --                                 There is different between MuJoCo dimension and Environment
## --                                 dimension. Add auto mapping state space between MuJoCo and
## --                                 Environment
## -------------------------------------------------------------------------------------------------


"""
Ver. 1.1.1  (2023-01-07)

This module wraps bf.Systems with MuJoCo Simulation functionality.
"""


import time
import glfw
import mujoco
import numpy as np
from threading import Lock
from lxml import etree
import math

from mlpro.rl.models import *
from mlpro.wrappers.models import Wrapper




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class CallbacksViewer():
    """
    All callbacks function for the viewer
    """

    def __init__(self) -> None:
        self._gui_lock = Lock()
        self._button_left_pressed = False
        self._button_right_pressed = False
        self._last_mouse_x = 0
        self._last_mouse_y = 0
        self._paused = False
        self._render_every_frame = True
        self._time_per_render = 1 / 60.0
        self._run_speed = 1.0
        self._loop_count = 0
        self._advance_by_one_step = False


## -------------------------------------------------------------------------------------------------
    def _key_callback(self, window, key, scancode, action, mods):
        if action != glfw.RELEASE:
            return
        # Pause
        elif key == glfw.KEY_SPACE and self._paused is not None:
            self._paused = not self._paused
        # Advances simulation by one step.
        elif key == glfw.KEY_RIGHT and self._paused is not None:
            self._advance_by_one_step = True
            self._paused = True
        # Quit
        if key == glfw.KEY_ESCAPE:
            glfw.destroy_window(self.window)
            glfw.terminate()


## -------------------------------------------------------------------------------------------------
    def _scroll_callback(self, window, x_offset, y_offset):
            with self._gui_lock:
                mujoco.mjv_moveCamera(
                    self.model, 
                    mujoco.mjtMouse.mjMOUSE_ZOOM, 
                    0, 
                    -0.05 * y_offset, 
                    self.scn, 
                    self.cam)


## -------------------------------------------------------------------------------------------------
    def _cursor_pos_callback(self, window, xpos, ypos):
        if not (self._button_left_pressed or self._button_right_pressed):
            return

        shift_pressed = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        if self._button_right_pressed:
            if shift_pressed:
                action = mujoco.mjtMouse.mjMOUSE_MOVE_H
            else:
                action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self._button_left_pressed:
            if shift_pressed:
                action = mujoco.mjtMouse.mjMOUSE_ROTATE_H
            else:
                action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        dx = int(self._scale * xpos) - self._last_mouse_x
        dy = int(self._scale * ypos) - self._last_mouse_y
        width, height = glfw.get_framebuffer_size(window)

        with self._gui_lock:
            mujoco.mjv_moveCamera(
                self.model, action, dx / height, dy / height, self.scn, self.cam
            )

        self._last_mouse_x = int(self._scale * xpos)
        self._last_mouse_y = int(self._scale * ypos)    


## -------------------------------------------------------------------------------------------------
    def _mouse_button_callback(self, window, button, act, mods):
        self._button_left_pressed = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        )
        self._button_right_pressed = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        )

        x, y = glfw.get_cursor_pos(window)
        self._last_mouse_x = int(self._scale * x)
        self._last_mouse_y = int(self._scale * y)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RenderViewer(CallbacksViewer):
    def __init__(self, model, data, xyz_pos=None, elevation=None, distance=None) -> None:
        super().__init__()

        self.model = model
        self.data = data

        # Init GLFW
        glfw.init()

        # Get Width and Height of monitor
        width, height = glfw.get_video_mode(glfw.get_primary_monitor()).size

        # Create Window
        self.window = glfw.create_window(
            width, height, "MuJoCo in MLPRo Viewer", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(self.window)
        window_width, _ = glfw.get_window_size(self.window)
        self._scale = framebuffer_width * 1.0 / window_width
        
        # Set Callbacks
        glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        glfw.set_key_callback(self.window, self._key_callback)

        # get viewport
        self.viewport = mujoco.MjrRect(0, 0, framebuffer_width, framebuffer_height)

        # overlay, markers
        self._overlays = {}

        mujoco.mj_forward(self.model, self.data)

        self.scn = mujoco.MjvScene(self.model, maxgeom=10000)
        self.cam = mujoco.MjvCamera()
        self.vopt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()
        self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        self._init_camera(xyz_pos, elevation, distance)


## -------------------------------------------------------------------------------------------------
    def _init_camera(self, xyz_pos=None, elevation=None, distance=None):
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.fixedcamid = -1
        
        # X Y Z Position of the camera
        if xyz_pos is None:
            for i in range(3):
                self.cam.lookat[i] = np.median(self.data.geom_xpos[:, i])
            self.cam.lookat[0] = 0
        else:
            for i in range(3):
                self.cam.lookat[i] = xyz_pos[i]

        # Camera Distance
        if distance is None:
            self.cam.distance = self.model.stat.extent * 3.0
        else:
            self.cam.distance = distance

        if elevation is None:
            self.cam.elevation = -20
        else:
            self.cam.elevation = elevation


## -------------------------------------------------------------------------------------------------
    def _create_overlays(self):
        """
        Should be user customizeable
        """
        pass


## -------------------------------------------------------------------------------------------------
    def render(self):
        def update():
            self._create_overlays()

            render_start = time.time()

            if self.window is None:
                return
            elif glfw.window_should_close(self.window):
                glfw.destroy_window(self.window)
                glfw.terminate()

            self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(self.window)

            with self._gui_lock:
                # update scene
                mujoco.mjv_updateScene(
                    self.model,
                    self.data,
                    self.vopt,
                    self.pert,
                    self.cam,
                    mujoco.mjtCatBit.mjCAT_ALL.value,
                    self.scn)

                # render
                mujoco.mjr_render(self.viewport, self.scn, self.con)

                # overlays
                for gridpos, [t1, t2] in self._overlays.items():
                        mujoco.mjr_overlay(
                            mujoco.mjtFontScale.mjFONTSCALE_150,
                            gridpos,
                            self.viewport,
                            t1,
                            t2,
                            self.con,
                        )   

                glfw.swap_buffers(self.window)

            glfw.poll_events()
            self._time_per_render = 0.9 * self._time_per_render + 0.1 * (
                time.time() - render_start
            )

            self._overlays.clear()

        if self._paused:
            while self._paused:
                update()
                if self._advance_by_one_step:
                    self._advance_by_one_step = False
                    break
        else:
            self._loop_count += self.model.opt.timestep / (self._time_per_render * self._run_speed)
            if self._render_every_frame:
                self._loop_count = 1
            while self._loop_count > 0:
                update()
                self._loop_count -= 1


## -------------------------------------------------------------------------------------------------
    def close(self):
        glfw.destroy_window(self.window)
        glfw.terminate()





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrMujocoWrapper(Wrapper):
    """
    Wrap native MLPRo System with MuJuCo functionality.
    """

    C_NAME = 'MuJoCo'
    C_TYPE = 'Wrapper MuJoCo -> MLPro'
    C_WRAPPED_PACKAGE   = 'mujoco'
    C_MINIMUM_VERSION = '2.3.1'

    C_PLOT_ACTIVE       = True


## ------------------------------------------------------------------------------------------------------
    def __init__(self, p_sys_env, p_mujoco_handler, p_vis_state_name, p_logging=Log.C_LOG_ALL):
        self._sys_env = p_sys_env
        self._vis_state_name_list = p_vis_state_name
        Wrapper.__init__(self, p_logging)
        self._mujoco_handler = p_mujoco_handler
        

## ------------------------------------------------------------------------------------------------------
    def setup_spaces(self):
        return self._sys_env.get_state_space(), self._sys_env.get_action_space()


## ------------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None) -> None:
        """
        This method is used to reset the environment. The environment is reset to the initial position set during
        the initialization of the environment.

        Parameters
        ----------
        p_seed : int, optional
            The default is None.

        """
        # Do the reset from original to detect customized reset by the environment
        self._sys_env._reset()

        if self._mujoco_handler._system_type != WrMujoco.C_VISUALIZE:
            current_state = self._sys_env.get_state()
            if callable(getattr(self._sys_env, '_obs_to_mujoco', None)):
                current_state = self._sys_env._obs_to_mujoco(current_state)
            
            ob = self._mujoco_handler._reset_simulation(current_state)
            
            self._state = State(self.get_state_space())
            self._state.set_values(ob)
        else:
            self._state = State(self._sys_env.get_state_space())
            self._state.set_values(self._sys_env.get_state().get_values())


## ------------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state:State, p_action:Action):
        """
        This method is used to calculate the next states of the system after a set of actions.

        Parameters
        ----------
        p_state : State
            current State.
            p_action : Action
                current Action.

        Returns
        -------
            _state : State
                Current states after the simulation of latest action on the environment.

        """
        if self._mujoco_handler._system_type != WrMujoco.C_VISUALIZE:
            action = p_action.get_sorted_values()

            self._mujoco_handler._step_simulation(action)

            # Delay because of the simulation
            time.sleep(self.get_latency().total_seconds())
            ob = self._mujoco_handler._get_obs()

            if callable(getattr(self._sys_env, '_obs_from_mujoco', None)):
                ob = self._sys_env._obs_from_mujoco(ob)

            current_state = State(self.get_state_space())
            current_state.set_values(ob)

            return current_state
        else:
            return self._sys_env._simulate_reaction(p_state, p_action)


## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure: Figure = None, p_plot_settings: list = ..., p_plot_depth: int = 0, p_detail_level: int = 0, p_step_rate: int = 0, **p_kwargs):
        if self._mujoco_handler._system_type != WrMujoco.C_VISUALIZE:
            if self._visualize: 
                self._mujoco_handler._render()
        elif self._mujoco_handler._visualize:
            state_value = []
            current_state = self.get_state()

            if callable(getattr(self._sys_env, '_obs_to_mujoco', None)):
                current_state = self._sys_env._obs_to_mujoco(current_state)

            try:
                for state_name in self._vis_state_name_list:
                    state_id = self.get_state_space().get_dim_by_name(state_name).get_id()
                    state_value.append(current_state.get_value(state_id))
            except:
                raise Error("Name of the state is not valid")

            if not self._mujoco_handler._use_radian:
                state_value = list(map(math.radians, state_value))

            self._mujoco_handler._set_state(state_value, np.zeros(len(state_value)))

            self._mujoco_handler._render()


## -------------------------------------------------------------------------------------------------
    def update_plot(self, **p_kwargs):
        if self._mujoco_handler._system_type != WrMujoco.C_VISUALIZE:
            if self._visualize: 
                self._mujoco_handler._render()
        elif self._mujoco_handler._visualize:
            state_value = []
            current_state = self.get_state()

            if callable(getattr(self._sys_env, '_obs_to_mujoco', None)):
                current_state = self._sys_env._obs_to_mujoco(current_state)

            for state_name in self._vis_state_name_list:
                state_id = self.get_state_space().get_dim_by_name(state_name).get_id()
                state_value.append(current_state.get_value(state_id))

            if not self._mujoco_handler._use_radian:
                state_value = list(map(math.radians, state_value))

            self._mujoco_handler._set_state(state_value, np.zeros(len(state_value)))

            self._mujoco_handler._render()


## -------------------------------------------------------------------------------------------------
    def _compute_reward(self, p_state_old, p_state_new):
        return self._sys_env._compute_reward(p_state_old, p_state_new)


## -------------------------------------------------------------------------------------------------
    def _compute_success(self, p_state):
        return self._sys_env._compute_success(p_state)


## -------------------------------------------------------------------------------------------------
    def _compute_broken(self, p_state):
        return self._sys_env._compute_broken(p_state)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrMujocoClassCreator():
    def __new__(cls, p_cls):
        class MujocoWrapper(WrMujocoWrapper, p_cls):
            C_NAME = p_cls.__name__
            def __init__(self, p_environment, p_mujoco_handler, p_vis_state_name, p_visualize, p_logging=Log.C_LOG_ALL):
                WrMujocoWrapper.__init__(self, p_environment, p_mujoco_handler, p_vis_state_name, p_logging)
                p_cls.__init__(self, p_mode=Mode.C_MODE_SIM, p_latency=None, p_visualize=p_visualize, p_logging=p_logging)

        return MujocoWrapper





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrMujoco():

    C_ENVIRONMENT = 0
    C_SYSTEM = 1
    C_VISUALIZE = 2

    def __new__(cls, 
                p_system, 
                p_model_file, 
                p_frame_skip=1, 
                p_system_type=C_ENVIRONMENT, 
                p_vis_state_name=None, 
                p_state_mapping=None,
                p_use_radian=True,
                p_visualize=False, 
                p_camera_conf=(None, None, None), 
                p_logging=Log.C_LOG_ALL
                ):

        if isinstance(p_system, Environment):
            wr_mujoco_class_creator = WrMujocoClassCreator(Environment)
        elif isinstance(p_system, System):
            wr_mujoco_class_creator = WrMujocoClassCreator(System)
        else:
            raise Error("Type of environment or system is not supported")

        # Create MuJoCo hanlder
        mujoco_handler = MujocoHandler(p_model_file, 
                                    p_frame_skip, 
                                    p_system_type=p_system_type,
                                    p_system_state_space=p_system.get_state_space(),
                                    p_state_mapping=p_state_mapping,
                                    p_use_radian=p_use_radian,
                                    p_visualize=p_visualize, 
                                    p_camera_conf=p_camera_conf)

        p_wrapped_obj = wr_mujoco_class_creator(p_system, mujoco_handler, p_vis_state_name, p_visualize, p_logging)

        # Set Latency for Simulation
        if p_system_type != WrMujoco.C_VISUALIZE:
            p_wrapped_obj.set_latency(timedelta(0,0.05,0))

        return p_wrapped_obj





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MujocoHandler:
    """
    Module provides the functionality of MuJoCo
    """
## -------------------------------------------------------------------------------------------------
    def __init__(self, 
                p_model_file, 
                p_frame_skip, 
                p_system_type=WrMujoco.C_ENVIRONMENT, 
                p_system_state_space=None,
                p_state_mapping=None,
                p_use_radian=True, 
                p_visualize=False, 
                p_camera_conf=(None, None, None)
                ):
        
        self._viewer = None
        self._frame_skip = p_frame_skip
        self._visualize = p_visualize
        self._system_type = p_system_type
        self._xyz_camera, self._distance_camera, self._elavation_camera = p_camera_conf
        self._model_path = p_model_file

        self._system_state_space = p_system_state_space
        self._state_mapping = p_state_mapping
        self._use_radian = p_use_radian
        self._model_data_list = {}
        self._sim_obs_list_order = []
        self._vis_obs_list_order = []

        self._initialize_simulation()

        self._init_qpos = self._data.qpos.ravel().copy()
        self._init_qvel = self._data.qvel.ravel().copy()


## -------------------------------------------------------------------------------------------------
    def __del__(self):
        try:
            self._close()
            self.log(self.C_LOG_TYPE_I, 'Closed')
        except:
            pass


## -------------------------------------------------------------------------------------------------
    def _get_model_list_sim(self):
        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.parse(self._model_path, parser=parser)
        xml_root = tree.getroot()
        etree.strip_tags(xml_root, etree.Comment)
        etree.cleanup_namespaces(xml_root)
        
        # Body list
        self._model_data_list = {**self._model_data_list, **dict([(elem.attrib["name"], "body") for elem in xml_root.iter("body")])}

        # Joint list
        self._model_data_list = {**self._model_data_list, **dict([(elem.attrib["name"], "joint") for elem in xml_root.iter("joint")])}

        # Create tuple in order with the system space
        if self._state_mapping is not None:
            state_mapping =  dict((x, y) for x, y in self._state_mapping)
            try:
                for dim in self._system_state_space.get_dims():
                    state_name = state_mapping[dim.get_name_short()].split("_")
                    self._sim_obs_list_order.append((state_name[0], [self._model_data_list[state_name[0]], state_name[1]]))
            except:
                raise Error("Name of the state is not valid")
        else:
            # This is when there is no state mapping or the name of the state is already in correct naming
            try:
                for dim in self._system_state_space.get_dims():
                    state_name = dim.get_name_short().split("_")
                    self._sim_obs_list_order.append((state_name[0], [self._model_data_list[state_name[0]], state_name[1]]))
            except:
                raise Error("Name of the state is not valid")

## -------------------------------------------------------------------------------------------------
    def _get_model_list_vis(self):
        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.parse(self._model_path, parser=parser)
        xml_root = tree.getroot()
        etree.strip_tags(xml_root, etree.Comment)
        etree.cleanup_namespaces(xml_root)
        
        # Body list
        self._model_data_list = {**self._model_data_list, **dict([(elem.attrib["name"], "body") for elem in xml_root.iter("body")])}

        # Joint list
        self._model_data_list = {**self._model_data_list, **dict([(elem.attrib["name"], "joint") for elem in xml_root.iter("joint")])}

        # Create tuple in order with the system space
        for dim in self._system_state_space.get_dims():
            state_name = dim.get_name_short().split("_")
            self._vis_obs_list_order.append((state_name[0], [self._model_data_list[state_name[0]], state_name[1]]))


## ------------------------------------------------------------------------------------------------------
    def _get_obs(self):
        # Get all the data according to the tuple order
        state_value = []
        for state in self._sim_obs_list_order:
            if state[1][0] == "joint":
                if state[1][1] == "pos":
                    if not self._use_radian:
                        state_value.append(np.degrees(self._data.joint(state[0]).qpos))
                    else:
                        state_value.append(self._data.joint(state[0]).qpos)
                elif state[1][1] == "vel":
                    if not self._use_radian:
                        state_value.append(np.degrees(self._data.joint(state[0]).qvel))
                    else:
                        state_value.append(self._data.joint(state[0]).qvel)
                else:
                    raise Error("State name is not compatible")

            elif state[1][0] == "body":
                if state[1][1] == "pos":
                    state_value.append(self._data.body(state[0]).xpos)
                elif state[1][1] == "rot":
                    state_value.append(self._data.body(state[0]).xquat)
                else:
                    raise Error("State name is not compatible")

        return np.concatenate(state_value).ravel()


## -------------------------------------------------------------------------------------------------
    def _reset_model(self, reset_state=None):
        if reset_state is None:
            qpos = self._init_qpos
            qvel = self._init_qpos
            self._set_state(qpos, qvel)
        else:
            if self._state_mapping is not None:
                state_mapping =  dict((x, y) for x, y in self._state_mapping)
            for dim in self._system_state_space.get_dims():
                if self._state_mapping is not None:
                    state = state_mapping[dim.get_name_short()].split("_")
                    if state[1] == "pos":
                        if not self._use_radian:
                            self._data.joint(state[0]).qpos = np.radians(reset_state.get_value(dim.get_id()))
                        else:
                            self._data.joint(state[0]).qpos = reset_state.get_value(dim.get_id())
                    elif state[1] == "vel":
                        if not self._use_radian:
                            self._data.joint(state[0]).qvel = np.radians(reset_state.get_value(dim.get_id()))
                        else:
                            self._data.joint(state[0]).qvel = reset_state.get_value(dim.get_id())
                else:
                    state = dim.get_name_short().split("_")
                    if state[1] == "pos":
                        if not self._use_radian:
                            self._data.joint(state[0]).qpos = np.radians(reset_state.get_value(dim.get_id()))
                        else:
                            self._data.joint(state[0]).qpos = reset_state.get_value(dim.get_id())
                    elif state[1] == "vel":
                        if not self._use_radian:
                            self._data.joint(state[0]).qvel = np.radians(reset_state.get_value(dim.get_id()))
                        else:
                            self._data.joint(state[0]).qvel = reset_state.get_value(dim.get_id())
                
        return self._get_obs()


## -------------------------------------------------------------------------------------------------    
    def _set_state(self, *args):
        if len(args) == 2:
            self._data.qpos[:] = np.copy(args[0])
            self._data.qvel[:] = np.copy(args[1])

        if self._model.na == 0:
            self._data.act[:] = None
        mujoco.mj_forward(self._model, self._data)


## -------------------------------------------------------------------------------------------------    
    def _set_action_space(self):
        bounds = self._model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        return low, high


## -------------------------------------------------------------------------------------------------    
    def _initialize_simulation(self):
        self._model = mujoco.MjModel.from_xml_path(self._model_path)
        self._model.vis.global_.offwidth = 480
        self._model.vis.global_.offheight = 480
        self._data = mujoco.MjData(self._model)
        
        if self._system_type == WrMujoco.C_VISUALIZE:
            # self._get_model_list_vis()
            pass
        else:
            self._get_model_list_sim()

## -------------------------------------------------------------------------------------------------    
    def _get_viewer(self):
        if self._viewer is None:
            self._viewer = RenderViewer(self._model, self._data, self._xyz_camera, self._distance_camera, self._elavation_camera)
        
        return self._viewer


## -------------------------------------------------------------------------------------------------
    def _reset_simulation(self, reset_state=None):
        mujoco.mj_resetData(self._model, self._data)
        ob = self._reset_model(reset_state)
        return ob


## -------------------------------------------------------------------------------------------------
    def _step_simulation(self, action):
        self._data.ctrl[:] = action
        mujoco.mj_step(self._model, self._data, nstep=self._frame_skip)
        mujoco.mj_rnePostConstraint(self._model, self._data)


## -------------------------------------------------------------------------------------------------
    def _render(self):
        self._get_viewer().render()


## -------------------------------------------------------------------------------------------------
    def _close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
