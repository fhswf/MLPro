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



# ## -------------------------------------------------------------------------------------------------
#     def init_plot(self, p_figure: Figure = None, p_plot_settings: list = ..., p_plot_depth: int = 0, p_detail_level: int = 0, p_step_rate: int = 0, **p_kwargs):
#         if self._mujoco_handler._system_type != WrMujoco.C_VISUALIZE:
#             if self._visualize: 
#                 self._mujoco_handler._render()
#         elif self._mujoco_handler._visualize:
#             state_value = []
#             current_state = self.get_state()

#             if callable(getattr(self._sys_env, '_obs_to_mujoco', None)):
#                 current_state = self._sys_env._obs_to_mujoco(current_state)

#             try:
#                 for state_name in self._vis_state_name_list:
#                     state_id = self.get_state_space().get_dim_by_name(state_name).get_id()
#                     state_value.append(current_state.get_value(state_id))
#             except:
#                 raise Error("Name of the state is not valid")

#             if not self._mujoco_handler._use_radian:
#                 state_value = list(map(math.radians, state_value))

#             self._mujoco_handler._set_state(state_value, np.zeros(len(state_value)))

#             self._mujoco_handler._render()


# ## -------------------------------------------------------------------------------------------------
#     def update_plot(self, **p_kwargs):
#         if self._mujoco_handler._system_type != WrMujoco.C_VISUALIZE:
#             if self._visualize: 
#                 self._mujoco_handler._render()
#         elif self._mujoco_handler._visualize:
#             state_value = []
#             current_state = self.get_state()

#             if callable(getattr(self._sys_env, '_obs_to_mujoco', None)):
#                 current_state = self._sys_env._obs_to_mujoco(current_state)

#             for state_name in self._vis_state_name_list:
#                 state_id = self.get_state_space().get_dim_by_name(state_name).get_id()
#                 state_value.append(current_state.get_value(state_id))

#             if not self._mujoco_handler._use_radian:
#                 state_value = list(map(math.radians, state_value))

#             self._mujoco_handler._set_state(state_value, np.zeros(len(state_value)))

#             self._mujoco_handler._render()




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MujocoHandler(Wrapper):
    """
    Module provides the functionality of MuJoCo
    """
    C_NAME = 'MuJoCo'
    C_TYPE = 'Wrapper MuJoCo -> MLPro'
    C_WRAPPED_PACKAGE   = 'mujoco'
    C_MINIMUM_VERSION = '2.3.1'


    ## -------------------------------------------------------------------------------------------------
    def __init__(self, 
                p_mujoco_file, 
                p_frame_skip, 
                p_system_state_space=None,
                p_system_action_space=None,
                p_state_mapping=None,
                p_action_mapping=None,
                p_use_radian=True, 
                p_camera_conf=(None, None, None),
                p_visualize=False,
                p_logging=Log.C_LOG_ALL
                ):
        
        self._viewer = None
        self._frame_skip = p_frame_skip
        self._xyz_camera, self._distance_camera, self._elavation_camera = p_camera_conf
        self._model_path = p_mujoco_file
        self._visualize = p_visualize

        self._system_state_space = p_system_state_space
        self._system_action_space = p_system_action_space
        self._state_mapping = p_state_mapping
        self._action_mapping = p_action_mapping
        self._use_radian = p_use_radian
        self._sim_obs_list_order = []
        self._sim_act_list_order = {}
        self._vis_obs_list_order = []

        self._initialize_simulation()

        self._init_qpos = self._data.qpos.ravel().copy()
        self._init_qvel = self._data.qvel.ravel().copy()
        Wrapper.__init__(self, p_logging)


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

        obs_data_list = {}
        act_data_list = {}
        
        # Body list
        obs_data_list = {**obs_data_list, **dict([(elem.attrib["name"], "body") for elem in xml_root.iter("body")])}

        # Joint list
        obs_data_list = {**obs_data_list, **dict([(elem.attrib["name"], "joint") for elem in xml_root.iter("joint")])}

        # Actuator list
        idx = 0
        for elem in xml_root.iter("motor"):
            act_data_list = {**act_data_list, **dict([(elem.attrib["name"], idx)])}
            idx += 1
        
        for dim in self._system_action_space.get_dims():
            try:
                action_name = dim.get_name_short()
                self._sim_act_list_order[action_name] = act_data_list[action_name]
            except:
                raise Error("Name of the action is not valid")
            

        # Create tuple in order with the system space
        if self._state_mapping is not None:
            state_mapping =  dict((x, y) for x, y in self._state_mapping)
            try:
                for dim in self._system_state_space.get_dims():
                    state_name = state_mapping[dim.get_name_short()].split("_")
                    self._sim_obs_list_order.append((state_name[0], [obs_data_list[state_name[0]], state_name[1]]))
            except:
                raise Error("Name of the state is not valid")
        else:
            # This is when there is no state mapping or the name of the state is already in correct naming
            try:
                for dim in self._system_state_space.get_dims():
                    state_name = dim.get_name_short().split("_")
                    self._sim_obs_list_order.append((state_name[0], [obs_data_list[state_name[0]], state_name[1]]))
            except:
                raise Error("Name of the state is not valid")

## -------------------------------------------------------------------------------------------------
    def _get_model_list_vis(self):
        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.parse(self._model_path, parser=parser)
        xml_root = tree.getroot()
        etree.strip_tags(xml_root, etree.Comment)
        etree.cleanup_namespaces(xml_root)

        obs_data_list = {}
        
        # Body list
        obs_data_list = {**obs_data_list, **dict([(elem.attrib["name"], "body") for elem in xml_root.iter("body")])}

        # Joint list
        obs_data_list = {**obs_data_list, **dict([(elem.attrib["name"], "joint") for elem in xml_root.iter("joint")])}

        # Create tuple in order with the system space
        for dim in self._system_state_space.get_dims():
            state_name = dim.get_name_short().split("_")
            self._vis_obs_list_order.append((state_name[0], [obs_data_list[state_name[0]], state_name[1]]))


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
    def _initialize_simulation(self):
        self._model = mujoco.MjModel.from_xml_path(self._model_path)
        self._model.vis.global_.offwidth = 480
        self._model.vis.global_.offheight = 480
        self._data = mujoco.MjData(self._model)
        
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
    def _step_simulation(self, p_action : Action):
        # Re arrange action
        old_action = p_action.get_sorted_values()
        new_action = [0 for _ in range(len(old_action))]
        

        idx = 0
        for ids in self._system_action_space.get_dim_ids():
            action_name = self._system_action_space.get_dim(ids).get_name_short()
            new_action[self._sim_act_list_order[action_name]] = old_action[idx]
            idx = idx + 1
        
        
        self._data.ctrl[:] = new_action
        mujoco.mj_step(self._model, self._data, nstep=self._frame_skip)
        mujoco.mj_rnePostConstraint(self._model, self._data)
        if self._visualize:
            self.render()


## -------------------------------------------------------------------------------------------------
    def render(self):
        self._get_viewer().render()


## -------------------------------------------------------------------------------------------------
    def _close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
