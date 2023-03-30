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
## -- 2023-01-27  1.2.0     MRD       Remove previous implementation and only define the MujocoHandler
## --                                 as the handler to handle the MuJoCo simulation
## -- 2023-02-13  1.2.1     MRD       Simplify State Space and Action Space generation
## -- 2023-02-23  1.2.2     MRD       Disable auto detect body position and orientation
## --                                 Detect Joint boundaries, default inf
## --                                 Disable custom reset from MLPro, now reset only from MuJoCo
## -- 2023-03-08  1.2.3     MRD       Add get_latency() function to get latency from xml
## -------------------------------------------------------------------------------------------------


"""
Ver. 1.2.3  (2023-03-08)

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
            self.cam.lookat = np.zeros(3)
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
                p_state_mapping=None,
                p_action_mapping=None,
                p_camera_conf=(None, None, None),
                p_visualize=False,
                p_logging=Log.C_LOG_ALL
                ):
        
        self._viewer = None
        self._frame_skip = p_frame_skip
        self._xyz_camera, self._distance_camera, self._elavation_camera = p_camera_conf
        self._model_path = p_mujoco_file
        self._visualize = p_visualize

        self._system_state_space = None
        self._system_action_space = None
        self._state_mapping = p_state_mapping
        self._action_mapping = p_action_mapping
        self._sim_obs_list_order = []
        self._sim_act_list_order = {}
        self._vis_obs_list_order = []

        self._initialize_simulation()

        self._init_qpos = self._data.qpos.ravel().copy()
        self._init_qvel = self._data.qvel.ravel().copy()
        
        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.parse(self._model_path, parser=parser)
        self._xml_root = tree.getroot()
        etree.strip_tags(self._xml_root, etree.Comment)
        etree.cleanup_namespaces(self._xml_root)
        
        Wrapper.__init__(self, p_logging)


## -------------------------------------------------------------------------------------------------
    def __del__(self):
        try:
            self._close()
            self.log(self.C_LOG_TYPE_I, 'Closed')
        except:
            pass
        

## -------------------------------------------------------------------------------------------------
    def setup_spaces(self):
        return self._get_state_space(), self._get_action_space()
    
    
## -------------------------------------------------------------------------------------------------
    def _get_state_space(self):
        self._system_state_space = ESpace()
        
        # # Extract Position and Orientation, if a body
        # for elem in self._xml_root.iter("body"):
        #     self._system_state_space.add_dim(p_dim = Dimension(p_name_short=elem.attrib["name"]+str(".pos.body")))
        #     self._system_state_space.add_dim(p_dim = Dimension(p_name_short=elem.attrib["name"]+str(".rot.body")))
        
        # Extract Position, Velocity, and Acceleration, if a joint
        for world_body_elem in self._xml_root.iter("worldbody"):
            for elem in world_body_elem.iter("joint"):
                try:
                    bound = elem.attrib["range"].split(" ")
                    self._system_state_space.add_dim(p_dim = Dimension(p_name_short=elem.attrib["name"]+str(".pos.joint"), p_boundaries=[float(bound[0]), float(bound[1])]))
                except KeyError as e:
                    self._system_state_space.add_dim(p_dim = Dimension(p_name_short=elem.attrib["name"]+str(".pos.joint"), p_boundaries=[float('inf'), float('inf')]))
                
                self._system_state_space.add_dim(p_dim = Dimension(p_name_short=elem.attrib["name"]+str(".vel.joint"), p_boundaries=[float('inf'), float('inf')]))
                self._system_state_space.add_dim(p_dim = Dimension(p_name_short=elem.attrib["name"]+str(".acc.joint"), p_boundaries=[float('inf'), float('inf')]))
        
        return self._system_state_space
    
    
## -------------------------------------------------------------------------------------------------
    def _get_action_space(self):
        self._system_action_space = ESpace()
        
        # Actuator list
        for elem in self._xml_root.iter("motor"):
            try:
                bound = elem.attrib["ctrlrange"].split(" ")
                self._system_action_space.add_dim(p_dim = Dimension(p_name_short=elem.attrib["name"], p_boundaries=[float(bound[0]), float(bound[1])]))
            except KeyError as e:
                self._system_action_space.add_dim(p_dim = Dimension(p_name_short=elem.attrib["name"], p_boundaries=[float('inf'), float('inf')]))
            
        return self._system_action_space


## ------------------------------------------------------------------------------------------------------
    def _get_obs(self):
        state_value = []
        for dim in self._system_state_space.get_dims():
            state = dim.get_name_short().split(".")
            if state[2] == "body":
                if state[1] == "pos":
                    state_value.append(self._data.body(state[0]).xpos)
                elif state[1] == "rot":
                    state_value.append(self._data.body(state[0]).xquat)
            elif state[2] == "joint":
                if state[1] == "pos":
                    state_value.append(self._data.joint(state[0]).qpos)
                elif state[1] == "vel":
                    state_value.append(self._data.joint(state[0]).qvel)
                elif state[1] == "acc":
                    state_value.append(self._data.joint(state[0]).qacc)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

        return np.concatenate(state_value).ravel()


## -------------------------------------------------------------------------------------------------
    def _reset_model(self, reset_state=None):
        qpos = self._init_qpos
        qvel = self._init_qpos
        self._set_state(qpos, qvel)
        # if reset_state is None:
        #     qpos = self._init_qpos
        #     qvel = self._init_qpos
        #     self._set_state(qpos, qvel)
        # else:
        #     for dim in self._system_state_space.get_dims():
        #         state = dim.get_name_short().split("_")
        #         if state[2] == "body":
        #             pass
        #         elif state[2] == "joint":
        #             if state[1] == "pos":
        #                 self._data.joint(state[0]).qpos = reset_state.get_value(dim.get_id())
        #             elif state[1] == "vel":
        #                 self._data.joint(state[0]).qvel = reset_state.get_value(dim.get_id())
        #             elif state[1] == "acc":
        #                 self._data.joint(state[0]).qacc = reset_state.get_value(dim.get_id())
        #             else:
        #                 raise NotImplementedError
        #         else:
        #             raise NotImplementedError
            
        #     if self._model.na == 0:
        #         self._data.act[:] = None
        #     mujoco.mj_forward(self._model, self._data)
                
        return self._get_obs()


## -------------------------------------------------------------------------------------------------    
    def _set_state(self, *args):
        if len(args) == 2:
            self._data.qpos[:] = np.copy(args[0])
            self._data.qvel[:] = np.copy(args[1])
        
        if len(args) == 3:
            self._data.qpos[:] = np.copy(args[0])
            self._data.qvel[:] = np.copy(args[1])
            self._data.qacc[:] = np.copy(args[2])

        if self._model.na == 0:
            self._data.act[:] = None
        mujoco.mj_forward(self._model, self._data)


## -------------------------------------------------------------------------------------------------    
    def _initialize_simulation(self):
        self._model = mujoco.MjModel.from_xml_path(self._model_path)
        self._model.vis.global_.offwidth = 480
        self._model.vis.global_.offheight = 480
        self._data = mujoco.MjData(self._model)


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
        self._data.ctrl[:] = p_action
        mujoco.mj_step(self._model, self._data, nstep=self._frame_skip)
        mujoco.mj_rnePostConstraint(self._model, self._data)
        if self._visualize:
            self.render()


## -------------------------------------------------------------------------------------------------
    def render(self):
        self._get_viewer().render()
        
        
## -------------------------------------------------------------------------------------------------
    def get_latency(self):
        for option_elem in self._xml_root.iter("option"):
            try:
                timestep = option_elem.attrib["timestep"]
                return float(timestep)
            except KeyError as e:
                return None
            

## -------------------------------------------------------------------------------------------------
    def _close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
