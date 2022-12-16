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
## -------------------------------------------------------------------------------------------------


"""
Ver. 1.0.0 (2022-12-11)

This module wraps bf.Systems with MuJoCo Simulation functionality.
"""


import time
import glfw
import os
import mujoco
import numpy as np
from threading import Lock

import mlpro
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
    def __init__(self, model, data) -> None:
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


## -------------------------------------------------------------------------------------------------
    def _init_camera(self):
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.fixedcamid = -1
        for i in range(3):
            self.cam.lookat[i] = np.median(self.data.geom_xpos[:, i])
        self.cam.distance = self.model.stat.extent


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
class WrSysMujoco(Wrapper, System):
    """
    Wrap native MLPRo System with MuJuCo functionality.
    """

    C_NAME = 'MuJoCo'
    C_TYPE = 'Wrapper MuJoCo -> MLPro'
    C_WRAPPED_PACKAGE   = 'mujoco'
    C_MINIMUM_VERSION = '2.3.1'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_model_file, p_frame_skip, p_model_path=None, p_visualize=False, p_logging=Log.C_LOG_ALL ):

        self._viewer = None
        self._frame_skip = p_frame_skip

        if p_model_path is None:
            self._model_path = os.path.join(os.path.dirname(mlpro.__file__), "rl/pool/envs/mujoco/assets", p_model_file)
        else:
            self._model_path = os.path.join(p_model_path, p_model_file)

        self._initialize_simulation()

        self._init_qpos = self._data.qpos.ravel().copy()
        self._init_qvel = self._data.qvel.ravel().copy()

        System.__init__(self, p_mode=Mode.C_MODE_SIM, p_latency=None, p_visualize=p_visualize, p_logging=p_logging)
        Wrapper.__init__(self, p_logging=p_logging)


## -------------------------------------------------------------------------------------------------
    def __del__(self):
        try:
            self._close()
            self.log(self.C_LOG_TYPE_I, 'Closed')
        except:
            pass


## -------------------------------------------------------------------------------------------------    
    def set_state(self, qpos, qvel):
        self._data.qpos[:] = np.copy(qpos)
        self._data.qvel[:] = np.copy(qvel)
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

## -------------------------------------------------------------------------------------------------    
    def _get_viewer(self):
        if self._viewer is None:
            self._viewer = RenderViewer(self._model, self._data)
        
        return self._viewer


## -------------------------------------------------------------------------------------------------
    def _get_simulation_state(self):
        pass


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
        
        ob = self._reset_simulation()

        self._state.set_values(ob)


## -------------------------------------------------------------------------------------------------
    def _reset_model(self):
        raise NotImplementedError


## -------------------------------------------------------------------------------------------------
    def _reset_simulation(self):
        mujoco.mj_resetData(self._model, self._data)
        ob =  self._reset_model()

        if self.get_visualization():
            self._render()
        return ob


## -------------------------------------------------------------------------------------------------
    def _step_simulation(self, action):
        self._data.ctrl[:] = action
        mujoco.mj_step(self._model, self._data, nstep=self._frame_skip)
        mujoco.mj_rnePostConstraint(self._model, self._data)

        if self.get_visualization():
            self._render()


## ------------------------------------------------------------------------------------------------------
    def _get_obs(self):
        raise NotImplementedError


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
        action = p_action.get_sorted_values()
        self._step_simulation(action)
        ob = self._get_obs()

        current_state = State(self._state_space)
        current_state.set_values(ob)

        return current_state


## -------------------------------------------------------------------------------------------------
    def _render(self):
        self._get_viewer().render()


## -------------------------------------------------------------------------------------------------
    def _close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure: Figure = None, p_plot_settings: list = ..., p_plot_depth: int = 0, p_detail_level: int = 0, p_step_rate: int = 0, **p_kwargs):
        if self._visualize: self._render()


## -------------------------------------------------------------------------------------------------
    def update_plot(self, **p_kwargs):
        if self._visualize: self._render()
