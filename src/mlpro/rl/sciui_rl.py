## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.rl
## -- Module  : sciui_rl.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-07-30  0.0.0     SY       Creation
## -- 2021-10-07  1.0.0     SY       Release of first draft
## -- 2022-01-06  1.0.1     DA       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2022-01-06)

This file provides reusable SciUI components for SciUI_RL.
"""



from tkinter import *
import tkinter
import tkinter.font as tkFont
from tkinter.ttk import *
from tkinter.scrolledtext import *
from mpl_toolkits.mplot3d import Axes3D

from mlpro.bf.ui.sciui.framework import *
from mlpro.bf.math import ESpace
import mlpro.rl.pool.scenarios as ScePool
from mlpro.rl.models import *
import os
import pkgutil
import numpy as np



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLDetails(SciUIFrameParam):
    """
    ...
    """

    C_NAME          = 'Details'
    C_SCE_LISTS     = []

## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()
        self.first_refresh  = True
        self.sce_var        = StringVar()
        self.env_var        = StringVar()
        self.alg_var        = StringVar()
        
        self.auto_scan_sce  = Scenario.__subclasses__()
        for i in range(len(self.auto_scan_sce)):
            if self.auto_scan_sce[i].C_NAME != '????':
                self.C_SCE_LISTS.append(self.auto_scan_sce[i].C_NAME)  


## -------------------------------------------------------------------------------------------------
    def refresh(self, p_parent_frame):
        if self.first_refresh: 
            self.first_refresh = False
        else: 
            if self.shared_db.rl_sce_state_btn == False:
                self.button_sce.config(state="disabled")
                

        super().refresh(p_parent_frame=p_parent_frame)
        
        self.sce_label      = Label(self.frame, text='Scenario', pady=3, font=self.fontstyle, width=20, height=1, anchor='w').grid(row=0, column=0, sticky='W')
        self.sce_box        = Entry(self.frame, textvariable=self.sce_var, width=28, justify='left', state='disabled', background='white', font=self.fontstyle)
        self.sce_box.grid(row=0, column=1)
        self.sce_var.trace('w', self.__cb_sce_changed)
        self.sce_button     = Frame(self.frame, highlightbackground = "black", bd=0, width=7)
        self.sce_button.grid(row=0, column=2, sticky='w')
        self.button_sce     = Button(self.sce_button, text='...', font=self.fontstyle, fg = 'black', command=self.__cb_sce_button)
        self.button_sce.pack()
        self.sce_i_button   = Frame(self.frame, highlightbackground = "black", bd=0, width=5)
        self.sce_i_button.grid(row=0, column=3, sticky='w')
        self.button_sce_i   = Button(self.sce_i_button, text='i', font=self.fontstyle, fg = 'black', state='disabled', command=self.__cb_sce_button_info)
        self.button_sce_i.pack()
        
        self.env_label      = Label(self.frame, text='Environment', pady=3, font=self.fontstyle, width=20, height=1, anchor='w').grid(row=1, column=0, sticky='W')
        self.env_box        = Entry(self.frame, textvariable=self.env_var, width=28, justify='left', state='disabled', background='white', font=self.fontstyle)
        self.env_box.grid(row=1, column=1)
        self.env_var.trace('w', self.__cb_env_changed)
        
        self.alg_label      = Label(self.frame, text='Algorithm', pady=3, font=self.fontstyle, width=20, height=1, anchor='w').grid(row=2, column=0, sticky='W')
        self.alg_box        = Entry(self.frame, textvariable=self.alg_var, width=28, justify='left', state='disabled', background='white', font=self.fontstyle)
        self.alg_box.grid(row=2, column=1)
        self.alg_var.trace('w', self.__cb_alg_changed)
        

## -------------------------------------------------------------------------------------------------
    def __cb_sce_changed(self, *args):
        try:
            sce   = self.sce_var.get()
        except:
            return

        self.sce_var.set(sce)
        self.env_var.set(self.shared_db.rl_env)
        self.alg_var.set(self.shared_db.rl_learning_alg)
        
        self.shared_db.rlui_selected_scenario = True
        self.shared_db.start_global_refresh()
        

## -------------------------------------------------------------------------------------------------
    def __cb_env_changed(self, *args):
        pass
    
    
## -------------------------------------------------------------------------------------------------
    def __cb_alg_changed(self, *args):
        pass
    

## -------------------------------------------------------------------------------------------------
    def __cb_sce_button(self, *args):
        self.button_sce.config(state="disabled")
        
        self.sce_pop                = Toplevel(self.frame)
        self.sce_pop.title("RL Scenario Pool")
        self.sce_pop.geometry("500x450")
        
        self.left_frame_sce         = Frame(self.sce_pop)
        self.left_frame_sce.pack(side='left', fill='both')
        
        self.sce_list               = Listbox(self.left_frame_sce, width=72, height=24)
        self.sce_list.grid(row=0, column=0, sticky='w')
        for item in self.C_SCE_LISTS:
            self.sce_list.insert('end',item)
        self.sce_list.pack()
        
        self.sce_scrollbar          = Scrollbar(self.sce_pop)
        self.sce_scrollbar.pack(side='right', fill='both')
        self.sce_list.config(yscrollcommand = self.sce_scrollbar.set)
        self.sce_scrollbar.config(command = self.sce_list.yview)
        
        self.apply_button_sce       = Button(self.left_frame_sce, text='Apply', font=self.fontstyle, fg = 'black', command=self.__cb_apply_scenario)
        self.apply_button_sce.pack()
        
        self.cancel_button_sce      = Button(self.left_frame_sce, text='Cancel', font=self.fontstyle, fg = 'black', command=self.__cb_cancel_scenario)
        self.cancel_button_sce.pack()
        
        self.sce_pop.protocol("WM_DELETE_WINDOW", self.__cb_sce_pop_closing)
    

## -------------------------------------------------------------------------------------------------
    def __cb_sce_pop_closing(self, *args):
        self.sce_pop.destroy()
        self.button_sce.config(state="normal")

## -------------------------------------------------------------------------------------------------
    def __cb_sce_button_info(self, *args):
        self.sce_i_pop       = Toplevel(self.frame)
        self.sce_i_pop.title("Information about Environment")
        self.sce_i_pop.geometry("500x450")
        self.sce_i_label      = Label(self.sce_i_pop, text='No information!', pady=3, width=20, height=1, anchor='w').grid(row=0, column=0, sticky='W')
        
        
## -------------------------------------------------------------------------------------------------
    def __cb_apply_scenario(self, *args):
        try:
            self.applied_sce                = self.sce_list.get('active')
        except:
            return
        
        for i in range(len(self.auto_scan_sce)):
            if self.auto_scan_sce[i].C_NAME == self.applied_sce:
                self.shared_db.rl_scenario          = self.auto_scan_sce[i](p_mode=Environment.C_MODE_SIM, p_ada=True, p_cycle_limit=10, p_visualize=True, p_logging=True)
                self.shared_db.rl_scenario_class    = self.auto_scan_sce[i]
                break
        self.shared_db.rl_env               = self.shared_db.rl_scenario.get_env().C_NAME
        self.shared_db.rl_learning_alg      = self.shared_db.rl_scenario.get_agent().C_NAME
        
        self.sce_var.set(self.applied_sce)
        self.sce_pop.destroy()
        self.button_sce.config(state="normal")
        self.button_sce_i.config(state="normal")
        
    
        
## -------------------------------------------------------------------------------------------------
    def __cb_cancel_scenario(self, *args):
        self.sce_pop.destroy()
        self.button_sce.config(state="normal")
    
    
    
    
        
        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLParam(SciUITabCTRL):
    """
    ...
    """

    C_NAME          = 'Parameters'
    
    C_FONT_FAMILY   = 'Lucida Grande'
    C_FONT_SIZE     = 10 

## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()
        self.first_refresh              = True
        self.fontstyle                  = tkFont.Font(family=self.C_FONT_FAMILY, size=self.C_FONT_SIZE)
        self.tab_list                   = []
        self.tab_ctrl                   = None 
        self.add_tab('Environment', None)
        self.add_tab('RL Agent(s)', None)
        self.add_tab('Data Storing', None)
        
        self.ep_var                     = IntVar()
        self.cycle_time_var             = IntVar()
        self.ep_var.set(10)
        self.cycle_time_var.set(100)
        
        self.param_var                  = []
        
        self.save_data                  = IntVar()
        self.collect_states             = IntVar()
        self.collect_actions            = IntVar()
        self.collect_rewards            = IntVar()
        self.collect_training           = IntVar()
        self.data_path                  = StringVar()
        self.save_data.set(self.shared_db.rl_save_data)
        self.collect_states.set(self.shared_db.rl_collect_states)
        self.collect_actions.set(self.shared_db.rl_collect_actions)
        self.collect_rewards.set(self.shared_db.rl_collect_rewards)
        self.collect_training.set(self.shared_db.rl_collect_training)
        self.data_path.set("%s"%os.getcwd())

## -------------------------------------------------------------------------------------------------
    def refresh(self, p_parent_frame):
        if not self.first_refresh: 
            self.__del__()
            self.init_component()
        self.first_refresh = False
        super().refresh(p_parent_frame=p_parent_frame)
        self.tab_ctrl.config(width=400, height=572)
        self.__cb_frame()


 ## -------------------------------------------------------------------------------------------------
    def __cb_frame(self, *args): 
            
        ### Scrollbar ###
        env_canvas          = Canvas(self.tab_list[0][1])
        env_canvas.pack(side='left', expand=1, fill ="both")
        env_scrollbar       = Scrollbar(self.tab_list[0][1], orient="vertical", command=env_canvas.yview)
        env_scrollbar.pack(side='right', fill='y')
        env_canvas.configure(yscrollcommand=env_scrollbar.set)
        env_canvas.bind('<Configure>', lambda e:env_canvas.configure(scrollregion=env_canvas.bbox("all")))
        self.env_frame      = Frame(env_canvas)
        env_canvas.create_window((0,0), window=self.env_frame, anchor="nw")
        
        alg_canvas          = Canvas(self.tab_list[1][1])
        alg_canvas.pack(side='left', expand=1, fill ="both")
        alg_scrollbar       = Scrollbar(self.tab_list[1][1], orient="vertical", command=alg_canvas.yview)
        alg_scrollbar.pack(side='right', fill='y')
        alg_canvas.configure(yscrollcommand=alg_scrollbar.set)
        alg_canvas.bind('<Configure>', lambda e:alg_canvas.configure(scrollregion=alg_canvas.bbox("all")))
        self.alg_frame      = Frame(alg_canvas)
        alg_canvas.create_window((0,0), window=self.alg_frame, anchor="nw")
        
        data_canvas         = Canvas(self.tab_list[2][1])
        data_canvas.pack(side='left', expand=1, fill ="both")
        data_scrollbar      = Scrollbar(self.tab_list[2][1], orient="vertical", command=data_canvas.yview)
        data_scrollbar.pack(side='right', fill='y')
        data_canvas.configure(yscrollcommand=data_scrollbar.set)
        data_canvas.bind('<Configure>', lambda e:data_canvas.configure(scrollregion=data_canvas.bbox("all")))
        self.data_frame     = Frame(data_canvas)
        data_canvas.create_window((0,0), window=self.data_frame, anchor="nw")
        
        if (self.shared_db.rlui_selected_scenario == 0):
            Label(self.env_frame, text ="Please follow the given procedures :").grid(row=0, column=0, padx = 5, pady = 0, sticky="w")
            Label(self.env_frame, text ="1. Select RL scenario").grid(row=1, column=0, padx = 5, pady = 0, sticky="w")
            Label(self.env_frame, text ="3. Fill all the parameters").grid(row=2, column=0, padx = 5, pady = 0, sticky="w")
            Label(self.env_frame, text ="4. Run environment").grid(row=3, column=0, padx = 5, pady = 0, sticky="w")
        else:
            self.__cb_environment_tab()
            self.__cb_agents_tab()
            self.__cb_data_storing_tab()
        
 ## -------------------------------------------------------------------------------------------------
    def __cb_environment_tab(self, *args):
        Label(self.env_frame, text='Max. Episodes', pady=3, font=self.fontstyle).grid(row=0, column=0, padx=5, pady=0, sticky='NW')
        self.ep_inputs = Entry(self.env_frame, textvariable=self.ep_var, width=5, justify='center', font=self.fontstyle, state=self.shared_db.rl_param_state_box)
        self.ep_inputs.grid(row=0, column=1, sticky='w')
        
        Label(self.env_frame, text='Cycle Time / Ep.', pady=3, font=self.fontstyle).grid(row=1, column=0, padx=5, pady=0, sticky='NW')
        self.cycle_time_inputs = Entry(self.env_frame, textvariable=self.cycle_time_var, width=10, justify='center', font=self.fontstyle, state=self.shared_db.rl_param_state_box)
        self.cycle_time_inputs.grid(row=1, column=1, sticky='w')
        
        Label(self.env_frame, text='  ', pady=3, font=self.fontstyle).grid(row=2, column=0, padx=5, pady=0, sticky='NW')
        Label(self.env_frame, text='  ', pady=3, font=self.fontstyle).grid(row=3, column=0, padx=5, pady=0, sticky='NW')
        
        self.submit_cycle_border = Frame(self.env_frame, highlightbackground = "black", highlightthickness = 1, bd=0)
        self.submit_cycle_border.grid(row=4, column=1, sticky='w')
        self.submit_cycle = Button(self.submit_cycle_border, text='Run - Next Cycle Time', font=self.fontstyle, fg = 'black', bg = 'honeydew2', command=self.__cb_simulation_run_cycle, state=self.shared_db.rl_run_state_btn)
        self.submit_cycle.pack()
        
        self.submit_ep_border = Frame(self.env_frame, highlightbackground = "black", highlightthickness = 1, bd=0)
        self.submit_ep_border.grid(row=5, column=1, sticky='w')
        self.submit_ep = Button(self.submit_ep_border, text='Run - Next Episode', font=self.fontstyle, fg = 'black', bg = 'LightCyan2', command=self.__cb_simulation_run_ep, state=self.shared_db.rl_run_state_btn)
        self.submit_ep.pack()
        
        self.submit_border = Frame(self.env_frame, highlightbackground = "black", highlightthickness = 1, bd=0)
        self.submit_border.grid(row=6, column=1, sticky='w')
        self.submit = Button(self.submit_border, text='Run', font=self.fontstyle, fg = 'black', bg = 'LightSteelBlue1', command=self.__cb_simulation_run, state=self.shared_db.rl_run_state_btn)
        self.submit.pack()
        
 ## -------------------------------------------------------------------------------------------------
    def __cb_agents_tab(self, *args):
        try:
            num_agents              = len(self.shared_db.rl_scenario.get_agent().get_agents())
            multiagent              = True
        except:
            num_agents = 1
            multiagent              = False
        
        n_row   = 0
        for n in range(num_agents):
            if multiagent:
                self.agent_         = self.shared_db.rl_scenario.get_agent().get_agents()[n][0]
            else:
                self.agent_         = self.shared_db.rl_scenario.get_agent()
            Label(self.alg_frame, text=self.agent_.get_name(), pady=3, font=self.fontstyle).grid(row=n_row, column=0, padx=5, pady=0, sticky='NW')
            n_row += 1
            for x in range(self.agent_._policy._hyperparam_space.get_num_dim()):
                param_ids           = self.agent_._policy._hyperparam_space.get_dim_ids()[x]
                param_name          = self.agent_._policy._hyperparam_space.get_dim(param_ids).get_name_short()
                param_baseset       = self.agent_._policy._hyperparam_space.get_dim(param_ids).get_base_set()
                param_val           = self.agent_._policy._hyperparam_tupel.get_value(param_ids)
                if param_baseset == 'N' or param_baseset == 'Z':
                    self.param_var.append(IntVar())
                else:
                    self.param_var.append(DoubleVar())
                Label(self.alg_frame, text=param_name, pady=3, font=self.fontstyle).grid(row=n_row, column=0, padx=5, pady=0, sticky='NW')
                self.param_var[-1].set(param_val)
                self.param_inputs   = Entry(self.alg_frame, textvariable=self.param_var[-1], width=15, justify='center', font=self.fontstyle, state=self.shared_db.rl_param_state_box)
                self.param_inputs.grid(row=n_row, column=1, sticky='w')
                n_row += 1
            Label(self.alg_frame, text=' ', pady=3, font=self.fontstyle).grid(row=n_row, column=0, padx=5, pady=0, sticky='NW')
            n_row += 1       
        
            
        
 ## -------------------------------------------------------------------------------------------------
    def __cb_data_storing_tab(self, *args):
        
        Checkbutton(self.data_frame, text = "Save data", justify='left', variable=self.save_data, state=self.shared_db.rl_param_state_box).grid(row=0, column=0, padx=5, pady=0, sticky='NW')
        Checkbutton(self.data_frame, text = "Collect states", justify='left', variable=self.collect_states, state=self.shared_db.rl_param_state_box).grid(row=1, column=0, padx=5, pady=0, sticky='NW')
        Checkbutton(self.data_frame, text = "Collect actions", justify='left', variable=self.collect_actions, state=self.shared_db.rl_param_state_box).grid(row=2, column=0, padx=5, pady=0, sticky='NW')
        Checkbutton(self.data_frame, text = "Collect rewards", justify='left', variable=self.collect_rewards, state=self.shared_db.rl_param_state_box).grid(row=3, column=0, padx=5, pady=0, sticky='NW')
        Checkbutton(self.data_frame, text = "Collect training", justify='left', variable=self.collect_training, state=self.shared_db.rl_param_state_box).grid(row=4, column=0, padx=5, pady=0, sticky='NW')
        
        Label(self.data_frame, text='Path', padx=5, pady=0, font=self.fontstyle).grid(row=5, column=0, sticky='NW')
        self.path = Entry(self.data_frame, width=35, textvariable=self.data_path, justify='left', font=self.fontstyle)
        self.path.grid(row=5, column=1, sticky='w')
        
            
 ## -------------------------------------------------------------------------------------------------
    def __first_training_run(self, *args):
        if self.shared_db.rl_sim_started == False and self.shared_db.rl_sim_stop == True:
            cycle_time                  = self.cycle_time_var.get()
            self.shared_db.rl_scenario  = self.shared_db.rl_scenario_class(p_mode=Environment.C_MODE_SIM, p_ada=True, p_cycle_limit=cycle_time, p_visualize=True, p_logging=True)
            
            try:
                num_agents              = len(self.shared_db.rl_scenario.get_agent().get_agents())
                multiagent              = True
            except:
                num_agents = 1
                multiagent              = False
                
            n_param = 0
            for n in range(num_agents):
                if multiagent:
                    self.agent_         = self.shared_db.rl_scenario.get_agent().get_agents()[n][0]
                else:
                    self.agent_         = self.shared_db.rl_scenario.get_agent()
                for x in range(self.agent_._policy._hyperparam_space.get_num_dim()):
                    param_ids           = self.agent_._policy._hyperparam_space.get_dim_ids()[x]
                    param_value         = self.param_var[n_param].get()
                    self.agent_._policy._hyperparam_tupel.set_value(param_ids,param_value)
                    n_param += 1
                    
            self.shared_db.rl_save_data         = self.save_data.get()
            self.shared_db.rl_collect_states    = self.collect_states.get()
            self.shared_db.rl_collect_actions   = self.collect_actions.get()
            self.shared_db.rl_collect_rewards   = self.collect_rewards.get()
            self.shared_db.rl_collect_training  = self.collect_training.get()
            ep_limit                            = self.ep_var.get()
            self.shared_db.rl_training  = Training(p_scenario=self.shared_db.rl_scenario,
                                                   p_episode_limit=ep_limit,
                                                   p_cycle_limit=cycle_time,
                                                   p_collect_states=self.shared_db.rl_collect_states,
                                                   p_collect_actions=self.shared_db.rl_collect_actions,
                                                   p_collect_rewards=self.shared_db.rl_collect_rewards,
                                                   p_collect_training=self.shared_db.rl_collect_training,
                                                   p_logging=self.shared_db.rl_log_training
                                                   )
            self.shared_db.rl_scenario._env.switch_logging(self.shared_db.rl_log_env)
            self.shared_db.rl_scenario._agent.switch_logging(self.shared_db.rl_log_agent)
            self.shared_db.rl_scenario.switch_logging(self.shared_db.rl_log_process)
         
        
 ## -------------------------------------------------------------------------------------------------
    def __env_finish_check(self, *args):
        if self.shared_db.rl_training._episode_id >= self.shared_db.rl_training._episode_limit-1:
            self.shared_db.rl_sim_finished = True
         
        
 ## -------------------------------------------------------------------------------------------------
    def __cb_simulation_run(self, *args):
        self.__first_training_run()
        self.shared_db.last_refresh         = 0
        if not self.shared_db.rl_sim_finished:
            self.shared_db.rl_sim_started   = True
            self.shared_db.rl_sim_stop      = False
            while self.shared_db.rl_training._episode_id < self.shared_db.rl_training._episode_limit:
                current_episode_id          = self.shared_db.rl_training._episode_id
                while self.shared_db.rl_training._episode_id == current_episode_id:
                    self.shared_db.rl_training.run_cycle()
                    if (self.shared_db.last_refresh % self.shared_db.refresh_rate) == 0:
                        self.shared_db.ep_monitor_frame.refresh(None)
                        self.shared_db.plot_frame.refresh(None)
                        self.shared_db.log_frame.refresh(None)
                    self.shared_db.last_refresh += 1
            self.shared_db.rl_sim_stop      = True
            self.__env_finish_check()
            self.shared_db.start_global_refresh()
        
 ## -------------------------------------------------------------------------------------------------
    def __cb_simulation_run_ep(self, *args):
        self.__first_training_run()
        self.shared_db.last_refresh         = 0
        if not self.shared_db.rl_sim_finished:
            self.shared_db.rl_sim_started   = True
            self.shared_db.rl_sim_stop      = False
            current_episode_id              = self.shared_db.rl_training._episode_id
            while self.shared_db.rl_training._episode_id == current_episode_id:
                self.shared_db.rl_training.run_cycle()
                if (self.shared_db.last_refresh % self.shared_db.refresh_rate) == 0:
                    self.shared_db.ep_monitor_frame.refresh(None)
                    self.shared_db.plot_frame.refresh(None)
                    self.shared_db.log_frame.refresh(None)
                self.shared_db.last_refresh += 1
            self.shared_db.rl_sim_stop      = True
            self.__env_finish_check()
            self.shared_db.start_global_refresh()
        
 ## -------------------------------------------------------------------------------------------------
    def __cb_simulation_run_cycle(self, *args):
        self.__first_training_run()
        if not self.shared_db.rl_sim_finished:
            self.shared_db.rl_sim_started   = True
            self.shared_db.rl_sim_stop      = False
            self.shared_db.rl_training.run_cycle() 
            self.shared_db.rl_sim_stop      = True
            self.__env_finish_check()
            self.shared_db.start_global_refresh()
    
    
  
    
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLPlot2D(SciUISubplot2D):
    """
    ...
    """

    C_NAME                      = 'RL Plot 2D'

## -------------------------------------------------------------------------------------------------
    def init_component(self):
        self.C_AX_RECTANGLE              = [0.10, 0.11, 0.85, 0.85]

        super().init_component()

        self.plot               = None
        self.min_val            = 1
        self.max_val            = 0
        self.val_range_changed  = False


## -------------------------------------------------------------------------------------------------
    def refresh_custom(self):
        super().refresh_custom()

        if self.plot == None:
            self.ax.grid()
            self.ax.plot([0],[0],color='blue', lw=1)
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 1)
        
    
    
  
    
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLHeatmap2D(SciUISubplot2D):
    """
    ...
    """

    C_NAME                      = 'RL Heatmap 2D'

## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()

        self.plot               = None
        self.min_val            = 1
        self.max_val            = 0
        self.val_range_changed  = False
        
        self.random_data        = np.random.random((16, 16))
        self.title_heatmap      = None


## -------------------------------------------------------------------------------------------------
    def refresh_custom(self):
        super().refresh_custom()

        if self.plot == None:
            im = self.ax.imshow(self.random_data, cmap='winter', interpolation='nearest')
        
        
    
        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLStatesToggle(SciUIFrameParam):
    """
    ...
    """

    C_NAME                  = 'States Toggle'
    C_AGENT_LABELS          = ['Agent 01','Agent 02','Agent 03','Agent 04','Agent 05']
    C_STATES_LABELS         = ['State 01','State 02']


## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()
        self.first_refresh  = True
        
        self.agent_var      = StringVar()
        self.states_var     = StringVar()


## -------------------------------------------------------------------------------------------------
    def refresh(self, p_parent_frame):
        super().refresh(p_parent_frame=p_parent_frame)
        
        self.agent_box      = Combobox(self.frame, textvariable=self.agent_var, width=15, justify='left', state='normal', background='white', font=self.fontstyle, values=self.C_AGENT_LABELS)
        self.agent_box.grid(row=0, column=0, padx=5, pady=5, sticky='NW')
        self.agent_var.trace('w', self.__cb_agent_changed)
        
        self.state_box      = Combobox(self.frame, textvariable=self.states_var, width=15, justify='left', state='normal', background='white', font=self.fontstyle, values=self.C_STATES_LABELS)
        self.state_box.grid(row=0, column=1, padx=5, pady=5, sticky='NW')
        self.states_var.trace('w', self.__cb_states_changed)
  
        
 ## -------------------------------------------------------------------------------------------------
    def __cb_agent_changed(self, *args):
        pass 
        
        
 ## -------------------------------------------------------------------------------------------------
    def __cb_states_changed(self, *args):
        pass 
        
        
        
    
        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLStatesView(SciUIFrameParam):
    """
    ...
    """

    C_NAME                          = 'States'
    C_AGENT_LABELS                  = ['Agent 01','Agent 02','Agent 03','Agent 04','Agent 05']
    C_STATES_LABELS                 = ['State 01','State 02']


## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()
        self.first_refresh              = True
        
        self.states_plt                 = RLPlot2D(self.shared_db, p_row=1, p_col=0, p_width_perc=0.36, p_height_perc=0.26, p_logging=self._level)
        self.states_plt.frame_visible   = False
        self.states_toggle              = RLStatesToggle(p_shared_db=self.shared_db, p_row=0, p_col=0, p_pady=5,
                                                     p_logging=self._level, p_height_perc=1)
        
        self.add_component(self.states_plt)
        self.add_component(self.states_toggle)


## -------------------------------------------------------------------------------------------------
    def refresh(self, p_parent_frame):
        super().refresh(p_parent_frame=p_parent_frame)
        
        
        
    
        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLActionsView(SciUIFrameParam):
    """
    ...
    """

    C_NAME                                  = 'Actions'
    C_AGENT_LABELS                          = ['Agent 01','Agent 02','Agent 03','Agent 04','Agent 05']


## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()
        self.first_refresh                  = True
        
        self.actions_plt                    = RLPlot2D(self.shared_db, p_row=1, p_col=0, p_width_perc=0.36, p_height_perc=0.26, p_logging=self._level)
        self.actions_plt.frame_visible      = False
        self.add_component(self.actions_plt)
        
        self.agent_var                      = StringVar()


## -------------------------------------------------------------------------------------------------
    def refresh(self, p_parent_frame):
        super().refresh(p_parent_frame=p_parent_frame)
        
        self.agent_box                      = Combobox(self.frame, textvariable=self.agent_var, width=15, justify='left', state='normal', background='white', font=self.fontstyle, values=self.C_AGENT_LABELS)
        self.agent_box.grid(row=0, column=0, padx=15, pady=10, sticky='NW')
        self.agent_var.trace('w', self.__cb_agent_changed)
  
        
 ## -------------------------------------------------------------------------------------------------
    def __cb_agent_changed(self, *args):
        pass 
        
        
        
    
        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLRewardsView(SciUIFrameParam):
    """
    ...
    """

    C_NAME                              = 'Rewards'
    C_AGENT_LABELS                      = ['Agent 01','Agent 02','Agent 03','Agent 04','Agent 05','Total Reward']


## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()
        self.first_refresh              = True
        
        self.rewards_plt                = RLPlot2D(self.shared_db, p_row=1, p_col=0, p_width_perc=0.36, p_height_perc=0.26, p_logging=self._level)
        self.rewards_plt.frame_visible  = False
        self.add_component(self.rewards_plt)
        
        self.agent_var                  = StringVar()


## -------------------------------------------------------------------------------------------------
    def refresh(self, p_parent_frame):
        super().refresh(p_parent_frame=p_parent_frame)
        
        self.agent_box                  = Combobox(self.frame, textvariable=self.agent_var, width=15, justify='left', state='normal', background='white', font=self.fontstyle, values=self.C_AGENT_LABELS)
        self.agent_box.grid(row=0, column=0, padx=15, pady=10, sticky='NW')
        self.agent_var.trace('w', self.__cb_agent_changed)
  
        
 ## -------------------------------------------------------------------------------------------------
    def __cb_agent_changed(self, *args):
        pass 
        
        
        
        
        
    
        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLGoalsCompletedView(SciUIFrameParam):
    """
    ...
    """

    C_NAME                              = 'Goals Completed'
    C_AGENT_LABELS                      = []


## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()
        self.C_AGENT_LABELS.append(self.shared_db.rl_env)
        self.first_refresh              = True
        
        self.exp_rate_plt               = RLPlot2D(self.shared_db, p_row=1, p_col=0, p_width_perc=0.36, p_height_perc=0.26, p_logging=self._level)
        self.exp_rate_plt.frame_visible = False
        self.add_component(self.exp_rate_plt)
        
        self.agent_var                  = StringVar()


## -------------------------------------------------------------------------------------------------
    def refresh(self, p_parent_frame):
        super().refresh(p_parent_frame=p_parent_frame)
        
        self.agent_box                  = Combobox(self.frame, textvariable=self.agent_var, width=15, justify='left', state='normal', background='white', font=self.fontstyle, values=self.C_AGENT_LABELS)
        self.agent_box.grid(row=0, column=0, padx=15, pady=10, sticky='NW')
        self.agent_var.trace('w', self.__cb_agent_changed)
  
        
 ## -------------------------------------------------------------------------------------------------
    def __cb_agent_changed(self, *args):
        pass 
        
        
        
        
    
        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLCustomView(SciUIFrameParam):
    """
    ...
    """

    C_NAME                              = 'Customs'
    C_AGENT_LABELS                      = []


## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()
        self.first_refresh              = True
        
        self.custom_plt                 = RLPlot2D(self.shared_db, p_row=1, p_col=0, p_width_perc=0.36, p_height_perc=0.26, p_logging=self._level)
        self.custom_plt.frame_visible   = False
        self.add_component(self.custom_plt)
        
        self.agent_var                  = StringVar()


## -------------------------------------------------------------------------------------------------
    def refresh(self, p_parent_frame):
        super().refresh(p_parent_frame=p_parent_frame)
        
        self.agent_box                  = Combobox(self.frame, textvariable=self.agent_var, width=15, justify='left', state='normal', background='white', font=self.fontstyle, values=self.C_AGENT_LABELS)
        self.agent_box.grid(row=0, column=0, padx=15, pady=10, sticky='NW')
        self.agent_var.trace('w', self.__cb_agent_changed)
  
        
 ## -------------------------------------------------------------------------------------------------
    def __cb_agent_changed(self, *args):
        pass 

        
        
        
        
    
        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLHeatmapsView(SciUIFrameParam):
    """
    ...
    """

    C_NAME                                      = 'Performance Maps - Action and Reward'
    C_AGENT_LABELS                              = ['Agent 01','Agent 02','Agent 03','Agent 04','Agent 05']


## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()
        self.first_refresh                      = True
        
        self.maps_actions_plt                   = RLHeatmap2D(self.shared_db, p_row=1, p_col=0, p_width_perc=0.18, p_height_perc=0.26, p_logging=self._level)
        self.maps_actions_plt.title_heatmap     = "Performance Maps - Actions"
        self.maps_actions_plt.frame_visible     = False
        
        self.maps_rewards_plt                   = RLHeatmap2D(self.shared_db, p_row=1, p_col=1, p_width_perc=0.18, p_height_perc=0.26, p_logging=self._level)
        self.maps_rewards_plt.title_heatmap     = "Performance Maps - Rewards"
        self.maps_rewards_plt.frame_visible     = False
        
        self.add_component(self.maps_actions_plt)
        self.add_component(self.maps_rewards_plt)
        
        self.agent_var                          = StringVar()


## -------------------------------------------------------------------------------------------------
    def refresh(self, p_parent_frame):
        super().refresh(p_parent_frame=p_parent_frame)
        
        self.agent_box                          = Combobox(self.frame, textvariable=self.agent_var, width=15, justify='left', state='normal', background='white', font=self.fontstyle, values=self.C_AGENT_LABELS)
        self.agent_box.grid(row=0, column=0, padx=15, pady=10, sticky='NW')
        self.agent_var.trace('w', self.__cb_agent_changed)
  
        
 ## -------------------------------------------------------------------------------------------------
    def __cb_agent_changed(self, *args):
        pass 
        




    
    
    
    
        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLGeneralView(SciUIFrame):
    """
    ...
    """

    C_NAME          = 'General View'


## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()
        self.first_refresh  = True

        self.add_component(RLStatesView(p_shared_db=self.shared_db, p_row=0, p_col=0, p_pady=8,
                                        p_padx=10, p_logging=self._level, p_visible=True))
        self.add_component(RLActionsView(p_shared_db=self.shared_db, p_row=0, p_col=1, p_pady=8,
                                         p_padx=10, p_logging=self._level, p_visible=True))
        self.add_component(RLRewardsView(p_shared_db=self.shared_db, p_row=1, p_col=0, p_pady=8,
                                         p_padx=10, p_logging=self._level, p_visible=True))
        self.add_component(RLGoalsCompletedView(p_shared_db=self.shared_db, p_row=1, p_col=1, p_pady=8,
                                                p_padx=10, p_logging=self._level, p_visible=True))


## -------------------------------------------------------------------------------------------------
    def refresh(self, p_parent_frame):
        super().refresh(p_parent_frame=p_parent_frame)
        

        




    
    
    
    
        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLEnvironmentView(SciUIFrame):
    """
    ...
    """

    C_NAME          = 'Environment View'


## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()
        self.first_refresh  = True

        self.add_component(RLCustomView(p_shared_db=self.shared_db, p_row=0, p_col=0, p_pady=8,
                                        p_padx=10, p_logging=self._level, p_visible=True))


## -------------------------------------------------------------------------------------------------
    def refresh(self, p_parent_frame):
        super().refresh(p_parent_frame=p_parent_frame)
        

        




    
    
    
    
        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLAgentView(SciUIFrame):
    """
    ...
    """

    C_NAME          = 'RL Agent(s) View'


## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()
        self.first_refresh  = True

        self.add_component(RLHeatmapsView(p_shared_db=self.shared_db, p_row=0, p_col=0, p_pady=8,
                                          p_padx=10, p_logging=self._level, p_visible=True))


## -------------------------------------------------------------------------------------------------
    def refresh(self, p_parent_frame):
        super().refresh(p_parent_frame=p_parent_frame)

        



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLView(SciUITabCTRL):
    """
    ...
    """

    C_NAME          = 'Real-Time Monitoring'
    
    C_FONT_FAMILY   = 'Lucida Grande'
    C_FONT_SIZE     = 10 


## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()
        self.first_refresh  = True
        self.fontstyle  = tkFont.Font(family=self.C_FONT_FAMILY, size=self.C_FONT_SIZE)
        self.tab_list   = []
        self.tab_ctrl   = None 
        self.add_tab('General view', RLGeneralView(p_shared_db=self.shared_db, p_row=0, p_col=0, p_pady=5, p_logging=self._level))
        self.add_tab('Environment view', RLEnvironmentView(p_shared_db=self.shared_db, p_row=0, p_col=0, p_pady=5, p_logging=self._level))
        self.add_tab('RL Agent(s) view', RLAgentView(p_shared_db=self.shared_db, p_row=0, p_col=0, p_pady=5, p_logging=self._level))

## -------------------------------------------------------------------------------------------------
    def refresh(self, p_parent_frame):
        super().refresh(p_parent_frame=p_parent_frame)
        self.tab_ctrl.config(width=1440, height=680)
        
        
        
        
        

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLEpMonitoring(SciUIFrameParam):
    """
    ...
    """

    C_NAME          = 'Current Status'

## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()
        self.first_refresh          = True
        self.ep_monitor_var         = IntVar()
        self.cycletime_monitor_var  = IntVar()
        self.run_monitor_var        = StringVar()
        self.ep_monitor_var.set(0)
        self.cycletime_monitor_var.set(0)
        self.run_monitor_var.set('Stop')


## -------------------------------------------------------------------------------------------------
    def refresh(self, p_parent_frame):
        if self.first_refresh: 
            self.first_refresh  = False
        else: 
            try:
                self.ep_monitor_var.set(self.shared_db.rl_training._episode_id)
                self.cycletime_monitor_var.set(self.shared_db.rl_training._cycle_id)
            except:
                self.ep_monitor_var.set(0)
                self.cycletime_monitor_var.set(0)
                
            if self.shared_db.rl_sim_finished:
                self.run_monitor_var.set('Done')
                self.shared_db.bg_color_run_button = "yellow"
            elif self.shared_db.rl_sim_stop:
                self.run_monitor_var.set('Stop')
                self.shared_db.bg_color_run_button = "red"
            else:
                self.run_monitor_var.set('Running')
                self.shared_db.bg_color_run_button = "green"
                

        super().refresh(p_parent_frame=p_parent_frame)
        
        self.env_label_1        = Label(self.frame, text='Episode', pady=5, font=self.fontstyle, width=15, height=5, anchor='s').grid(row=0, column=0, sticky='s')
        self.alg_label_1        = Label(self.frame, text='Cycle Time', pady=5, font=self.fontstyle, width=15, height=5, anchor='s').grid(row=0, column=1, sticky='s')
        self.run_label_1        = Label(self.frame, text='Simulation (on/off)', pady=5, font=self.fontstyle, width=20, height=5, anchor='s').grid(row=0, column=2, sticky='s')
    
        self.env_label_2        = Label(self.frame, textvariable=self.ep_monitor_var, pady=4, font=self.fontstyle, width=15, height=5, anchor='n').grid(row=1, column=0, sticky='n')
        self.alg_label_2        = Label(self.frame, textvariable=self.cycletime_monitor_var, pady=4, font=self.fontstyle, width=15, height=5, anchor='n').grid(row=1, column=1, sticky='n')
        self.run_label_2        = Label(self.frame, textvariable=self.run_monitor_var, pady=4, font=self.fontstyle, width=20, height=1, anchor='n', borderwidth=2, relief='raised', bg=self.shared_db.bg_color_run_button).grid(row=1, column=2, sticky='n')
 

        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLLoggingSelection(SciUIFrameParam):
    """
    ...
    """

    C_NAME          = 'Logging settings'

## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()
        self.first_refresh      = True
        self.set_env_log        = BooleanVar()
        self.set_agent_log      = BooleanVar()
        self.set_process_log    = BooleanVar()
        self.set_train_log      = BooleanVar()
        self.set_env_log.set(self.shared_db.rl_log_env)
        self.set_agent_log.set(self.shared_db.rl_log_agent)
        self.set_process_log.set(self.shared_db.rl_log_process)
        self.set_train_log.set(self.shared_db.rl_log_training)


## -------------------------------------------------------------------------------------------------
    def refresh(self, p_parent_frame):
        if self.first_refresh: 
            self.first_refresh  = False
        else: 
            if self.shared_db.rl_log_state_box:
                self.shared_db.rl_log_state_btn = 'normal'
            else:
                self.shared_db.rl_log_state_btn = 'disabled'

        super().refresh(p_parent_frame=p_parent_frame)
        
        self.set_env_button     = Checkbutton(self.frame, text = "Environment", justify='left', variable=self.set_env_log, height=2, width=12, anchor='w', state=self.shared_db.rl_log_state_btn).grid(row=0, column=0, padx=5, pady=0, sticky='nw')
        self.set_env_log.trace('w', self.__cb_set_env_changed)
        
        self.set_agent_button   = Checkbutton(self.frame, text = "Agent(s)", justify='left', variable=self.set_agent_log, height=2, width=12, anchor='w', state=self.shared_db.rl_log_state_btn).grid(row=1, column=0, padx=5, pady=0, sticky='nw')
        self.set_agent_log.trace('w', self.__cb_set_agent_changed)
        
        self.set_process_button = Checkbutton(self.frame, text = "Process", justify='left', variable=self.set_process_log, height=2, width=12, anchor='w', state=self.shared_db.rl_log_state_btn).grid(row=2, column=0, padx=5, pady=0, sticky='nw')
        self.set_process_log.trace('w', self.__cb_set_process_changed)
        
        self.set_train_button   = Checkbutton(self.frame, text = "Training", justify='left', variable=self.set_train_log, height=2, width=12, anchor='w', state=self.shared_db.rl_log_state_btn).grid(row=3, column=0, padx=5, pady=0, sticky='nw')
        self.set_train_log.trace('w', self.__cb_set_train_changed)
        
        Label(self.frame, text = " ", justify='left', height=1, width=12, anchor='w').grid(row=4, column=0, padx=5, pady=2.4, sticky='nw')
        
## -------------------------------------------------------------------------------------------------
    def __cb_set_env_changed(self, *args):
        try:
            val = self.set_env_log.get()
        except:
            return
        
        self.shared_db.rl_log_env = val
        self.set_env_log.set(self.shared_db.rl_log_env)
        self.shared_db.rl_scenario._env.switch_logging(self.set_env_log.get())
        
## -------------------------------------------------------------------------------------------------
    def __cb_set_agent_changed(self, *args):
        try:
            val = self.set_agent_log.get()
        except:
            return
        
        self.shared_db.rl_log_agent = val
        self.set_agent_log.set(self.shared_db.rl_log_agent)
        self.shared_db.rl_scenario._agent.switch_logging(self.set_agent_log.get())
        
## -------------------------------------------------------------------------------------------------
    def __cb_set_process_changed(self, *args):
        try:
            val = self.set_process_log.get()
        except:
            return
        
        self.shared_db.rl_log_process = val
        self.set_process_log.set(self.shared_db.rl_log_process)
        self.shared_db.rl_scenario.switch_logging(self.set_process_log.get())
        
## -------------------------------------------------------------------------------------------------
    def __cb_set_train_changed(self, *args):
        try:
            val = self.set_train_log.get()
        except:
            return
        
        self.shared_db.rl_log_training = val
        self.set_train_log.set(self.shared_db.rl_log_training)
        self.shared_db.rl_training.switch_logging(self.set_train_log.get())
        
        
        

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLLogging(SciUIFrameParam):
    """
    ...
    """

    C_NAME          = 'Simulation Log'
    C_NUM_ROWS      = 10

## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()
        self.first_refresh  = True


## -------------------------------------------------------------------------------------------------
    def refresh(self, p_parent_frame):
        if self.first_refresh: 
            self.first_refresh = False
        else: 
            return

        super().refresh(p_parent_frame=p_parent_frame)
        
        self.log_frame = ScrolledText(self.frame, height=9, width=177, wrap=WORD)
        
        ### 1. Header ###
        self.log_frame.config(pady=20, padx=20, font=self.fontstyle, state='normal')
        self.log_frame.insert(INSERT, "Logging taken from MLPro class!\n")
        # self.log_frame.insert(END, "Logging taken from MLPro class!\n")
        # self.log_frame.yview(END)
        self.log_frame.config(state='disabled')
        self.log_frame.pack()
        self.log_frame.insert(INSERT, "Logging taken from MLPro class!\n")
        self.log_frame.pack()
        
        
        
        
        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLFrameTopLeft(SciUIFrame):
    """
    ...
    """

    C_NAME                      = 'RL Frame Top-left'

## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()

        self.add_component(RLDetails(p_shared_db=self.shared_db, p_row=0, p_col=0, p_pady=5,
                                     p_logging=self._level, p_visible=True, p_height_perc=1))

        self.add_component(RLParam(p_shared_db=self.shared_db, p_row=1, p_col=0, p_pady=5,
                                     p_logging=self._level, p_visible=True, p_height_perc=1))
        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLFrameTopRight(SciUIFrame):
    """
    ...
    """

    C_NAME                      = 'RL Frame Top-Right'

## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()
        
        self.shared_db.plot_frame = RLView(p_shared_db=self.shared_db, p_row=0, p_col=0, p_pady=5,
                                           p_logging=self._level, p_visible=True, p_height_perc=1)
        self.add_component(self.shared_db.plot_frame)

        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLFrameTop(SciUIFrame):
    """
    ...
    """

    C_NAME                      = 'RL Frame Top'

## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()

        self.add_component(RLFrameTopLeft(p_shared_db=self.shared_db, p_row=0, p_col=0, p_pady=5,
                                          p_logging=self._level))

        self.add_component(RLFrameTopRight(p_shared_db=self.shared_db, p_row=0, p_col=1, p_pady=5,
                                           p_logging=self._level))
        
        
        
        
        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLFrameBottomLeft(SciUIFrame):
    """
    ...
    """

    C_NAME                      = 'RL Frame Bottom-left'

## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()
        
        self.shared_db.ep_monitor_frame = RLEpMonitoring(p_shared_db=self.shared_db, p_row=0, p_col=0, p_pady=5,p_padx=5,
                                                         p_logging=self._level, p_visible=True, p_height_perc=1)
        self.add_component(self.shared_db.ep_monitor_frame)
        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLFrameBottomRight(SciUIFrame):
    """
    ...
    """

    C_NAME                      = 'RL Frame Bottom-Right'

## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()
        
        self.add_component(RLLoggingSelection(p_shared_db=self.shared_db, p_row=0, p_col=0, p_pady=5,p_padx=5,
                                              p_logging=self._level, p_visible=True, p_height_perc=1))
        self.shared_db.log_frame = RLLogging(p_shared_db=self.shared_db, p_row=0, p_col=1, p_pady=5,p_padx=5,
                                             p_logging=self._level, p_visible=True, p_height_perc=1) 
        self.add_component(self.shared_db.log_frame)


        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLFrameBottom(SciUIFrame):
    """
    ...
    """

    C_NAME                      = 'RL Frame Bottom'

## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()

        self.add_component(RLFrameBottomLeft(p_shared_db=self.shared_db, p_row=0, p_col=0, p_pady=5,
                                             p_logging=self._level))
        self.add_component(RLFrameBottomRight(p_shared_db=self.shared_db, p_row=0, p_col=1, p_pady=5,
                                              p_logging=self._level))


        
        
        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class RLInteractiveUI(SciUIFrame): 
    """
    Main class for the RL interactive UI.
    """

    C_NAME                      = 'RL Interactive UI'
    C_ENV_LABELS                = ''
    C_ALG_LABELS                = ''
    
    
## -------------------------------------------------------------------------------------------------
    def __init__(self, p_shared_db:SciUISharedDB, p_row, p_col, p_title=None, p_width_perc=0.0,
                 p_height_perc=0.0, p_visible=False, p_padx=5, p_pady=0, p_sticky='NW', p_logging=True,
                 p_refresh_rate=100): 
        """
        Parameters:
            p_shared_db         Shared DB object for scenario-internal communication
            p_row               Row in superior frame
            p_col               Column in superior frame
            p_title             Frame title
            p_width_perc        Frame width in percent
            p_height_perc       Frame height in percent
            p_visible           Boolean switch for visibility
            p_padx              Horizontal distance to neighbor frames
            p_pady              Vertical distance to neighbor frames
            p_sticky            Tkinter sticky parameter 
            p_logging           Boolean switch for logging
            p_refresh_rate      Refresh rate per cycle time
        """

        self.row                        = p_row
        self.col                        = p_col
        self.width_perc                 = p_width_perc              # Frame width in percent of main window width
        self.height_perc                = p_height_perc             # Frame height in percent of main window height
        self.width_pix                  = p_shared_db.window_width * self.width_perc
        self.height_pix                 = p_shared_db.window_height * self.height_perc
        self.frame_visible              = p_visible

        if p_title == None:
            self.frame_text             = self.C_NAME
        else:
            self.frame_text             = p_title

        self.frame                      = None
        self.components                 = []
        self.sticky                     = p_sticky
        self.padx                       = p_padx
        self.pady                       = p_pady
        self.popup_menu                 = None
        self.refresh_rate               = p_refresh_rate
        
        super().__init__(p_shared_db, p_row, p_col, p_logging=p_logging) 
        
        
## -------------------------------------------------------------------------------------------------
    @staticmethod
    def enrich_shared_db(p_shared_db:SciUISharedDB):
        p_shared_db.rlui_selected_scenario  = False
        p_shared_db.refresh_rate            = 0
        p_shared_db.last_refresh            = 0
        
        p_shared_db.rl_env                  = ""
        p_shared_db.rl_learning_alg         = ""
        p_shared_db.rl_scenario_class       = None
        p_shared_db.rl_scenario             = None
        p_shared_db.rl_training             = None
        
        p_shared_db.rl_ep_limit             = 0
        p_shared_db.rl_cycle_limit          = 0
        p_shared_db.rl_collect_states       = True
        p_shared_db.rl_collect_rewards      = True
        p_shared_db.rl_collect_actions      = True
        p_shared_db.rl_collect_training     = True
        p_shared_db.rl_save_data            = True
        
        p_shared_db.rl_log_env              = False
        p_shared_db.rl_log_agent            = False
        p_shared_db.rl_log_process          = False
        p_shared_db.rl_log_training         = True
        
        p_shared_db.rl_sim_started          = False
        p_shared_db.rl_sim_stop             = True
        p_shared_db.rl_sim_finished         = False
        p_shared_db.rl_refresh_rate         = 10
        
        p_shared_db.rl_sce_state_btn        = True
        p_shared_db.rl_param_state_box      = 'normal'
        p_shared_db.rl_run_state_btn        = 'normal'
        p_shared_db.rl_log_state_box        = True
        p_shared_db.rl_log_state_btn        = 'normal'
        p_shared_db.bg_color_run_button     = 'red'
        
        p_shared_db.log_frame               = None
        p_shared_db.plot_frame              = None
        p_shared_db.ep_monitor_frame        = None


## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()
        self.add_component(RLFrameTop(p_shared_db=self.shared_db, p_row=0, p_col=0, p_pady=5,
                                      p_logging=self._level))
        self.add_component(RLFrameBottom(p_shared_db=self.shared_db, p_row=1, p_col=0, p_pady=5,
                                         p_logging=self._level))

 ## -------------------------------------------------------------------------------------------------
    def refresh(self, p_parent_frame):
        self.shared_db.refresh_rate                = self.refresh_rate
        if self.shared_db.rl_sim_started == False and self.shared_db.rl_sim_stop == True:
            self.shared_db.rl_sce_state_btn        = True
            self.shared_db.rl_param_state_box      = 'normal'
            self.shared_db.rl_run_state_btn        = 'normal'
            self.shared_db.rl_log_state_box        = True
        elif self.shared_db.rl_sim_started == True and self.shared_db.rl_sim_stop == False:
            self.shared_db.rl_sce_state_btn        = False
            self.shared_db.rl_param_state_box      = 'disabled'
            self.shared_db.rl_run_state_btn        = 'disabled'
            self.shared_db.rl_log_state_box        = False
        elif self.shared_db.rl_sim_started == True and self.shared_db.rl_sim_stop == True:
            self.shared_db.rl_sce_state_btn        = False
            self.shared_db.rl_param_state_box      = 'disabled'
            self.shared_db.rl_run_state_btn        = 'normal'
            self.shared_db.rl_log_state_box        = True
            if self.shared_db.rl_sim_finished:
                self.shared_db.rl_run_state_btn    = 'disabled'
                self.shared_db.rl_log_state_box    = False

        super().refresh(p_parent_frame=p_parent_frame)
    
    
    
    
    