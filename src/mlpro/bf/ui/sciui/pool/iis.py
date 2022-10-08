## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.ui.scui.pool
## -- Module  : iis
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-06-20  0.0.0     DA       Creation
## -- 2021-07-04  1.0.0     DA       Released first version
## -- 2021-07-07  1.0.1     DA       Minor corrections related to window size and update rate
## -- 2021-09-11  1.0.1     MRD      Change Header information to match our new library name
## -- 2022-03-21  1.0.2     SY       Refactoring following class Dimensions update 
## -- 2022-10-08  1.0.3     DA       Refactoring following class Dimensions update 
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.3 (2022-10-08)

This module provides the reusable SciUI component 'Interactive Input Space'. It represents a 2D or 3D 
input space based on a mathematical space and allows manual (2D) and automated (2D,3D) input creation.
It consists of time depending subplots for each input dimension and a 2D/3D main subplot for the input
space. To be reused and enriched in own SciUI scenarios.
"""


from math import sin, cos, pi
from tkinter import *
import tkinter
from tkinter.ttk import Combobox
from mpl_toolkits.mplot3d import Axes3D

from mlpro.bf.ui.sciui.framework import *
from mlpro.bf.ui.sciui.pool.iisbenchmark import *
from mlpro.bf.math import ESpace





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class IISIVar(SciUISubplot2D):
    """
    Subplot for input variable.
    """

    C_NAME                      = 'IIS Input Variable'

## -------------------------------------------------------------------------------------------------
    def init_component(self):
        if self.shared_db.iis_ispace.get_num_dim() == 2:
            self.C_AX_RECTANGLE              = [0.10, 0.11, 0.85, 0.85]
        else:
            self.C_AX_RECTANGLE              = [0.10, 0.15, 0.85, 0.80]

        super().init_component()

        self.plot               = None
        self.min_val            = 1
        self.max_val            = 0
        self.val_range_changed  = False


## -------------------------------------------------------------------------------------------------
    def refresh_custom(self):

        # 0 Initialization
        super().refresh_custom()
 
        if self.plot == None:
            # 1 Initial plot setup
            self.ax.grid()
            self.plot,           = self.ax.plot(self.shared_db.iis_input_ids, self.shared_db.iis_input_val[self.dim_id], color='blue', lw=1)

        elif self.shared_db.iis_new_input is not None:
            # 2 Update plot
            val = self.shared_db.iis_new_input[self.dim_id]

            if self.min_val > self.max_val:
                self.min_val        = val - 0.001
                self.max_val        = val + 0.001
                self.val_range_changed   = True

            elif val > self.max_val: 
                self.max_val        = val
                self.val_range_changed   = True

            elif val < self.min_val: 
                self.min_val = val
                self.val_range_changed   = True

            if self.shared_db.redraw:
                self.plot.set_xdata(self.shared_db.iis_input_ids)
                self.plot.set_ydata(self.shared_db.iis_input_val[self.dim_id])

        if not self.shared_db.redraw: return

        if self.shared_db.refresh_full:
            self.plot.set_xdata(self.shared_db.iis_input_ids)
            self.plot.set_ydata(self.shared_db.iis_input_val[self.dim_id])

        # 3 Update ax limits
        self.ax.set_xlim(self.shared_db.iis_min_id, self.shared_db.iis_max_id)
        if self.val_range_changed:
            self.ax.set_ylim(self.min_val, self.max_val)
            self.val_range_changed = False

        self.set_flush_events(self.shared_db.iis_flush_events)



## -------------------------------------------------------------------------------------------------
    def assign_dim(self, p_dim_id):
        self.dim_id = p_dim_id





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class IISMain2D(SciUISubplot2D):
    """
    ...
    """
    
    C_NAME                      = 'Interactive 2D Input Space'
    C_AX_RECTANGLE              = [0.09, 0.09 ,0.88 ,0.88]
    C_AX_FRAME                  = True
    C_AX_CURSOR                 = True
            
## -------------------------------------------------------------------------------------------------
    def cb_mouse_moved(self, event):
        super().cb_mouse_moved(event)
        self.shared_db.iis_update_counter = 0
        self.shared_db.iis_flush_events   = False
        self.shared_db.iis_sim_input_cb([event.xdata, event.ydata])
        self.shared_db.iis_flush_events   = True

        
## -------------------------------------------------------------------------------------------------
    def cb_mbutton_released(self, event):
        super().cb_mbutton_released(event)
        self.shared_db.iis_update_counter = 0
        self.shared_db.iis_flush_events   = False
        self.shared_db.iis_sim_input_cb([event.xdata, event.ydata])
        self.shared_db.iis_flush_events   = True


## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()
        self.axes_up_to_date    = False
        self.plot               = None


## -------------------------------------------------------------------------------------------------
    def refresh_custom(self):
        super().refresh_custom()

        if self.shared_db.refresh_full or not self.axes_up_to_date:
            self.refresh_axes()
            self.axes_up_to_date = True

        if not self.shared_db.refresh_full: return

        if self.plot is None:
            self.plot,  = self.ax.plot(self.shared_db.iis_input_val[0], self.shared_db.iis_input_val[1], 'b+')

        elif self.shared_db.redraw:
            self.plot.set_xdata(self.shared_db.iis_input_val[0])
            self.plot.set_ydata(self.shared_db.iis_input_val[1])

        self.set_flush_events(self.shared_db.iis_flush_events)
        

## -------------------------------------------------------------------------------------------------
    def refresh_axes(self):
        if not self.shared_db.redraw: return
        
        dims    = self.shared_db.iis_ispace.get_dims()
        dim0    = dims[0] 
        dim1    = dims[1]
        
        self.ax.set_xlabel('Input Variable ' + dim0.get_name_short())
        self.ax.set_ylabel('Input Variable ' + dim1.get_name_short())
        self.ax.set_xbound(dim0.get_boundaries())
        self.ax.set_ybound(dim1.get_boundaries())
        self.ax.set_xlim(dim0.get_boundaries())
        self.ax.set_ylim(dim1.get_boundaries())
        self.ax.grid(True)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class IISMain3D(SciUISubplot3D):
    """
    ...
    """
    
    C_NAME                      = 'Interactive 3D Input Space'

## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()
        self.axes_up_to_date    = False
        self.plot               = None


## -------------------------------------------------------------------------------------------------
    def refresh_custom(self):
        super().refresh_custom()

        if not self.axes_up_to_date:
            self.refresh_axes()
            self.axes_up_to_date = True

        if self.shared_db.iis_new_input is None: return

        if self.plot is None:
            # Create an object of type mpl_toolkits.mplot3d.art3d.Lines3D...
            self.plot, = self.ax.plot3D(self.shared_db.iis_input_val[0], self.shared_db.iis_input_val[1], self.shared_db.iis_input_val[2], color='blue', linestyle='None', marker='+')

        else:
            self.plot.set_data_3d(self.shared_db.iis_input_val[0], self.shared_db.iis_input_val[1], self.shared_db.iis_input_val[2])
 
        self.set_flush_events(self.shared_db.iis_flush_events)


## -------------------------------------------------------------------------------------------------
    def refresh_axes(self):
        if not self.shared_db.refresh_full: return 

        dims    = self.shared_db.iis_ispace.get_dims()

        dim0    = dims[0] 
        dim1    = dims[1] 
        dim2    = dims[2] 
        
        self.ax.set_xlabel('Input Variable ' + dim0.get_name_short())
        self.ax.set_ylabel('Input Variable ' + dim1.get_name_short())
        self.ax.set_zlabel('Input Variable ' + dim2.get_name_short())
        self.ax.set_xbound(dim0.get_boundaries())
        self.ax.set_ybound(dim1.get_boundaries())
        self.ax.set_zbound(dim2.get_boundaries())
        self.ax.set_xlim(dim0.get_boundaries())
        self.ax.set_ylim(dim1.get_boundaries())
        self.ax.set_zlim(dim2.get_boundaries())





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class IISParamGeneral(SciUIFrameParam):
    """
    ...
    """

    C_NAME                      = 'General'
    C_WSIZE_VALUES              = [0, 10, 50, 100, 200, 500, 1000, 2000]
    C_URATE_VALUES              = [1, 2, 5, 10, 50, 100]


## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()

        self.first_refresh  = True
        self.wsize_var      = IntVar()
        self.wsize_var.set(self.shared_db.iis_input_horizon)
        self.urate_var      = IntVar()
        self.urate_var.set(self.shared_db.iis_update_step)


## -------------------------------------------------------------------------------------------------
    def refresh(self, p_parent_frame):
        if self.first_refresh: 
            self.first_refresh = False
        else: 
            return

        super().refresh(p_parent_frame=p_parent_frame)

        # 1 Window size
        wsize_label = Label(self.frame, text='Window size           ', font=self.fontstyle)     
        wsize_label.grid(row=0, column=0, sticky='w')
        self.wsize  = Combobox(self.frame, textvariable=self.wsize_var, width=6, justify='right', state='normal', background='white', font=self.fontstyle, values=self.C_WSIZE_VALUES)
        self.wsize.grid(row=0, column=1)
        self.wsize_var.trace('w', self.__cb_wsize_changed)
        Label(self.frame, text='                                                                                                  ').grid(row=0, column=2)

        # 2 Update rate (with tool tip)
        urate_label = Label(self.frame, text='Update rate', font=self.fontstyle)
        urate_label.grid(row=1, column=0, sticky='w')
        self.urate  = Combobox(self.frame, textvariable=self.urate_var, width=6, justify='right', state='normal', background='white', font=self.fontstyle, values=[1,2,5,10,20,50,100,200,500,1000])
        self.urate.grid(row=1, column=1)
        self.urate_var.trace('w', self.__cb_urate_changed)
        SciUITooltip(self.urate, 'Number of inputs after which the screen is updated')


## -------------------------------------------------------------------------------------------------
    def __cb_wsize_changed(self, *args):
        try:
            wsize   = self.wsize_var.get()
        except:
            return

        if wsize == 1: 
            wsize = 2
            self.wsize_var.set(wsize)

        self.shared_db.iis_input_horizon    = wsize
        self.shared_db.iis_update_counter   = 1
        self.log(self.C_LOG_TYPE_I, 'Data window size changed to', str(self.shared_db.iis_input_horizon))

        self.shared_db.refresh_full         = True
        self.shared_db.start_global_refresh()



 ## -------------------------------------------------------------------------------------------------
    def __cb_urate_changed(self, *args):
        try:
            ustep = self.urate_var.get()
        except:
            return

        if ustep < 1:
            ustep = 1
            self.urate_var.set(ustep)

        self.shared_db.iis_update_step      = ustep
        self.shared_db.iis_update_counter   = 1

        self.log(self.C_LOG_TYPE_I, 'Screen update rate changed to', str(self.shared_db.iis_update_step))






## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class IISParamBenchmark(SciUIFrameParam):
    """
    ...
    """

    C_NAME                      = 'Benchmark Test'

## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()
 
        self.first_refresh      = True
        self.benchmark_var      = StringVar()
        self.benchmarks_cbox    = []
        self.benchmarks         = []
        self.num_inputs_var     = IntVar()
        self.width_cbox         = 55    # 80

        if self.shared_db.iis_ispace.get_num_dim() == 2:
            benchmark_classes = IISBenchmark2D.__subclasses__()

        elif self.shared_db.iis_ispace.get_num_dim() == 3:
            benchmark_classes = IISBenchmark3D.__subclasses__()

        for bm_class in benchmark_classes:
            bm_key = bm_class.C_GROUP + ' - ' + bm_class.C_NAME
            self.benchmarks.append([bm_key, bm_class])
            self.benchmarks_cbox.append(bm_key)

        self.benchmarks_cbox.sort()


## -------------------------------------------------------------------------------------------------
    def refresh(self, p_parent_frame):
        if self.first_refresh: 
            self.first_refresh = False
        else: 
            return

        super().refresh(p_parent_frame=p_parent_frame)

        # 1 Combo box for benchmark selection
        Label(self.frame, text='Benchmark Test     ', pady=3, font=self.fontstyle).grid(row=0, column=0, sticky='W')
        self.benchmark_cbox = Combobox(self.frame, textvariable=self.benchmark_var, values=self.benchmarks_cbox, state='readonly')
        self.benchmark_cbox.config(width=self.width_cbox, font=self.fontstyle)
        self.benchmark_cbox.grid(row=0, column=1, sticky='w')
        self.benchmark_var.set(self.benchmarks_cbox[0])
        self.benchmark_var.trace('w', self.__cb_benchmark_changed)

        # 2 Text box for benchmark description
        Label(self.frame, text='Description', pady=3, font=self.fontstyle).grid(row=1, column=0, sticky='NW')
        self.description    = Text(self.frame, height=5, width=self.width_cbox-5)
        self.description.grid(row=1, column=1, sticky='w')

        # 3 Info field for proposed window size/horizon

         
        # 4 Info field for number of inputs to be processed 
        Label(self.frame, text='Number of inputs', pady=3, font=self.fontstyle).grid(row=2, column=0, sticky='NW')
        self.num_inputs  = Entry(self.frame, textvariable=self.num_inputs_var, width=8, justify='right', font=self.fontstyle)
        self.num_inputs.grid(row=2, column=1, sticky='w')


        Label(self.frame, text=' ', pady=3, width=self.width_cbox+2, font=self.fontstyle).grid(row=3, column=1, sticky='W')

        # 5 Button for submitting the benchmark test
        self.submit = Button(self.frame, text='Run', command=self.__cb_benchmark_run)
        self.submit.grid(row=4, column=1, sticky='E')

        # 6 Update detail fields
        self.__benchmark_show_details(self.benchmarks_cbox[0])


## -------------------------------------------------------------------------------------------------
    def __benchmark_show_details(self, p_bm):
        for bm in self.benchmarks:
            if bm[0] == p_bm: break

        self.description.delete("1.0","end")
        self.description.insert(tkinter.END, bm[1].C_DESCRIPTION)
        self.num_inputs_var.set(bm[1].C_INPUTS)


## -------------------------------------------------------------------------------------------------
    def __cb_benchmark_changed(self, *args):
        benchmark   = self.benchmark_var.get()
        self.log(self.C_LOG_TYPE_I, 'Benchmark changed to "' + benchmark + '"')
        self.__benchmark_show_details(benchmark)


## -------------------------------------------------------------------------------------------------
    def __cb_benchmark_run(self, *args):
        benchmark   = self.benchmark_var.get()
 
        for bm in self.benchmarks:
            if bm[0] == benchmark: break

        self.log(self.C_LOG_TYPE_I, 'Started benchmark "' + benchmark + '"')
        bm[1](self.shared_db.iis_ispace, self.shared_db.iis_sim_input_cb, p_logging=self._level).run()
        self.shared_db.iis_update_counter = 0
        self.shared_db.start_global_refresh()

       



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class IISParam(SciUIFrameParam):
    """
    Main parameter frame of IIS.
    """

    C_NAME                      = 'Settings'

## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()

        self.add_component(IISParamGeneral(p_shared_db=self.shared_db, p_row=0, p_col=0, p_pady = 5, p_visible=True, p_logging=self._level))
        self.add_component(IISParamBenchmark(p_shared_db=self.shared_db, p_row=1, p_col=0, p_pady = 5, p_visible=True, p_logging=self._level))





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class IISFrameLeft(SciUIFrame):
    """
    ...
    """

    C_NAME                      = 'IIS Frame left'
    C_HEIGHT_VAR_PLOTS          = 0.5

## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()

        # 2 Create subplots for each input space dimension
        num_dim     = self.shared_db.iis_ispace.get_num_dim() 
        height_perc = self.C_HEIGHT_VAR_PLOTS / num_dim

        for i, dim_id in enumerate(self.shared_db.iis_ispace.get_dim_ids()):
            dim = self.shared_db.iis_ispace.get_dim(dim_id)
            iisivar_comp = IISIVar(self.shared_db, p_row=i, p_col=0, p_title='Input Variable ' + dim.get_name_short(), p_width_perc=0.25, p_height_perc=height_perc, p_logging=self._level)
            iisivar_comp.assign_dim(i)
            self.add_component(iisivar_comp)

        self.add_component(IISParam(self.shared_db, p_row=num_dim, p_col=0, p_visible=True, p_logging=self._level))





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class InteractiveInputSpace(SciUIFrame): 
    """
    Main class for the interactive 2D/3D input space.
    """

    C_NAME                      = 'Interactive Input Space'

## -------------------------------------------------------------------------------------------------
    @staticmethod
    def enrich_shared_db(p_shared_db:SciUISharedDB):
        p_shared_db.iis_ispace          = ESpace()
        p_shared_db.iis_new_input       = None
        p_shared_db.iis_input_id        = -1
        p_shared_db.iis_input_num       = 0
        p_shared_db.iis_input_ids       = []
        p_shared_db.iis_input_val       = [[],[],[]]
        p_shared_db.iis_input_horizon   = 100
        p_shared_db.iis_min_id          = 0
        p_shared_db.iis_max_id          = p_shared_db.iis_input_horizon - 1
        p_shared_db.iis_update_step     = 1
        p_shared_db.iis_update_counter  = 1
        p_shared_db.iis_flush_events    = True


## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()
        self.shared_db.iis_sim_input_cb = self.simulate_input


        # 1 Detect input space dimensionality
        try:
            self.dim = self.shared_db.iis_ispace.get_num_dim()
            self.log(self.C_LOG_TYPE_I, 'Input space with', str(self.dim), 'dimensions detected')
            if ( self.dim != 2 ) and ( self.dim != 3 ):
                self.log(self.C_LOG_TYPE_E, 'class is suitable for 2D and 3D input spaces only!') 
                return
        except:
            self.log(self.C_LOG_TYPE_E, 'Missing input space object (self.shared_db.iis_ispace)!')
            return


        # 2 Create left subframe with subplots for each input dimension
        self.add_component(IISFrameLeft(self.shared_db, p_row=0, p_col=0, p_padx=5, p_logging=self._level))


        #  3 Create input space subplot
        if self.dim == 2:
            self.add_component(IISMain2D(self.shared_db, p_row=0, p_col=1, p_height_perc=1.0, p_padx=5, p_logging=self._level))
        elif self.dim == 3:
            self.add_component(IISMain3D(self.shared_db, p_row=0, p_col=1, p_height_perc=1.0, p_padx=5, p_logging=self._level))


 ## -------------------------------------------------------------------------------------------------
    def refresh(self, p_parent_frame):
        if self.shared_db.refresh_full: self.update_data()

        if self.shared_db.iis_update_counter <= 1:
            self.shared_db.iis_update_counter = self.shared_db.iis_update_step
            self.shared_db.redraw = True
        else:
            self.shared_db.iis_update_counter -= 1
            self.shared_db.redraw = False

        super().refresh(p_parent_frame=p_parent_frame)


 ## -------------------------------------------------------------------------------------------------
    def update_data(self):
        self.shared_db.iis_max_id = max(1, self.shared_db.iis_input_id, self.shared_db.iis_input_horizon)      

        if self.shared_db.iis_input_horizon > 0: 
            if self.shared_db.iis_max_id > self.shared_db.iis_input_horizon:
                self.shared_db.iis_min_id = self.shared_db.iis_max_id - self.shared_db.iis_input_horizon + 1
            else:
                self.shared_db.iis_min_id = 0

            num_pop = max(len(self.shared_db.iis_input_ids) - self.shared_db.iis_input_horizon, 0)

            for i in range(num_pop): 
                self.shared_db.iis_input_ids.pop(0)

                for d in range(self.shared_db.iis_ispace.get_num_dim()):
                    self.shared_db.iis_input_val[d].pop(0)


## -------------------------------------------------------------------------------------------------
    def simulate_input(self, p_x): 
        self.log(self.C_LOG_TYPE_I, 'Simulate input', p_x)
        
        # 0 Initialization
        self.shared_db.iis_new_input    = p_x
        horizon                         = self.shared_db.iis_input_horizon
        ids                             = self.shared_db.iis_input_ids
        values                          = self.shared_db.iis_input_val

        # 1 Update input ids
        self.shared_db.iis_input_id     += 1
        ids.append(self.shared_db.iis_input_id)
        self.shared_db.iis_max_id       = max(1, self.shared_db.iis_input_id, horizon)      

        # 2 Append new input components to value lists
        for d, val in enumerate(p_x): values[d].append(val)

        # 3 Start global refresh
        self.shared_db.refresh_full = True
        self.shared_db.start_global_refresh()   

        # 4 Reset shared input variable
        self.shared_db.iis_new_input    = None         
