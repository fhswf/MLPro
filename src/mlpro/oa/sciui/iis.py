## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.oa.sciui
## -- Module  : iis.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-06-20  0.0.0     DA       Creation
## -- 2021-07-04  1.0.0     DA       Released first version
## -- 2021-07-07  1.0.1     DA       Minor corrections related to window size and update rate
## -- 2022-01-06  1.1.0     DA       Integration in mlpro
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.0 (2022-01-06)

This module provides the reusable SciUI component 'Interactive Input Space'. It represents a 2D or 3D 
input space based on a mathematical space and allows manual (2D) and automated (2D,3D) input creation.
It consists of time depending subplots for each input dimension and a 2D/3D main subplot for the input
space. To be reused and enriched in own SciUI scenarios.
"""


from math import sin, cos, pi
from tkinter import *
import tkinter
from tkinter import ttk
from tkinter.ttk import Combobox

from mpl_toolkits.mplot3d import Axes3D

import openml

from mlpro.bf.ui.sciui.framework import *
from mlpro.oa.models import *
from mlpro.oa.sciui.iis import *
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

        dim0    = self.shared_db.iis_ispace.get_dim(0) 
        dim1    = self.shared_db.iis_ispace.get_dim(1) 
        
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

        dim0    = self.shared_db.iis_ispace.get_dim(0) 
        dim1    = self.shared_db.iis_ispace.get_dim(1) 
        dim2    = self.shared_db.iis_ispace.get_dim(2) 
        
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




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class IISParamStream(SciUIFrameParam):
    """
    ...
    """

    C_NAME                  = 'Data Stream'
    C_STREAM_SRC_NATIVE     = 'Native'
    C_STREAM_SRC_OPENML     = 'OpenML'
    C_STREAM_SRC            = [ C_STREAM_SRC_NATIVE, C_STREAM_SRC_OPENML] 

    C_WIDTH_STREAM_CBOX     = 60


## -------------------------------------------------------------------------------------------------
    class IISDummyStream(Stream):
        """
        Internal dummy class for first line of stream list.
        """

        C_NAME          = 'Please select a stream'
        C_DESCRIPTION   = ''
        C_CITATION      = ''
        C_DOI           = ''
        C_FEATURES      = 0
        C_INSTANCES     = 0 
        C_URL           = ''


## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()
 
        self.first_refresh              = True

        self.stream_src_var             = StringVar()

        self.stream_var                 = StringVar()
        self.stream_cbox                = None

        self.num_features_var           = IntVar()
        self.feature_list               = []
        self.feature1_var               = StringVar()
        self.feature1_cbox              = None
        self.feature2_var               = StringVar()
        self.feature2_cbox              = None
        self.feature3_var               = StringVar()
        self.feature3_cbox              = None

        self.stream_list_native         = []
        self.stream_list_native_cbox    = []

        self.stream_list_openml         = []
        self.stream_list_openml_cbox    = []


## -------------------------------------------------------------------------------------------------
    def __fill_stream_list_native(self):
        stream_classes = Stream.__subclasses__()

        for stream_class in stream_classes:
            stream_key = stream_class.C_NAME

            if stream_key == self.IISDummyStream.C_NAME:
                self.stream_list_native.insert(0, [stream_key, stream_class])
            else:
                self.stream_list_native.append([stream_key, stream_class])
                self.stream_list_native_cbox.append(stream_key)

        self.stream_list_native_cbox.sort()
        self.stream_list_native_cbox.insert(0, self.IISDummyStream.C_NAME)


## -------------------------------------------------------------------------------------------------
    def __fill_stream_list_openml(self):
        openml_dict = openml.datasets.list_datasets(output_format='dict')
        # digits      = len(str(len(openml_dict)))

        for ds_id in openml_dict:
            ds          = openml_dict[ds_id]
            stream_key  = str(ds['did']) + ' - ' + ds['name'] + ' (Ver. ' + str(ds['version']) + ')'

            self.stream_list_openml.append([stream_key, ds])
            self.stream_list_openml_cbox.append(stream_key)

        self.stream_list_openml.insert(0, [0, self.IISDummyStream ])
        self.stream_list_openml_cbox.insert(0, self.IISDummyStream.C_NAME)


## -------------------------------------------------------------------------------------------------
    def refresh(self, p_parent_frame):

        # 0 Intro
        if self.first_refresh: 
            self.first_refresh = False
        else: 
            return

        super().refresh(p_parent_frame=p_parent_frame)


        # 1 Frame for stream selection
        self.fr_stream_src  = Frame(self.frame)
        self.fr_stream_src.grid(row=0, column=0)

        # 1.1 Combo box for stream source selection
        Label(self.fr_stream_src, text='Source ', pady=3, font=self.fontstyle).grid(row=0, column=0, sticky='W')
        self.stream_src_cbox = Combobox(self.fr_stream_src, textvariable=self.stream_src_var, values=self.C_STREAM_SRC, state='readonly')  
        self.stream_src_cbox.config(font=self.fontstyle)
        self.stream_src_cbox.grid(row=0, column=1, sticky='W')
        self.stream_src_var.trace('w', self.__cb_stream_src_changed)
        self.stream_src_var.set(self.C_STREAM_SRC[0])

        # 1.2 Static elements for stream selection and description
        Label(self.fr_stream_src, text='Stream ', pady=3, font=self.fontstyle).grid(row=1, column=0, sticky='W')
        Label(self.fr_stream_src, text='Description', pady=3, font=self.fontstyle).grid(row=2, column=0, sticky='W')


        # 2 Text box for stream description
        fr_stream_txt       = Frame(self.frame)
        fr_stream_txt.grid(row=1, column=0, sticky='W')
        self.description    = Text(fr_stream_txt, height=5, width=self.C_WIDTH_STREAM_CBOX+4)
        self.description.grid(row=0, column=0, sticky='W')


        # 3 Frame for feature selection
        self.fr_features = LabelFrame(self.frame, text='Features')
        self.fr_features.grid(row=3, column=0, sticky='W')

        Label(self.fr_features, text='Number of features  ', pady=3, font=self.fontstyle).grid(row=0, column=0, sticky='W')
        self.num_features  = Entry(self.fr_features, textvariable=self.num_features_var, width=8, justify='right', font=self.fontstyle)
        self.num_features.grid(row=0, column=1, sticky='W')

        Label(self.fr_features, text='Feature #1', pady=3, font=self.fontstyle).grid(row=1, column=0, sticky='W')
        Label(self.fr_features, text='Feature #2', pady=3, font=self.fontstyle).grid(row=2, column=0, sticky='W')
        Label(self.fr_features, text='Feature #3', pady=3, font=self.fontstyle).grid(row=3, column=0, sticky='W')


        # 5 Button for submitting the benchmark test
        self.submit = Button(self.frame, text='Run', command=self.__cb_stream_run)
        self.submit.grid(row=4, column=1, sticky='E')


## -------------------------------------------------------------------------------------------------
    def __cb_stream_src_changed(self, *args):
        stream_src = self.stream_src_var.get()
        self.log(self.C_LOG_TYPE_I, 'Stream source changed to "' + stream_src + '"')

        if stream_src == self.C_STREAM_SRC_NATIVE:
            if len(self.stream_list_native_cbox)==0: self.__fill_stream_list_native() 
            stream_list_cbox = self.stream_list_native_cbox
 
        elif stream_src == self.C_STREAM_SRC_OPENML:
            if len(self.stream_list_openml_cbox)==0: self.__fill_stream_list_openml() 
            stream_list_cbox = self.stream_list_openml_cbox

        if self.stream_cbox is not None:
            self.stream_cbox.destroy()
            self.stream_cbox = None

        self.stream_cbox = Combobox(self.fr_stream_src, textvariable=self.stream_var, values=stream_list_cbox, state='readonly')
        self.stream_cbox.config(width=self.C_WIDTH_STREAM_CBOX, font=self.fontstyle)
        self.stream_cbox.grid(row=1, column=1, sticky='W')
        self.stream_var.trace('w', self.__cb_stream_changed)
        self.stream_var.set(stream_list_cbox[0])


## -------------------------------------------------------------------------------------------------
    def __cb_stream_changed(self, *args):
        stream = self.stream_var.get()
        self.log(self.C_LOG_TYPE_I, 'Stream changed to "' + stream + '"')
#        self.__stream_show_details(stream)


## -------------------------------------------------------------------------------------------------
    def __stream_show_details(self, p_stream):
        for stream in self.streams:
            if stream[0] == p_stream: break

        # 1 Instantiate stream object
        self.shared_db.iis_stream = stream[1]()

        # 2 Update description
        self.description.delete('1.0', 'end')
        self.description.insert(tkinter.END, stream[1].C_DESCRIPTION)
        self.num_features_var.set(stream[1].C_FEATURES)

        # 3 Update feature elements
        self.feature_list = ['x1', 'x2', 'x3']

        # 3.1 Feature #1
        if self.feature1_cbox != None: 
            self.feature1_cbox.destroy()
            self.feature1 = None

        if self.shared_db.iis_stream.C_FEATURES > 0:
            self.feature1_cbox  = Combobox(self.fr_features, textvariable=self.feature1_var, values=self.feature_list, state='readonly')
            self.feature1_cbox.config(width=30, font=self.fontstyle)
            self.feature1_cbox.grid(row=1, column=1)
            self.feature1_var.set(self.feature_list[0])
            self.feature1_var.trace('w', self.__cb_feature1_changed)

        # 3.2 Feature #2
        if self.feature2_cbox != None: 
            self.feature2_cbox.destroy()
            self.feature2 = None

        if self.shared_db.iis_stream.C_FEATURES > 1:
            self.feature2_cbox  = Combobox(self.fr_features, textvariable=self.feature2_var, values=self.feature_list, state='readonly')
            self.feature2_cbox.config(width=30, font=self.fontstyle)
            self.feature2_cbox.grid(row=2, column=1)
            self.feature2_var.set(self.feature_list[1])
            self.feature2_var.trace('w', self.__cb_feature1_changed)

        # 3.3 Feature #3
        if self.feature2_cbox != None: 
            self.feature2_cbox.destroy()
            self.feature2 = None

        if self.shared_db.iis_stream.C_FEATURES > 2:
            self.feature3_cbox  = Combobox(self.fr_features, textvariable=self.feature3_var, values=self.feature_list, state='readonly')
            self.feature3_cbox.config(width=30, font=self.fontstyle)
            self.feature3_cbox.grid(row=3, column=1)
            self.feature3_var.set(self.feature_list[2])
            self.feature3_var.trace('w', self.__cb_feature1_changed)


        # 4 Update further elements
        # ...


## -------------------------------------------------------------------------------------------------
    def __cb_feature1_changed(self, *args):
        feature1 = self.feature1_var.get()
        self.log(self.C_LOG_TYPE_I, 'Feature #1 changed to "' + feature1 + '"')


## -------------------------------------------------------------------------------------------------
    def __cb_stream_run(self, *args):
        benchmark   = self.stream_var.get()
 
        for bm in self.streams:
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
        self.add_component(IISParamStream(p_shared_db=self.shared_db, p_row=1, p_col=0, p_pady = 5, p_visible=True, p_logging=self._level))





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

        # 1 Create subplots for each input space dimension
        num_dim     = self.shared_db.iis_ispace.get_num_dim() 
        height_perc = self.C_HEIGHT_VAR_PLOTS / num_dim

        for i, dim_id in enumerate(self.shared_db.iis_ispace.get_dim_ids()):
            dim = self.shared_db.iis_ispace.get_dim(dim_id)
            iisivar_comp = IISIVar(self.shared_db, p_row=i, p_col=0, p_title='Input Variable ' + dim.get_name_short(), p_width_perc=0.25, p_height_perc=height_perc, p_logging=self._level)
            iisivar_comp.assign_dim(i)
            self.add_component(iisivar_comp)


        # 2 Create tab control with a tab for each parameter group
        tab_ctrl = SciUITabCTRL(self.shared_db, p_row=num_dim, p_col=0, p_title='Configuration', p_visible=True, p_logging=self._level)
        tab_ctrl.add_tab('Stream', IISParamStream(p_shared_db=self.shared_db, p_row=0, p_col=0, p_pady=5, p_visible=False, p_logging=self._level))
        tab_ctrl.add_tab('Preprocessing', IISParamGeneral(p_shared_db=self.shared_db, p_row=0, p_col=0, p_pady=5, p_visible=False, p_logging=self._level))
        tab_ctrl.add_tab('Processing', None)
        tab_ctrl.add_tab('Control',None)
        self.add_component(tab_ctrl)





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