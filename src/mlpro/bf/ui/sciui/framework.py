## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.ui.scui
## -- Module  : framework
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-04-13  0.0.0     DA       Creation
## -- 2021-07-05  1.0.0     DA       Release of first version
## -- 2021-07-29  1.1.0     DA       Class SciUITabCTRL added
## -- 2021-09-11  1.1.0     MRD      Change Header information to match our new library name
## -- 2022-01-06  1.1.1     DA       Adjustments for matplotlib 3.5
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.1.1 (2022-01-06)

SciUI framework classes to be reused in own SciUI scenarios. Needs Matplotlib 3.3 or higher.
"""


import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'       # ... to avoid dll-exception under Windows

from math import floor
from pathlib import Path


from tkinter import *
import tkinter.font as tkFont
from tkinter.filedialog import askdirectory
from tkinter import ttk
from tkinter.ttk import Combobox

import matplotlib
from matplotlib.figure import Figure
from matplotlib.widgets import Cursor
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg 
from mpl_toolkits.mplot3d.axes3d import Axes3D

from mlpro.bf.various import *





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SciUISharedDB(): 
    """
    Container for scenario-internal data exchange and communication. Can be extended while runtime 
    by consuming classes.
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_root=None): 
        
        # 1 Root object
        self.root               = p_root

         # 2 Screen 
        self.screen_dpi         = 0

        # 3 Application window 
        self.window_height      = 0
        self.window_width       = 0 
        self.window_resized     = False
        self.window_fullscreen  = False
        
        # 4 Application control
        self.refresh_all        = IntVar()
        self.refresh_all.set(0)
        self.refresh_full       = True
        self.redraw             = True          # Switch for graphical output
        
        
## -------------------------------------------------------------------------------------------------
    def start_global_refresh(self):   
        self.refresh_all.set( 1 - self.refresh_all.get() )
        
        
             


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SciUIRoot(Log): 
    """
    SciUI root class with overarching properties.
    """
    
    pass





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SciUIWindow(SciUIRoot, Tk):
    """
    Root class for SciUI window apllications.
    """

    C_TYPE          = 'SciUI Window'
    C_NAME          = '????'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_logging=False):
        SciUIRoot.__init__(self, p_logging=p_logging)
        Tk.__init__(self)
        self.title(self.C_NAME)


## -------------------------------------------------------------------------------------------------
    def start(self):
        self.mainloop()       





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SciUICursor (SciUIRoot, Cursor):
    """
    Enriched matplotlib cursor widget.
    """
    
## -------------------------------------------------------------------------------------------------
    def __init__(self, ax, horizOn=True, vertOn=True, useblit=False, **lineprops):
        Cursor.__init__(self, ax, horizOn=horizOn, vertOn=vertOn, useblit=useblit, **lineprops)
        self.cb_button_press_event      = None
        self.cb_button_release_event    = None
        self.cb_motion_notify_event     = None
        self.st_button_press_event      = False
        self.st_button_release_event    = False
        self.st_motion_notify_event     = False

         
## -------------------------------------------------------------------------------------------------
    def connect_event(self, event, callback):
        if event == 'button_press_event':
            self.cb_button_press_event      = callback
            Cursor.connect_event(self, event, self.onbuttonpressed)
        elif event == 'button_release_event':
            self.cb_button_release_event    = callback
            Cursor.connect_event(self, event, self.onbuttonreleased)
        elif event == 'motion_notify_event':
            self.cb_motion_notify_event     = callback
            Cursor.connect_event(self, event, self.onmove)
        else:
            Cursor.connect_event(self, event, callback)           

        
## -------------------------------------------------------------------------------------------------
    def set_event_status(self, event, status):
        if event == 'button_press_event':
            self.st_button_press_event      = status
        elif event == 'button_release_event':
            self.st_button_release_event    = status
        elif event == 'motion_notify_event':
            self.st_motion_notify_event     = status        

        
## -------------------------------------------------------------------------------------------------
    def onmove(self, event):
        Cursor.onmove(self, event)
        if ( event.inaxes != self.ax ) or ( self.st_motion_notify_event == False ) or ( event.button == None ) or ( event.xdata == None ) or ( event.ydata == None ): return 
        if self.cb_motion_notify_event != None: self.cb_motion_notify_event(event)       
        
        
## -------------------------------------------------------------------------------------------------
    def onbuttonpressed(self, event):
        if ( event.inaxes != self.ax ) or ( self.st_button_press_event == False ) or ( event.dblclick ) or ( event.button != 1 ): return 
        self.cb_button_press_event(event)       

        
## -------------------------------------------------------------------------------------------------
    def onbuttonreleased(self, event):
        if ( event.inaxes != self.ax ) or ( self.st_button_release_event == False ): return 
        self.cb_button_release_event(event)       





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SciUITooltip(object):
    """
    Enriched tooltip class.
    """

## -------------------------------------------------------------------------------------------------
    def __init__(self, widget, text='widget info'):
        self.waittime = 500     #miliseconds
        self.wraplength = 180   #pixels
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None

    
## -------------------------------------------------------------------------------------------------
    def enter(self, event=None):
        self.schedule()


## -------------------------------------------------------------------------------------------------   
    def leave(self, event=None):
        self.unschedule()
        self.hidetip()


## -------------------------------------------------------------------------------------------------   
    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.showtip)


## -------------------------------------------------------------------------------------------------   
    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)


## -------------------------------------------------------------------------------------------------   
    def showtip(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        # creates a toplevel window
        self.tw = Toplevel(self.widget)
        # Leaves only the label and removes the app window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = Label(self.tw, text=self.text, justify='left',
                       background="#ffffff", relief='solid', borderwidth=1,
                       wraplength = self.wraplength)
        label.pack(ipadx=1)


## -------------------------------------------------------------------------------------------------   
    def hidetip(self):
        tw = self.tw
        self.tw= None
        if tw:
            tw.destroy()
            
            
            
            

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SciUIComponent(SciUIRoot): 
    """
    Elementry screen object in SciUI framework.
    """

    C_TYPE          = 'SciUI Component'
    C_NAME          = ''
            
## -------------------------------------------------------------------------------------------------
    def __init__(self, p_shared_db:SciUISharedDB, p_logging=True):
        """
        Parameters:
            p_shared_db         Shared DB object for scenario-internal communication
            p_logging           Boolean switch for logging 
        """

        super().__init__(p_logging=p_logging)
        self.shared_db      = p_shared_db
        self.init_component()


## -------------------------------------------------------------------------------------------------
    def init_component(self): 
        """
        Initialization of component-specific elements at instance creation time. To be redefined.
        """

        pass

        
## -------------------------------------------------------------------------------------------------
    def get_name(self): 
        return self.C_NAME        
        
        
## -------------------------------------------------------------------------------------------------
    def refresh(self, p_parent_frame=None): 
        """
        Refresh of all component-specific elements. To be redefined. Please call super().refresh()
        at the beginning of your own implementation.

        Parameters:
            p_parent_frame      Parent frame object
        """

        self.parent_frame = p_parent_frame
        self.log(self.C_LOG_TYPE_I, 'refresh')        





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SciUIFrame(SciUIComponent): 
    """
    Enriched wrapper class for the Tkinter (Label-)Frame class, based on the Tkinter grid positioning model.
    """
    
    C_TYPE          = 'SciUI Frame'
    C_NAME          = ''
        
## -------------------------------------------------------------------------------------------------
    def __init__(self, p_shared_db:SciUISharedDB, p_row, p_col, p_title=None, p_width_perc=0.0, p_height_perc=0.0, p_visible=False, p_padx=5, p_pady=0, p_sticky='NW', p_logging=True): 
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
        """

        self.row            = p_row
        self.col            = p_col
        self.width_perc     = p_width_perc              # Frame width in percent of main window width
        self.height_perc    = p_height_perc             # Frame height in percent of main window height
        self.width_pix      = p_shared_db.window_width * self.width_perc
        self.height_pix     = p_shared_db.window_height * self.height_perc
        self.frame_visible  = p_visible

        if p_title == None:
            self.frame_text     = self.C_NAME
        else:
            self.frame_text     = p_title

        self.frame          = None
        self.components     = []
        self.sticky         = p_sticky
        self.padx           = p_padx
        self.pady           = p_pady
        self.popup_menu     = None
        
        super().__init__(p_shared_db, p_logging=p_logging)       
        
        
## -------------------------------------------------------------------------------------------------
    def init_popup_menu(self): 
        """
        Initializes popup menu of the component. To be redefined.
        """
        
        pass

#        Sample popup menu coding
#         if self.popup_menu != None: return
        
#         self.popup_menu = Menu(self.shared_db.root, tearoff=False)
#         self.popup_menu.add_command(label='My first entry!') #, command=self.__cb_my_first_entry)           
        
        
## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()
        self.init_popup_menu()       
        
        
## -------------------------------------------------------------------------------------------------
    def cb_popup_menu(self, p_event):
        self.log(self.C_LOG_TYPE_I, 'popup menu opened')
        self.popup_menu.tk_popup(p_event.x_root, p_event.y_root)            
        
        
## -------------------------------------------------------------------------------------------------
    def determine_frame_size(self):
        self.width_pix      = 0
        self.height_pix     = 0
        
        if self.width_perc != 0.0:
            self.width_pix  = floor(self.shared_db.window_width * self.width_perc)

        if self.height_perc != 0.0:
            self.height_pix  = floor(self.shared_db.window_height * self.height_perc)
            
        if self.width_pix == 0:
            self.width_pix = self.height_pix

        if self.height_pix == 0:
            self.height_pix = self.width_pix
        
        
## -------------------------------------------------------------------------------------------------
    def refresh(self, p_parent_frame=None):
        super().refresh(p_parent_frame)
        self.determine_frame_size()
        
        if self.frame == None:
            if self.frame_visible:
                self.frame = LabelFrame(self.parent_frame,text=self.frame_text, padx=self.padx, pady=self.pady) 
                self.frame.grid(row=self.row, column=self.col, sticky=self.sticky)               
            else:
                self.frame = Frame(self.parent_frame, padx=self.padx, pady=self.pady) 
                self.frame.grid(row=self.row, column=self.col, sticky=self.sticky)

            self.init_popup_menu()
            if self.popup_menu != None: 
                self.frame.bind("<Button-3>",self.cb_popup_menu)
            
        for comp in self.components: comp.refresh(self.frame)    
    
        
## -------------------------------------------------------------------------------------------------
    def add_component(self, p_component:SciUIComponent):
        self.components.append(p_component)             
        
        
## -------------------------------------------------------------------------------------------------
    def __del__(self):
        del self.components[:]
        
        if self.frame != None: 
            self.frame.grid_forget()
            self.frame.grid_remove()
            self.frame = None
              




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SciUITabCTRL(SciUIFrame):
    """
    Enriched wrapper class for the Tkinter tab control.
    """

    C_TYPE          = 'SciUI Tabs'

## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()
        self.tab_list   = []
        self.tab_ctrl   = None       


## -------------------------------------------------------------------------------------------------
    def refresh(self, p_parent_frame):
        super().refresh(p_parent_frame=p_parent_frame)

        if self.tab_ctrl == None:
            self.tab_ctrl = ttk.Notebook(self.frame)
            self.tab_ctrl.grid(column=self.col, row=self.row)

            for tab in self.tab_list:
                tab[1] = ttk.Frame(self.tab_ctrl)
                self.tab_ctrl.add(tab[1], text=tab[0])

        for tab in self.tab_list:
            if tab[2] != None: tab[2].refresh(tab[1])


## -------------------------------------------------------------------------------------------------
    def add_component(self, p_component: SciUIComponent):
        pass

    
## -------------------------------------------------------------------------------------------------
    def add_tab(self, p_tab_name, p_component:SciUIComponent):
        self.tab_list.append([p_tab_name, None, p_component])





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SciUISubplotSaveDLG(SciUIWindow):
    """
    Small SciUI window application to choose folder and file name for saving a SciUI subplot.
    """

    C_NAME          = 'Save Plot'
    C_FONT_FAMILY   = 'Lucida Grande'
    C_FONT_SIZE     = 10 
    C_FILENAME      = 'myplot'
    C_SUFFIXES      = ['.pdf', '.png', '.svg']

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_fname=None, p_xpos=500, p_ypos=300, p_logging=False):
        """
        Parameters:
            p_fname         Default filename
            p_xpos          X position of window
            p_ypos          Y position of window
            p_logging       Boolean switch for logging
        """

        super().__init__(p_logging=p_logging)

        if ( p_fname is None ) or ( p_fname == '' ):
            self.filename   = self.C_FILENAME
        else:
            self.filename   = p_fname

        self.geometry("+%d+%d" %(p_xpos,p_ypos))
        self.attributes("-topmost", True)

        self.fontstyle  = tkFont.Font(family=self.C_FONT_FAMILY, size=self.C_FONT_SIZE)
        self.frame      = LabelFrame(self, padx=5, pady=5, font=self.fontstyle) 
        self.frame.grid(row=0, column=0)               

        # Row 1 - Destination folder
        Label(self.frame, text='Folder', font=self.fontstyle).grid(row=1, column=0, sticky='w')     
        self.folder_var = StringVar(self)
        self.folder_var.set(Path.home())
        self.folder     = Entry(self.frame, textvariable=self.folder_var, width=40, justify='left', state='readonly', font=self.fontstyle)
        self.folder.grid(row=1, column=1)
        self.but_chg_folder = Button(self.frame, text='Change', command=self.__cb_button_folder)
        self.but_chg_folder.grid(row=1, column=2)

        # Row 2 - Filename and suffix
        Label(self.frame, text='File name', font=self.fontstyle).grid(row=2, column=0, sticky='w')     
        self.fname_var  = StringVar(self)
        self.fname_var.set(self.filename)
        self.fname      = Entry(self.frame, textvariable=self.fname_var, width=40, justify='left', background='white', font=self.fontstyle)
        self.fname.grid(row=2, column=1)

        self.suffix_var = StringVar(self)
        self.suffix_var.set(self.C_SUFFIXES[0])
        self.suffix     = Combobox(self.frame, textvariable=self.suffix_var, width=4, justify='left', state='readonly', font=self.fontstyle, values=self.C_SUFFIXES)
        self.suffix.grid(row=2, column=2, sticky='w')

        # Row 3 - Empty
        Label(self.frame, text='', font=self.fontstyle).grid(row=3, column=0)     

        # Row 4 - Buttons Save/Cancel
        self.but_cancel = Button(self.frame, text='Cancel', width=6, command=self.__cb_button_cancel)
        self.but_cancel.grid(row=4, column=2)

        self.but_ok     = Button(self.frame, text='Save', width=6, command=self.__cb_button_ok)
        self.but_ok.grid(row=4, column=3)


## -------------------------------------------------------------------------------------------------
    def get_filename(self):
        return self.filename


## -------------------------------------------------------------------------------------------------
    def __cb_button_ok(self):
        self.filename = self.folder_var.get() + os.sep + self.fname_var.get() + self.suffix_var.get()
        self.quit()


## -------------------------------------------------------------------------------------------------
    def __cb_button_cancel(self):
        self.filename = ''
        self.quit()


## -------------------------------------------------------------------------------------------------
    def __cb_button_folder(self):
        f = askdirectory(initialdir=self.folder_var.get(), mustexist=True)
        if f != (): self.folder_var.set(f)
 
        self.focus_force()





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SciUISubplotRoot(SciUIFrame): 
    """
    Root class for specialized frame classes that embedd a matplotlib figure with one subplot into a
    Tkinter frame. Not intended for direct reuse. Please use inherited classes SciUISubplot2D, 
    SciUISubplot3D instead. 
    """

    C_TYPE              = 'SciUI Subplot'

    C_BACKEND           = 'TkAgg'
    
    C_FIG_FACECOLOR     = 'white'

    C_AX_RECTANGLE      = [0.1, 0.1, 0.85, 0.85]
    C_AX_FRAME          = True
    C_AX_FACECOLOR      = 'white'

## -------------------------------------------------------------------------------------------------
    def create_subplot(self):
        """
        Internally used to create a suitable subplot. New subplot needs to be bound to self.ax. To be
        redefined.
        """

        self.ax = None


## -------------------------------------------------------------------------------------------------
    def init_popup_menu(self): 
        if self.popup_menu != None: return
        SciUIFrame.init_popup_menu(self)
        
        self.popup_menu = Menu(self.shared_db.root, tearoff=False)
        self.popup_menu.add_command(label='Undock', state='disabled') #, command=self.__cb_my_first_entry)
        self.popup_menu.add_separator()
        self.popup_menu.add_command(label='Save As...', command=self.cb_save_plot) 

        
## -------------------------------------------------------------------------------------------------
    def init_component(self):
        """
        Initialization of component-specific elements at instance creation time. To be redefined.
        Please call super().init_component() at beginning of your implementation.
        """

        SciUIFrame.init_component(self)

        if ( self.width_perc == 0.0 ) and ( self.height_perc == 0.0 ):
            raise Exception ('Subplots need to be sized explicitly!')

        if matplotlib.get_backend() != self.C_BACKEND:
            matplotlib.use(self.C_BACKEND)

        self.padx               = 0
        self.pady               = 0
        self.frame_visible      = True
        self.dpi                = 100               # Matplotlib default density
        self.figure             = None
        self.canvas             = None
        self.width_inch         = 0
        self.width_inch_bak     = 0
        self.height_inch        = 0
        self.height_inch_bak    = 0
        self.set_flush_events(True)
        
        
## -------------------------------------------------------------------------------------------------
    def determine_frame_size(self):
        SciUIFrame.determine_frame_size(self)
        
        # Determine plot size in inches
        self.width_inch     = self.width_pix / self.dpi
        self.height_inch    = self.height_pix / self.dpi

        
## -------------------------------------------------------------------------------------------------
    def set_flush_events(self, p_flush:bool):
        self.canvas_flush_events = p_flush


## -------------------------------------------------------------------------------------------------
    def refresh(self, p_parent_frame):
        SciUIFrame.refresh(self, p_parent_frame)
        
        # 1 Refresh Matplotlib figure, canvas, ax
        if self.figure == None:
            self.figure = Figure(figsize=(self.width_inch, self.height_inch), dpi=self.dpi, facecolor=self.C_FIG_FACECOLOR)
            
            # 1.1 Embedd figure into matplotlib backend
            self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame) 
            self.create_subplot()

            if self.popup_menu != None: self.canvas.get_tk_widget().bind("<Button-3>",self.cb_popup_menu)
            
        elif ( self.width_inch != self.width_inch_bak ) or ( self.height_inch != self.height_inch_bak ):
            self.figure.set_size_inches(self.width_inch, self.height_inch) 
            self.width_inch_bak     = self.width_inch
            self.height_inch_bak    = self.height_inch
        
        # 2 Refresh custom objects
        self.refresh_custom()
        
        # 3 Update view
        if not self.shared_db.redraw: return
        self.canvas.draw()
        if self.canvas_flush_events: self.canvas.flush_events()
        self.canvas.get_tk_widget().grid()        
        
        
## -------------------------------------------------------------------------------------------------
    def refresh_custom(self):    
        """
        Additional refresh activities. To be redefined.
        """

        self.log(self.C_LOG_TYPE_I, 'refresh custom objects')

      
## -------------------------------------------------------------------------------------------------
    def add_component(self, p_component): 
        """
        Adding further subcomponents is disabled here.
        """
        pass
    
        
## -------------------------------------------------------------------------------------------------
    def cb_save_plot(self):
        self.log(self.C_LOG_TYPE_I, 'Plot save dialog called from context menu')

        dlg = SciUISubplotSaveDLG(p_fname=self.frame_text, p_logging=self._level)
        dlg.start()
        f   = dlg.get_filename()
        if f != '':
            self.figure.savefig(f)
            self.log(self.C_LOG_TYPE_I, 'Plot saved to "' + f + '"')
        else:
            self.log(self.C_LOG_TYPE_I, 'Plot save dialog cancelled')

        dlg.destroy()


## -------------------------------------------------------------------------------------------------
    def __del__(self):
        if self.figure != None:
            if self.ax != None:
                self.figure.delaxes(self.ax) 
                self.ax = None
                
            self.figure = None
            
        try:
            SciUIFrame.__del__(self)
        except:
            pass
    
    



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SciUISubplot2D(SciUISubplotRoot): 
    """
    Specialized frame class that embedds a 2D matplotlib figure with one subplot into a Tkinter 
    frame. A cross hair cursor functinality with mouse event handling can be switched on/off via
    class constant C_AX_CURSOR.  
    """

    C_TYPE              = 'SciUI 2D Subplot'
    C_AX_RECTANGLE      = [0.1, 0.1, 0.85, 0.85]
    C_AX_CURSOR         = False
    C_AX_CURSOR_COLOR   = 'blue'   

## -------------------------------------------------------------------------------------------------
    def create_subplot(self):
        self.ax =  self.figure.add_axes(self.C_AX_RECTANGLE, frame_on=self.C_AX_FRAME, fc=self.C_AX_FACECOLOR)      
        if self.C_AX_CURSOR: self.__init_cursor()


## -------------------------------------------------------------------------------------------------
    def __init_cursor(self):   
        self.cursor = SciUICursor(self.ax, useblit=True, color=self.C_AX_CURSOR_COLOR, linewidth=0.5, linestyle='--' )
        self.cursor.connect_event('button_press_event', self.cb_mbutton_pressed)
        self.cursor.connect_event('button_release_event', self.cb_mbutton_released)
        self.cursor.connect_event('motion_notify_event', self.cb_mouse_moved)
        self.cursor.set_event_status('button_press_event', True)

        
## -------------------------------------------------------------------------------------------------
    def cb_mbutton_pressed(self, event):
        self.log(self.C_LOG_TYPE_I, 'left mouse button pressed')
        self.cursor.set_event_status('motion_notify_event', True)
        self.cursor.set_event_status('button_release_event', True)  

        
## -------------------------------------------------------------------------------------------------
    def cb_mouse_moved(self, event):
        self.log(self.C_LOG_TYPE_I, 'mouse moved')

        
## -------------------------------------------------------------------------------------------------
    def cb_mbutton_released(self, event): 
        self.log(self.C_LOG_TYPE_I, 'left mouse button released')        
        self.cursor.set_event_status('motion_notify_event', False)
        self.cursor.set_event_status('button_release_event', False)
        




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SciUISubplot3D(SciUISubplotRoot): 
    """
    Specialized frame class that embedds a 3D matplotlib figure with one subplot into a Tkinter frame.  
    """

    C_TYPE              = 'SciUI 3D Subplot'
    C_AX_RECTANGLE      = [0.1, 0.1, 0.85, 0.85]

## -------------------------------------------------------------------------------------------------
    def create_subplot(self):
        self.ax = Axes3D(self.figure, auto_add_to_figure=False, rect=self.C_AX_RECTANGLE, proj_type='persp')
        self.figure.add_axes(self.ax)





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SciUIFrameParam(SciUIFrame):
    """
    Class for parameter/text frames.
    """ 

    C_TYPE          = 'SciUI Parameter Frame'
    C_NAME          = ''
    
    C_FONT_FAMILY   = 'Lucida Grande'
    C_FONT_SIZE     = 10    
 
## -------------------------------------------------------------------------------------------------
    def init_component(self):
        super().init_component()
        self.fontstyle = tkFont.Font(family=self.C_FONT_FAMILY, size=self.C_FONT_SIZE)
 
 
 


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SciUIScenario(SciUIFrame): 
    """
    Top level class for an entire SciUI scenario that can be registered by the SciUI application 
    class. SciUI scenarios are visible and chooseable if the switches C_RELEASED and C_VISIBLE are
    set to True.
    """
    
    C_TYPE          = 'SciUI Scenario'
    C_NAME          = '????'
    C_VERSION       = '0.0.0'
    C_RELEASED      = False
    C_VISIBLE       = False    
        
## -------------------------------------------------------------------------------------------------
    def __init__(self, p_shared_db:SciUISharedDB, p_logging=Log.C_LOG_ALL): 
        SciUIFrame.__init__(self, p_shared_db, p_row=0, p_col=0, p_width_perc=1, p_height_perc=1, p_visible=p_shared_db.window_fullscreen, p_logging=p_logging)

        if p_shared_db.window_fullscreen:
            self.frame_text = self.C_TYPE + ' "' + self.C_NAME + '"'

        self.padx = 3
        self.pady = 1
