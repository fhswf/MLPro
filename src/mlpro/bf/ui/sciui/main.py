## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.bf.ui.scui
## -- Module  : main
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-04-13  0.0.0     DA       Creation
## -- 2021-06-20  0.5.0     DA       Pre-release with basic functionality
## -- 2021-07-05  0.6.0     DA       Pre-release with basic functionality
## -- 2021-09-11  0.6.0     MRD      Change Header information to match our new library name
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.6.0 (2021-07-05)

Provides the SciUI application class.
"""


from tkinter import *
import tkinter.messagebox
from mlpro.bf.ui.sciui.framework import *





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SciUI(SciUIWindow):

    C_TYPE          = 'Application'
    C_NAME          = 'SciUI' 
    C_VERSION       = '0.6.0'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_fullscreen=False, p_autoscan_scenarios=True, p_start_immediately=True, p_logging=False):
        """
        Parameters:
            p_fullscreen                If True, application starts in fullscreen mode.
            p_autoscan_screnarios       If True, all implemented scenario classes will be detected and 
                                        included.
            p_start_imediately          If True, application starts directly.
            p_logging                   Boolean switch for logging
        """

        super().__init__(p_logging=p_logging)

        # Init scenario handling
        self.scenario_pool      = []
        self.scenario_id        = IntVar()
        self.scenario_id.set(0)
        self.scenario           = None

        # Init shared db
        self.__init_shared_db()
        self.shared_db.window_fullscreen    = not p_fullscreen
        
        self.ignore_resize                  = True
        
        # Autoscan scenario classes
        if p_autoscan_scenarios:
            for cls in SciUIScenario.__subclasses__():
                self.register_scenario(p_scenario=cls, p_recursive=True, p_sort=False)  
                
            self.scenario_pool.sort()  
            
        # Start immediately, if requestsd
        if p_start_immediately: self.start()

        
## -------------------------------------------------------------------------------------------------
    def __init_shared_db(self): 
        self.shared_db = SciUISharedDB(self)
        self.shared_db.refresh_all.trace('w', self.__cb_global_refresh)

              
## -------------------------------------------------------------------------------------------------
    def __init_main_menu(self):       
        # 0 Init main menu
        menu = Menu(self, tearoff=False)
        self.config(menu=menu)
        
        # 1 Init Tk-Variables for menu control
        self.window_fullscreen = BooleanVar()
        self.window_fullscreen.set(self.shared_db.window_fullscreen)
        self.window_fullscreen.trace('w', self.__cb_toggle_fullscreen)
        
        # 2 Submenu 'File'
        file_menu = Menu(menu, tearoff=False)
        menu.add_cascade(label='File', menu=file_menu)
        # file_menu.add_command(label='Open Configuration', state='disabled', command=self.__cb_menu_file_open_configuration)
        # file_menu.add_command(label='Save Configuration', state='disabled', command=self.__cb_menu_file_save_configuration)
        # file_menu.add_separator()
        
        # 2.1 Submenu 'File/Change Scenario'
        scenario_menu = Menu(menu, tearoff=True)
        file_menu.add_cascade(label='Change Scenario', menu=scenario_menu)
        
        i = 0
        for scenario in self.scenario_pool:
            if scenario[1].C_RELEASED:
                state = 'active'
            else:
                state = 'disabled'
            scenario_menu.add_radiobutton(label=scenario[1].C_NAME + ' (Ver. ' + scenario[1].C_VERSION + ')', state=state, variable=self.scenario_id, value=i, command=self.__cb_submenu_file_change_scenario)
            i += 1
            
        file_menu.add_separator()
        
        # 2.2 Submenu 'File/Properties'
        file_menu.add_command(label='Properties', state='disabled', command=self.__cb_menu_file_properties)       
        file_menu.add_separator()
        
        file_menu.add_command(label='Exit', command=self.quit)

        # 3 Submenu 'View'
        view_menu = Menu(menu, tearoff=False)
        menu.add_cascade(label='View', menu=view_menu)
        view_menu.add_checkbutton(label='Fullscreen', onvalue=True, offvalue=False, variable=self.window_fullscreen)
        # view_menu.add_separator()
        # view_menu.add_command(label='Color Profile', state='disabled', command=self.__cb_menu_view_color_profile)
        
        # 4 Submenu 'Help'
        help_menu = Menu(menu, tearoff=False)
        menu.add_cascade(label='Help', menu=help_menu)
        help_menu.add_command(label='About...', command=self.__cb_menu_help_about)


## -------------------------------------------------------------------------------------------------
    def __init_main_window(self):
        self.ignore_resize = True
        self.title(self.C_NAME)

        try:
            self.attributes('-zoomed', True)
        except:
            self.state('zoomed')

        self.__set_fullscreen(not self.shared_db.window_fullscreen)  
        self.__init_main_menu()
        self.bind("<Configure>", self.__cb_window_resized)      
        

## -------------------------------------------------------------------------------------------------
    def start(self):
        
        # Check: at least one scenario registered?
        if len(self.scenario_pool) == 0: 
            self.log(self.C_LOG_TYPE_I, 'Please register at least one scenario')
            # exception...
            return 
        
        # Init main window
        self.__init_main_window()
        
        # Start application
        super().start()   
        

## -------------------------------------------------------------------------------------------------
    def register_scenario(self, p_scenario:SciUIScenario, p_recursive=False, p_sort=True): 
        self.__register_scenario_rec(p_scenario=p_scenario, p_recursive=p_recursive)
        if p_sort: self.scenario_pool.sort()      


## -------------------------------------------------------------------------------------------------
    def __register_scenario_rec(self, p_scenario:SciUIScenario, p_recursive): 
        if p_scenario.C_VISIBLE: 
            self.scenario_pool.append([p_scenario.C_NAME, p_scenario])
            self.log(self.C_LOG_TYPE_I, 'Scenario "' + p_scenario.C_NAME + ' Ver. (' + p_scenario.C_VERSION + ')" registered')

        if not p_recursive: return 
        
        for cls_child in p_scenario.__subclasses__():
            self.__register_scenario_rec(p_scenario=cls_child, p_recursive=p_recursive)


## -------------------------------------------------------------------------------------------------
    def __change_scenario(self):
        self.log(self.C_LOG_TYPE_W, '!!!')
        self.log(self.C_LOG_TYPE_W, 'GAP DEL SELF.SCENARIO')
        self.log(self.C_LOG_TYPE_W, '!!!')
#         del self.scenario
#         self.scenario = None
#         
        if self.scenario != None:
            # self.scenario.__del__()
            self.scenario = None  
              
        self.scenario         = self.scenario_pool[self.scenario_id.get()][1](self.shared_db, p_logging=self._level)
        self.shared_db.start_global_refresh()
        
        self.title(self.C_NAME + ': Scenario "' + self.scenario.C_NAME + '"')


## -------------------------------------------------------------------------------------------------
    def __set_fullscreen(self, p_fullscreen): 
        if self.shared_db.window_fullscreen == p_fullscreen: return
        self.log(self.C_LOG_TYPE_I, 'Main window fullscreen: ', p_fullscreen)
        self.attributes('-fullscreen', p_fullscreen)
        
        self.shared_db.window_fullscreen    = p_fullscreen
        self.shared_db.window_resized       = True
        

## -------------------------------------------------------------------------------------------------
    def __cb_global_refresh(self, *args):
        self.log(self.C_LOG_TYPE_I, 'Global Refresh')
        
        self.shared_db.refresh_full = True
        self.shared_db.redraw       = True
        if self.scenario != None: self.scenario.refresh()
        self.shared_db.refresh_full = False  #after_refresh()        


## -------------------------------------------------------------------------------------------------
    def __cb_toggle_fullscreen(self, *args):
        self.__set_fullscreen(self.window_fullscreen.get())        
        

## -------------------------------------------------------------------------------------------------
    def __cb_menu_file_properties(self):
        self.log(self.C_LOG_TYPE_I, 'Main Menu action: File/Properties')
        
        # Switch for Logging/Log Level


## -------------------------------------------------------------------------------------------------
    def __cb_menu_help_about(self):
        self.log(self.C_LOG_TYPE_I, 'Main Menu action: Help/About')    
        tkinter.messagebox.showinfo('About SciUI', 'SciUI - Scientific User Interface\nVersion ' + self.C_VERSION)


## -------------------------------------------------------------------------------------------------
    def __cb_submenu_file_change_scenario(self):
        self.log(self.C_LOG_TYPE_I, 'Sub Menu File/Change scenario:', self.scenario_id.get())
        self.__change_scenario()    


## -------------------------------------------------------------------------------------------------
    def __cb_window_resized(self, event):
        if event.widget != self: return
        if ( event.width == self.shared_db.window_width ) and ( event.height == self.shared_db.window_height ): return
        
        if self.ignore_resize:
            self.ignore_resize = False
            return

        self.log(self.C_LOG_TYPE_I, 'Main Window resized to {}x{}'.format(event.width, event.height))

        # Update screen and window properties
        self.shared_db.window_width     = event.width
        self.shared_db.window_height    = event.height

        # Rebuild active scenario
        self.shared_db.window_resized   = True
        self.__change_scenario()
        self.shared_db.window_resized   = False
