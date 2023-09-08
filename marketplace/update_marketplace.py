## -------------------------------------------------------------------------------------------------
## -- Project : MLPro Marketplace
## -- Module  : update_marketplace.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-09-08  0.0.0     DA       Creation 
## -- 2023-09-08  0.1.0     DA       Basic structure
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.1.0 (2023-09-08)

This standalone module collects meta data of all whitelisted GitHub repositories based on the
template repo /fhswf/MLPro-Extension.
"""


import sys, os.path, time
from mlpro.bf.various import Log


## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Marketplace (Log):

    C_TYPE              = 'Marketplace'
    C_NAME              = 'MLPro'

    C_FNAME_WHITELIST   = 'whitelist'
    C_FNAME_BLACKLIST   = 'blacklist'
    C_FNAME_DB          = 'marketplace.csv'

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_logging=Log.C_LOG_ALL):
        super().__init__(p_logging)


## -------------------------------------------------------------------------------------------------
    def _get_new_extensions(self):
        """
        This method gets all repositories based on the template MLPro-Extension that are neigther
        whitelisted nor blacklisted.
        """

        # 0 Intro
        self.log(Log.C_LOG_TYPE_S, 'Start scanning for new extensions...')

        # 1 Scanning for new extensions
        self.log(Log.C_LOG_TYPE_E, 'Not yet implemented')

        # 2 Outro
        self.log(Log.C_LOG_TYPE_S, 'End of scanning for new extensions')
        return ['pseudo']
    

## -------------------------------------------------------------------------------------------------
    def _report_new_extensions(self, p_new_extensions : list):

        # 0 Intro
        self.log(Log.C_LOG_TYPE_S, 'Start reporting new extensions...')

        # 1 Reporting
        self.log(Log.C_LOG_TYPE_E, 'Not yet implemented')

        # 2 Outro
        self.log(Log.C_LOG_TYPE_S, 'End of report of new extensions')


## -------------------------------------------------------------------------------------------------
    def _get_approved_extensions(self):
        """
        This method gets all whitelisted MLPro extensions that are not explicitely blacklisted.
        """

        # 0 Intro
        self.log(Log.C_LOG_TYPE_S, 'Start getting approved extensions...')

        # 1 Reporting
        self.log(Log.C_LOG_TYPE_E, 'Not yet implemented')

        # 2 Outro
        self.log(Log.C_LOG_TYPE_S, 'End of getting approved extensions')


## -------------------------------------------------------------------------------------------------
    def _build_marketplace_db(self, p_approved_extensions : list):

        # 0 Intro
        self.log(Log.C_LOG_TYPE_S, 'Start building marketplace DB...')

        # 1 Reporting
        self.log(Log.C_LOG_TYPE_E, 'Not yet implemented')

        # 2 Outro
        self.log(Log.C_LOG_TYPE_S, 'End of building marketplace DB')


## -------------------------------------------------------------------------------------------------
    def _update_db(self):

        # 0 Intro
        self.log(Log.C_LOG_TYPE_S, 'Start updating the marketplace DB...')


        # 1 Check whether the whitelist file is touched after the marketplace database
        tstamp_whitelist    = None
        tstamp_blacklist    = None
        tstamp_db           = None

        # 1.1 Get modification time of whitelist file
        fname_whitelist  = sys.path[0] + os.sep + self.C_FNAME_WHITELIST
        self.log(Log.C_LOG_TYPE_I, 'Whitelist file:', fname_whitelist)        
        try:
            tstamp_whitelist = os.path.getmtime(fname_whitelist)
            self.log(Log.C_LOG_TYPE_I, 'Whitelist touched at', time.ctime(tstamp_whitelist))
        except:
            # Whitelist file not found -> Abort
            self.log(Log.C_LOG_TYPE_E, 'Whitelist file', fname_whitelist, 'not found')
            return


        # 1.2 Get modification time of blacklist file
        fname_blacklist  = sys.path[0] + os.sep + self.C_FNAME_BLACKLIST
        self.log(Log.C_LOG_TYPE_I, 'Blacklist file:', fname_blacklist)        
        try:
            tstamp_blacklist = os.path.getmtime(fname_blacklist)
            self.log(Log.C_LOG_TYPE_I, 'Blacklist touched at', time.ctime(tstamp_blacklist))
        except:
            # Blacklist file not found
            self.log(Log.C_LOG_TYPE_W, 'Blacklist file', fname_blacklist, 'not found')


        # 1.3 Get modification time of marketplace db
        fname_db  = sys.path[0] + os.sep + self.C_FNAME_DB
        self.log(Log.C_LOG_TYPE_I, 'Database file:', fname_db)        
        try:
            tstamp_db = os.path.getmtime(fname_db)
            self.log(Log.C_LOG_TYPE_I, 'DB touched at', time.ctime(tstamp_db))
        except:
            # DB file not found
            self.log(Log.C_LOG_TYPE_W, 'DB file', fname_db, 'not found')
        

        # 1.4 Check: DB file up-to-date?
        if ( tstamp_db is not None ) and ( tstamp_whitelist < tstamp_db ) and ( ( tstamp_blacklist is None ) or ( tstamp_blacklist < tstamp_db) ):
            self.log(Log.C_LOG_TYPE_S, 'Marketplace DB is still up-to-date -> END')
            return
        

        # 2 Build marketplace DB file based on approved MLPro extensions
        self._build_marketplace_db( self._get_approved_extensions() )


        # 3 Outro
        self.log(Log.C_LOG_TYPE_S, 'End of updating the marketplace DB')


## -------------------------------------------------------------------------------------------------
    def run(self):

        # 1 Report new MLPro extensions  
        self.log(Log.C_LOG_TYPE_S, 'Step 1: Determination and report of new MLPro extensions...')   
        new_extensions = self._get_new_extensions()
        if len(new_extensions) == 0:
            self.log(Log.C_LOG_TYPE_I, 'No new MLPro extensions detected')
        else:
            self.log(Log.C_LOG_TYPE_I, 'New MLPro extensions detected: ', new_extensions)
            self._report_new_extensions(new_extensions)


        # 2 Update of MLPro marketplace
        self.log(Log.C_LOG_TYPE_S, 'Step 2: Update of MLPro marketplace DB')   
        self._update_db()


        
Marketplace().run()