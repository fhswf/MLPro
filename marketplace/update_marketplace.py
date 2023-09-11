## -------------------------------------------------------------------------------------------------
## -- Project : MLPro Marketplace
## -- Module  : update_marketplace.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-09-08  0.0.0     DA       Creation 
## -- 2023-09-08  0.1.0     DA       Basic structure
## -- 2023-09-11  0.2.0     DA       Method Marketplace._get_extensions implemented
## -------------------------------------------------------------------------------------------------

"""
Ver. 0.2.0 (2023-09-11)

This standalone module collects meta data of all whitelisted GitHub repositories based on the
template repo /fhswf/MLPro-Extension.
"""


import sys, os.path, time
from mlpro.bf.various import Log
from github import Auth, Github




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Marketplace (Log):

    C_TYPE              = 'Marketplace'
    C_NAME              = 'MLPro'

    C_FNAME_WHITELIST   = 'whitelist'
    C_FNAME_BLACKLIST   = 'blacklist'
    C_FNAME_DB          = 'marketplace.csv'
    
    C_STATUS_APPROVED   = 'Approved'
    C_STATUS_DENIED     = 'Denied'
    C_STATUS_PENDING    = 'Pending'
    
## -------------------------------------------------------------------------------------------------
    def __init__(self, p_token, p_logging=Log.C_LOG_ALL):
        super().__init__(p_logging)

        if p_token is not None:
            self.gh = Github( auth = Auth.Token(p_token) )
        else:
            self.gh = None
            self.log(Log.C_LOG_TYPE_E, 'Please provide valid Github access token as first parameter')


## -------------------------------------------------------------------------------------------------
    def _get_extensions(self):
        """
        This method gets all repositories based on the template MLPro-Extension that are neigther
        whitelisted nor blacklisted.
        """

        # 0 Intro
        self.log(Log.C_LOG_TYPE_S, 'Start scanning for new extensions...')
        extensions                          = {}
        extensions[self.C_STATUS_APPROVED]  = {}
        extensions[self.C_STATUS_DENIED]    = {}
        extensions[self.C_STATUS_PENDING]   = {}
        whitelist                           = {}


        # 1 Get whitelisted extensions
        with open( sys.path[0] + os.sep + self.C_FNAME_WHITELIST ) as f:
            for repo in f.readlines():
                whitelist[repo] = self.C_STATUS_APPROVED

        # 2 Get blacklisted extensions
        with open( sys.path[0] + os.sep + self.C_FNAME_BLACKLIST ) as f:
            for repo in f.readlines():
                whitelist[repo] = self.C_STATUS_DENIED


        # 3 Scanning for all Github repositories based on template MLPro-Extension
        results = self.gh.search_repositories(query='mlpro-extension in:topics')        

        for repo in results:
            if repo.is_template or repo.private: continue
            try:
                latest_release = repo.get_latest_release()
            except:
                continue

            try:
                status = whitelist[repo.full_name]
            except:
                status = self.C_STATUS_PENDING

            try:
                org = extensions[status][repo.organization.login]
            except:
                org = {}
                org['name']     = repo.organization.name
                org['location'] = repo.organization.location
                org['html_url'] = repo.organization.html_url
                org['repos']    = []
                extensions[status][repo.organization.login] = org

            org['repos'].append( ( repo.full_name,
                                   repo.name,
                                   repo.description, 
                                   latest_release.tag_name,
                                   latest_release.title,
                                   latest_release.last_modified,
                                   repo.html_url,
                                   repo.homepage,
                                   repo.topics) )


        # 4 Outro
        self.log(Log.C_LOG_TYPE_S, 'End of scanning for new extensions')
        return extensions
    

## -------------------------------------------------------------------------------------------------
    def _report_pending_extensions(self, p_new_extensions : list):

        # 0 Intro
        self.log(Log.C_LOG_TYPE_S, 'Start reporting pending extensions...')


        # 1 Get MLPro's administrators
        repo_mlpro          = self.gh.get_repo( 'fhswf/mlpro')
        mlpro_team          = repo_mlpro.get_teams()
        mlpro_collaborators = repo_mlpro.get_collaborators()

        for team in mlpro_team:
            print('Team: ', team)
            members = team.get_members(role='Maintainer')
            for member in members:
                print(member)
                # print(repo_mlpro.get_collaborator_permission(member))
                print('\n')


        # 1 Reporting
        self.log(Log.C_LOG_TYPE_E, 'Not yet implemented')

        # 2 Outro
        self.log(Log.C_LOG_TYPE_S, 'End of report of pending extensions')


## -------------------------------------------------------------------------------------------------
    def _build_marketplace_db(self, p_approved_extensions : list):

        # 0 Intro
        self.log(Log.C_LOG_TYPE_S, 'Start building marketplace DB...')

        # 1 Reporting
        self.log(Log.C_LOG_TYPE_E, 'Not yet implemented')

        # 2 Outro
        self.log(Log.C_LOG_TYPE_S, 'End of building marketplace DB')


## -------------------------------------------------------------------------------------------------
    def update(self):

        # 0 Check: Github access enabled?
        if self.gh is None: return


        # 1 Determine all MLPro extensions
        extensions = self._get_extensions()


        # 2 Report pending MLPro extensions
        pending = extensions[self.C_STATUS_PENDING]
        if len( pending.keys() ) > 0:
            self._report_pending_extensions(pending)
        else:
            self.log(Log.C_LOG_TYPE_I, 'No pending MLPro extensions detected')


        # 3 Build of MLPro marketplace DB file
        self._build_marketplace_db( extensions[self.C_STATUS_APPROVED] )
        
        




try:
    token = sys.argv[1]
except:
    token = None

Marketplace( p_token=token ).update()
