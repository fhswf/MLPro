## -------------------------------------------------------------------------------------------------
## -- Project : MLPro Marketplace
## -- Module  : update_marketplace.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-09-08  0.0.0     DA       Creation 
## -- 2023-09-08  0.1.0     DA       Basic structure
## -- 2023-09-11  0.2.0     DA       Method Marketplace._get_extensions implemented
## -- 2023-09-12  0.3.0     DA       Refactoring
## -- 2023-09-13  0.4.0     DA       Issue management implemented
## -- 2023-09-14  0.5.0     DA       Issue commenting implemented
## -- 2023-09-15  1.0.0     DA       First implementation completed
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-09-15)

This standalone module collects meta data of all whitelisted GitHub repositories based on the
template repo /fhswf/MLPro-Extension.
"""


import sys, os.path, time
import shutil
from mlpro.bf.various import Log
from github import Auth, Github, Issue




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Marketplace (Log):

    C_TYPE                      = 'Marketplace'
    C_NAME                      = 'MLPro'

    C_FNAME_WHITELIST           = 'whitelist'
    C_FNAME_BLACKLIST           = 'blacklist'

    C_FNAME_TPL_ISSUE_BODY      = 'templates' + os.sep + 'issue_body'
    C_FNAME_TPL_ISSUE_COMMENT   = 'templates' + os.sep + 'issue_comment'
    C_FNAME_TPL_RTD_EXT_OWNER   = 'templates' + os.sep + 'rtd_ext_owner.rst'
    C_FNAME_TPL_RTD_EXT_REPO    = 'templates' + os.sep + 'rtd_ext_repo.rst'

    C_PATH_RTD_ORG              = '01_extensions_org' + os.sep + 'org'
    C_PATH_RTD_USR              = '02_extensions_user' + os.sep + 'users'
    
    C_STATUS_APPROVED           = 'Approved'
    C_STATUS_DENIED             = 'Denied'
    C_STATUS_PENDING            = 'Pending'
    
## -------------------------------------------------------------------------------------------------
    def __init__(self, p_token, p_rtd_path, p_logging=Log.C_LOG_ALL):
        super().__init__(p_logging)

        if p_token is not None:
            self.gh = Github( auth = Auth.Token(p_token) )
        else:
            self.gh = None
            self.log(Log.C_LOG_TYPE_E, 'Please provide valid Github access token as first parameter')

        self._rtd_path_org      = sys.path[0] + os.sep + p_rtd_path + self.C_PATH_RTD_ORG
        self._rtd_path_usr      = sys.path[0] + os.sep + p_rtd_path + self.C_PATH_RTD_USR
        self._extension_issues  = None


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
                owner = extensions[status][repo.owner.login]
            except:
                owner = {}
                owner['name']     = repo.owner.name
                owner['bio']      = repo.owner.bio
                owner['blog']     = repo.owner.blog
                owner['location'] = repo.owner.location
                owner['html_url'] = repo.owner.html_url
                owner['type']     = repo.owner.type
                owner['repos']    = []
                extensions[status][repo.owner.login] = owner

            repo_admins = []
            for collaborator in repo.get_collaborators():
                if repo.get_collaborator_permission(collaborator.login) == 'admin': repo_admins.append(collaborator.login)

            owner['repos'].append( ( repo.full_name,
                                     repo.name,
                                     repo.description, 
                                     latest_release.tag_name,
                                     latest_release.title,
                                     latest_release.last_modified,
                                     repo.html_url,
                                     repo.homepage,
                                     repo.topics,
                                     repo_admins ) )


        # 4 Outro
        self.log(Log.C_LOG_TYPE_S, 'End of scanning for new extensions')
        return extensions
    

## -------------------------------------------------------------------------------------------------
    def _create_issue_body(self, p_extension) -> str:

        topics = ', '.join(p_extension[8])
        with open( sys.path[0] + os.sep + self.C_FNAME_TPL_ISSUE_BODY ) as f:
            body = f.read().format( vars='variables', url=p_extension[6], title=p_extension[1], topics=topics, version=p_extension[3], desc=p_extension[2] )

        return body
    

## -------------------------------------------------------------------------------------------------
    def _create_issue_comment(self, p_extension) -> str:

        if len( p_extension[9] ) == 0:
            self.log(Log.C_LOG_TYPE_W, 'No administrators determined for repo', p_extension[0] )
            return
        
        admin_list = []
        for admin in p_extension[9]:
            admin_list.append('@' + admin)

        admins = ' '.join(admin_list)
        
        with open( sys.path[0] + os.sep + self.C_FNAME_TPL_ISSUE_COMMENT ) as f:
            body = f.read().format( vars='variables', admins=admins, repo=p_extension[0], url=p_extension[6] )

        return body
   

## -------------------------------------------------------------------------------------------------
    def _create_issue(self, p_extension):

        # 0 Intro
        self.log(Log.C_LOG_TYPE_S, 'Start creating/updating issue for extension', p_extension[0])
    

        # 1 Preparation
        if self._extension_issues is None:
            self._mlpro  = self.gh.get_repo('fhswf/MLPro')

            # 1.1 Searching for open issues for pending extensions
            issues = self._mlpro.get_issues( state='open', labels=['pending-extension'] )
            self._extension_issues = {}

            for issue in issues:
                self._extension_issues[issue.title] = issue.number

            # 1.2 Get list of MLPro's administrators
            team_members        = {}
            self._mlpro_admins  = []

            for team in self._mlpro.get_teams():
                if ( team.name == 'MLPro' ) and ( team.organization.login == 'fhswf' ):
                    self._mlpro_team = team
                    break

            for team_member in self._mlpro_team.get_members():
                team_members[team_member.login] = True

            for collaborator in self._mlpro.get_collaborators():
                try:
                    if ( team_members[collaborator.login] ) and ( self._mlpro.get_collaborator_permission(collaborator) == 'admin' ):
                        self._mlpro_admins.append(collaborator.login)
                except:
                    pass

            if len( self._mlpro_admins ) == 0:
                self.log(Log.C_LOG_TYPE_E, 'No administrator found in repository fhswf/MLPro')
                return
                

        # 2 Check: does an open issue already exist for the given extension?
        issue_title = 'Pending MLPro-Extension "' + p_extension[0] + '"'
        try:
            issue_no = self._extension_issues[issue_title]
            self.log(Log.C_LOG_TYPE_I, 'Issue already exists:', issue_no)
        except:
            # 2.1 Issue needs to be created
            issue = self._mlpro.create_issue( title     = issue_title,
                                              labels    = [ 'pending-extension'],
                                              assignees = self._mlpro_admins,
                                              body      = self._create_issue_body(p_extension) )
            
            issue.create_comment(self._create_issue_comment(p_extension))
            
            self.log(Log.C_LOG_TYPE_I, 'New issue created:', issue.number)


        # 2 Outro
        self.log(Log.C_LOG_TYPE_S, 'End of creating/updating issue for extension', p_extension[0])


## -------------------------------------------------------------------------------------------------
    def _report_pending_extensions(self, p_extensions):
        """
        This method creates an issue for each pending MLPro extension and assigns it to all admin
        users of MLPro.
        """

        # 0 Intro
        self.log(Log.C_LOG_TYPE_S, 'Start reporting pending extensions...')


        # 1 Create/update issues for each pending extension
        for org in p_extensions.keys():
            for repo in p_extensions[org]['repos']:
                self._create_issue(repo)


        # 2 Outro
        self.log(Log.C_LOG_TYPE_S, 'End of report of pending extensions')


## -------------------------------------------------------------------------------------------------
    def _build_rtd_owner_repo(self, p_repo_data, p_path):

        self.log(Log.C_LOG_TYPE_I, 'Adding extension', p_repo_data[1], '...')

        if ( p_repo_data[4] == p_repo_data[3] ) or ( p_repo_data[4] == '' ):
            vertext = ''
        else:
            vertext = ' - ' + p_repo_data[4]

        topics = ', '.join( p_repo_data[8] )

        with open( sys.path[0] + os.sep + self.C_FNAME_TPL_RTD_EXT_REPO ) as f:
            repo_body = f.read().format( vars='variables', 
                                          name=p_repo_data[1], 
                                          topics=topics, 
                                          desc=p_repo_data[2], 
                                          ver=p_repo_data[3], 
                                          vertext=vertext, 
                                          modified=p_repo_data[5], 
                                          url_github=p_repo_data[6], 
                                          url=p_repo_data[7] )

        with open( p_path + os.sep + p_repo_data[1] + '.rst', 'w' ) as f:
            f.write(repo_body)


## -------------------------------------------------------------------------------------------------
    def _build_rtd_owner_structure(self, p_owner, p_data):

        # 0 Intro
        self.log(Log.C_LOG_TYPE_I, 'Start building RTD documentation for owner', p_owner)

        if p_data['type'] == 'Organization':
            path = self._rtd_path_org
        else:
            path = self._rtd_path_usr


        # 1 Create owner file and folder
        path_owner = path + os.sep + p_owner
        os.mkdir(path_owner)

        with open( sys.path[0] + os.sep + self.C_FNAME_TPL_RTD_EXT_OWNER ) as f:
            owner_body = f.read().format( vars='variables', owner=p_owner, name=p_data['name'], location=p_data['location'], bio=p_data['bio'], blog=p_data['blog'], html_url=p_data['html_url'] )

        with open( path + os.sep + p_owner + '.rst', 'w' ) as f:
            f.write(owner_body)
        

        # 2 Create rtd files for each extension
        for repo_data in p_data['repos']:
            self._build_rtd_owner_repo(repo_data, path_owner)


        # 3 Outro
        self.log(Log.C_LOG_TYPE_I, 'End of building RTD documentation for owner', p_owner)


## -------------------------------------------------------------------------------------------------
    def _build_rtd_documentation(self, p_approved_extensions : list):

        # 0 Intro
        self.log(Log.C_LOG_TYPE_S, 'Start building RTD documentation')

        
        # 1 Clear destination folders in RTD
        try:
            shutil.rmtree(self._rtd_path_org)
        except:
            pass

        try:
            shutil.rmtree(self._rtd_path_usr)
        except:
            pass

        os.mkdir(self._rtd_path_org)
        os.mkdir(self._rtd_path_usr)


        # 2 Create folder and file structure for each owner
        for owner in p_approved_extensions.keys():
            self._build_rtd_owner_structure( p_owner=owner, p_data=p_approved_extensions[owner])
            

        # 3 Outro
        self.log(Log.C_LOG_TYPE_S, 'End of building RTD documentation')


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


        # 3 Build of MLPro marketplace documentation
        self._build_rtd_documentation( extensions[self.C_STATUS_APPROVED] )
        
        




try:
    token = sys.argv[1]
except:
    token = None

Marketplace( p_token=token, p_rtd_path=sys.argv[2] ).update()
