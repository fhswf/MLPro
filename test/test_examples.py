## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro
## -- Module  : test_example.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2021-10-06  1.0.0     MRD      Creation
## -- 2021-10-06  1.0.0     MRD      Release First Version
## -- 2021-12-12  1.0.1     DA       Howto 17 added
## -- 2021-12-20  1.0.2     DA       Howto 08 disabled
## -- 2022-02-28  1.0.3     SY       Howto 06, 07 of basic functions are added
## -- 2022-02-28  1.0.4     SY       Howto 07 of basic functions is disabled
## -- 2022-05-29  1.1.0     DA       Update howto list after refactoring of all howto files
## -- 2022-06-21  1.1.1     SY       Update howto 20 and 21 RL
## -- 2022-09-13  1.1.2     SY       Add howto 22 RL and 03 GT
## -- 2022-10-06  1.1.3     SY       Add howto 23
## -- 2022-10-08  1.1.4     SY       Howto bf 009 and 010 are switched to bf uui 01 and bf uui 02
## -- 2022-10-12  1.2.0     DA       Incorporation of refactored bf howto files
## -- 2022-10-13  1.2.1     DA       Removed howto bf mt 001 due of it's multiprocessing parts
## -- 2022-10-14  1.3.0     SY       Incorporation of refactored bf howto files (RL/GT)
## -- 2022-10-19  1.3.1     DA       Renamed howtos rl_att_001, rl_att_002
## -- 2022-11-07  1.3.2     DA       Reactivated howto bf_streams_011
## -- 2022-11-10  1.3.3     DA       Renamed howtos bf_streams*
## -- 2022-11-22  1.3.4     DA       Removed howto_bf_streams_051 due to delay caused by OpenML
## -- 2022-12-05  1.4.0     DA       Added howto bf_systems_001
## -- 2022-12-09  1.4.1     DA       Temporarily removed howto rl_wp_003 due to problems with pettingzoo
## -- 2022-12-14  1.5.0     DA       Added howtos bf_streams_101, bf_streams_110, bf_streams_111
## -- 2022-12-20  1.6.0     DA       Added howtos bf_streams_112, bf_streams_113
## -- 2022-12-21  1.6.2     SY       - Reactivate howto rl_wp_003
## --                                - Temporarily removed howto rl_agent_005 and rl_wp_001
## -- 2023-01-14  1.6.3     SY       Add howto related to transfer functions
## -- 2023-01-16  1.6.4     SY       Add howto related to unit converters
## -- 2023-01-27  1.6.5     MRD      Add howto related to mujoco
## -- 2023-02-02  1.6.6     DA       Renamed some rl howtos
## -- 2023-02-04  1.6.7     SY       Renaming some bf howtos
## -- 2023-02-15  1.6.8     DA       Renaming and extension of howtos for bf.ml
## -- 2023-02-23  1.6.9     DA       Renamed some rl howtos
## -- 2023-02-23  1.7.0     MRD      new Howto RL Agent 006, Howto BF System 002 and 003
## -- 2023-03-04  1.7.1     DA       Renamed some rl howtos
## -- 2023-03-08  1.7.2     SY       Add Howto RL MB 003
## -- 2023-03-10  1.7.3     SY       Renumbering module
## -- 2023-03-24  1.7.4     DA       Add Howto BF 005
## -- 2023-09-16  1.7.5     DA       Temporarily disabled: 
## --                                - howto_rl_agent_011/021/022
## --                                - howto_rl_att_003
## -- 2023-09-25  2.0.0     DA       Refactoring:
## --                                - Howtos were moved from ./src to ./test/howtos
## --                                - New auto-scan of files to be tested
## -------------------------------------------------------------------------------------------------


"""
Ver. 2.0.0 (2023-09-25)

Unit test for all examples available.
"""


import sys
import os
from mlpro.bf.various import Log
import runpy
import pytest




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class HowtoTester(Log):

    C_TYPE      = 'Howto Tester'
    C_NAME      = 'MLPro'

## -------------------------------------------------------------------------------------------------
    def test(self, p_path, p_file):
        self.log(Log.C_LOG_TYPE_S, 'Testing file', p_file)
        runpy.run_path( p_path + os.sep + p_file )

        
## -------------------------------------------------------------------------------------------------
    def get_howtos(self, p_path:str):

        file_list = []

        for (root ,sub_dirs, files) in os.walk(p_path, topdown=True):
            sub_dirs.sort()

            for sub_dir in sub_dirs:
                for (root, dirs, files) in os.walk(p_path + os.sep + sub_dir, topdown=True):
                    self.log(Log.C_LOG_TYPE_S, 'Scanning folder', root)  
                    files.sort()

                    for file in files:
                        if os.path.splitext(file)[1] == '.py':
                            file_list.append( (root, file) )
                        else:
                            self.log(Log.C_LOG_TYPE_W, 'File ignored:', file)

            break

        return file_list



sys.path.append('src')

tester = HowtoTester()
howtos = tester.get_howtos( sys.path[0] + os.sep + 'howtos' )


if __name__ != '__main__':
    @pytest.mark.parametrize("p_path,p_file", howtos)
    def test_howto(p_path, p_file):
        runpy.run_path( p_path + os.sep + p_file )

else:
    for howto in howtos:
        tester.test(howto[0], howto[1])

    tester.log(Log.C_LOG_TYPE_S, 'Howtos tested:', len(howtos))
    

        

