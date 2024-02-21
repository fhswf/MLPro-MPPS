## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps
## -- Module  : test_example.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-02-27  0.0.0     SY/MRD   Creation
## -- 2023-02-27  1.0.0     SY/MRD   Release First Version
## -- 2023-03-28  1.0.1     SY/MRD   Replacing importlib with runpy, add howtos
## -- 2023-11-14  1.0.2     SY       Replacing importlib with runpy, add howtos
## -- 2024-02-21  2.0.0     SY       Shift the howtos outside from the framework
## -------------------------------------------------------------------------------------------------


"""
Ver. 2.0.0 (2024-02-21)

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
    

        

