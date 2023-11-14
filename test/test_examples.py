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
## -------------------------------------------------------------------------------------------------


"""
Ver. 1.0.2 (2023-11-14)

Unit test for all examples available.
"""


import pytest
import runpy


howto_list = {

# Basic Functions:

    "ht_001": "mlpro_mpps.examples.howto_001_set_up_components_and_modules_in_MPPS",
    "ht_002": "mlpro_mpps.examples.howto_002_set_up_MPPS",
    "ht_003": "mlpro_mpps.examples.howto_003_run_RL_on_BGLP_using_MPPS",
    "ht_004": "mlpro_mpps.examples.howto_004_run_GT_on_BGLP_using_MPPS",
    "ht_005": "mlpro_mpps.examples.howto_005_run_RL_on_LS_using_MPPS",
    "ht_006": "mlpro_mpps.examples.howto_006_run_EA_on_LS_using_MPPS",
    "ht_007": "mlpro_mpps.examples.howto_007_run_using_LS_MPPS_data_generation",
    "ht_008": "mlpro_mpps.examples.howto_008_run_GT_on_LS_BGLP_using_MPPS",
}



@pytest.mark.parametrize("cls", list(howto_list.keys()))
def test_howto(cls):
    runpy.run_path("src/"+howto_list[cls].replace(".","/")+".py")
