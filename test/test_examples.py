## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps
## -- Module  : test_example.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-02-27  0.0.0     SY/MRD   Creation
## -- 2023-02-27  1.0.0     SY/MRD   Release First Version
## -------------------------------------------------------------------------------------------------


"""
Ver. 1.0.0 (2023-02-27)

Unit test for all examples available.
"""


import pytest
import importlib


howto_list = {

# Basic Functions:

    "ht_001": "mlpro_mpps.examples.howto_001_set_up_components_and_modules_in_MPPS",
    "ht_002": "mlpro_mpps.examples.howto_002_set_up_MPPS",
    "ht_003": "mlpro_mpps.examples.howto_003_run_RL_on_BGLP_using_MPPS",
    "ht_004": "mlpro_mpps.examples.howto_004_run_GT_on_BGLP_using_MPPS",
}



@pytest.mark.parametrize("cls", list(howto_list.keys()))
def test_howto(cls):
    importlib.import_module(howto_list[cls])