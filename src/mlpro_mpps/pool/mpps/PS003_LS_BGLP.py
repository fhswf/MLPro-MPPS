## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.mpps
## -- Module  : PS003_LS_BGLP.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-11-09  0.0.0     SY       Creation
## -- 2023-??-??  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-??-??)

This module provides implementations of the LS-BGLP in MLPro-MPPS in three different settings,
such as:

1. LS_BGLP: based on http://dx.doi.org/10.1109/ETFA54631.2023.10275577

2. LS_BGLP_SG1: serial production system with obvious bottleneck situation

3. LS_BGLP_SG2: serial-parallel production system with obvious bottleneck situation

"""


from mlpro_mpps.mpps import *


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class LS_BGLP(SimMPPS):


## -------------------------------------------------------------------------------------------------
    def _setup_mpps(self, p_auto_adjust_names=True):
        
        # 0. Add reference
        self.C_SCIREF_TYPE          = self.C_SCIREF_TYPE_INPROCEEDINGS
        self.C_SCIREF_AUTHOR        = "Steve Yuwono, Andreas Schwung"
        self.C_SCIREF_TITLE         = "A Model-Based Deep Learning Approach for Self-Learning in Smart Production Systems"
        self.C_SCIREF_CONFERENCE    = "2023 IEEE 28th International Conference on Emerging Technologies and Factory Automation (ETFA)"
        self.C_SCIREF_YEAR          = "2023"
        self.C_SCIREF_CITY          = "Sinaia"
        self.C_SCIREF_COUNTRY       = "Romania"
        self.C_SCIREF_DOI           = "10.1109/ETFA54631.2023.10275577"

        pass


## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State, p_action: Action) -> State:
        pass


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class LS_BGLP_SG1(SimMPPS):


## -------------------------------------------------------------------------------------------------
    def _setup_mpps(self, p_auto_adjust_names=True):
        pass


## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State, p_action: Action) -> State:
        pass


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class LS_BGLP_SG2(SimMPPS):


## -------------------------------------------------------------------------------------------------
    def _setup_mpps(self, p_auto_adjust_names=True):
        pass


## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State, p_action: Action) -> State:
        pass