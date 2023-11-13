## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.mods
## -- Module  : PS003_M013_Filling_SP.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-11-13  0.0.0     SY       Creation
## -- 2023-11-13  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-11-13)

This module provides a default implementation of a module of the LS-BGLP, which is a Filling station.
"""


from mlpro_mpps.mpps import *
from mlpro_mpps.pool.comps.PS003_C001_Silo_17L import *
from mlpro_mpps.pool.comps.PS003_C024_Hopper_12L import *
from mlpro_mpps.pool.comps.PS003_C032_BucketElevator_SP import *
from mlpro_mpps.pool.comps.PS003_C021_DomeValve import *


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class FillingStation_SP(Module):


## -------------------------------------------------------------------------------------------------
    def _setup_module(self):

        silo = Silo17(p_name='SiloG')
        hopper = Hopper12(p_name='HopperG')
        act1 = BucketElevatorSP(p_name='ActG1')
        act2 = DomeValve(p_name='ActG2')
        
        self._add_component(p_component=silo)
        self._add_component(p_component=hopper)
        self._add_component(p_component=act1)
        self._add_component(p_component=act2)
    
