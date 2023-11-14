## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.mods
## -- Module  : PS003_M005_Storing.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-11-12  0.0.0     SY       Creation
## -- 2023-11-13  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-11-13)

This module provides a default implementation of a module of the LS-BGLP, which is a Storing station.
"""


from mlpro_mpps.mpps import *
from mlpro_mpps.pool.comps.PS003_C001_Silo_17L import *
from mlpro_mpps.pool.comps.PS003_C004_Hopper_10L import *
from mlpro_mpps.pool.comps.PS003_C011_VacuumPump1 import *
from mlpro_mpps.pool.comps.PS003_C019_VibratoryConveyor import *


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StoringStation(Module):


## -------------------------------------------------------------------------------------------------
    def _setup_module(self):

        silo = Silo17(p_name='SiloE')
        hopper = Hopper10(p_name='HopperE')
        act1 = VacuumPump1(p_name='ActE1')
        act2 = VibratoryConveyor(p_name='ActE2')
        
        self._add_component(p_component=silo)
        self._add_component(p_component=hopper)
        self._add_component(p_component=act1)
        self._add_component(p_component=act2)
    
