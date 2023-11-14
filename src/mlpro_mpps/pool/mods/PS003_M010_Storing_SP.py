## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.mods
## -- Module  : PS003_M010_Storing_SP.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-11-12  0.0.0     SY       Creation
## -- 2023-11-13  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-11-12)

This module provides a default implementation of a module of the LS-BGLP, which is a Storing station.
"""


from mlpro_mpps.mpps import *
from mlpro_mpps.pool.comps.PS003_C027_Silo_17L_SP import *
from mlpro_mpps.pool.comps.PS003_C026_Hopper_10L_SP import *
from mlpro_mpps.pool.comps.PS003_C029_VacuumPump1_SP import *
from mlpro_mpps.pool.comps.PS003_C019_VibratoryConveyor import *


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StoringStation_SP(Module):


## -------------------------------------------------------------------------------------------------
    def _setup_module(self):

        silo = Silo17SP(p_name='SiloE')
        hopper = Hopper10SP(p_name='HopperE')
        act1 = VacuumPump1SP(p_name='ActE1')
        act2 = VibratoryConveyor(p_name='ActE2')
        
        self._add_component(p_component=silo)
        self._add_component(p_component=hopper)
        self._add_component(p_component=act1)
        self._add_component(p_component=act2)
    
