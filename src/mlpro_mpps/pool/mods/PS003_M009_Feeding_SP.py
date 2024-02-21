## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.mods
## -- Module  : PS003_M009_Feeding_SP.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-11-12  0.0.0     SY       Creation
## -- 2023-11-12  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-11-12)

This module provides a default implementation of a module of the LS-BGLP, which is a Feeding station.
"""


from mlpro_mpps.mpps import *
from mlpro_mpps.pool.comps.PS003_C003_Silo_15L import *
from mlpro_mpps.pool.comps.PS003_C026_Hopper_10L_SP import *
from mlpro_mpps.pool.comps.PS003_C011_VacuumPump1 import *
from mlpro_mpps.pool.comps.PS003_C014_ScrewConveyor1 import *


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class FeedingStation_SP(Module):


## -------------------------------------------------------------------------------------------------
    def _setup_module(self):
        
        silo = Silo15(p_name='SiloB')
        hopper = Hopper10SP(p_name='HopperB')
        act1 = VacuumPump1(p_name='ActB1')
        act2 = ScrewConveyor1(p_name='ActB2')
        
        self._add_component(p_component=silo)
        self._add_component(p_component=hopper)
        self._add_component(p_component=act1)
        self._add_component(p_component=act2)
    
