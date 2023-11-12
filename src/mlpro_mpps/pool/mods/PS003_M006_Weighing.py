## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.mods
## -- Module  : PS003_M006_Weighing.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-11-12  0.0.0     SY       Creation
## -- 2023-11-12  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-11-12)

This module provides a default implementation of a module of the LS-BGLP, which is a Weighing station.
"""


from mlpro_mpps.mpps import *
from mlpro_mpps.pool.comps.PS003_C003_Silo_15L import *
from mlpro_mpps.pool.comps.PS003_C002_Hopper_9L import *
from mlpro_mpps.pool.comps.PS003_C017_BeltElevator2 import *
from mlpro_mpps.pool.comps.PS003_C018_RotaryFeeder import *


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WeighingStation(Module):


## -------------------------------------------------------------------------------------------------
    def _setup_module(self):

        silo = Silo15(p_name='SiloF')
        hopper = Hopper9(p_name='HopperF')
        act1 = BeltElevator2(p_name='ActF1')
        act2 = RotaryFeeder(p_name='ActF2')
        
        self._add_component(p_component=silo)
        self._add_component(p_component=hopper)
        self._add_component(p_component=act1)
        self._add_component(p_component=act2)
    
