## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.mods
## -- Module  : PS003_M003_Transporting.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-11-12  0.0.0     SY       Creation
## -- 2023-11-12  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-11-12)

This module provides a default implementation of a module of the LS-BGLP, which is a Transporting
station.
"""


from mlpro_mpps.mpps import *
from mlpro_mpps.pool.comps.PS003_C005_Silo_12L import *
from mlpro_mpps.pool.comps.PS003_C002_Hopper_9L import *
from mlpro_mpps.pool.comps.PS003_C016_BeltElevator1 import *
from mlpro_mpps.pool.comps.PS003_C010_ConveyorBelt2 import *


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TransportingStation(Module):


## -------------------------------------------------------------------------------------------------
    def _setup_module(self):
        
        silo = Silo12(p_name='SiloC')
        hopper = Hopper9(p_name='HopperC')
        act1 = BeltElevator1(p_name='ActC1')
        act2 = ConveyorBelt2(p_name='ActC2')
        
        self._add_component(p_component=silo)
        self._add_component(p_component=hopper)
        self._add_component(p_component=act1)
        self._add_component(p_component=act2)
    
