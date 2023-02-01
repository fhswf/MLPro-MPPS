## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.mods
## -- Module  : PS001_M004_Filling.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-12-30  0.0.0     SY       Creation
## -- 2022-12-30  1.0.0     SY       Release of first version
## -- 2023-02-01  1.0.1     SY       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2023-02-01)

This module provides a default implementation of a module of the BGLP, which is a Filling station
"""


from mlpro_mpps.mpps import *
from mlpro_mpps.pool.comps.PS001_C008_VacuumPump3 import *
from mlpro_mpps.pool.comps.PS001_C009_Inventory import *


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class FillingStation(Module):


## -------------------------------------------------------------------------------------------------
    def _setup_module(self):
        """
        Storing station consists of a silo, a hopper, a rotary feeder, and a vacuum pump.
        """
        inv = FinishedGoodsInventory(p_name='Inventory')
        vac = VacuumPump3(p_name='VacD')
        
        self._add_component(p_component=inv)
        self._add_component(p_component=vac)
    
