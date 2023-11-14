## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.mods
## -- Module  : PS003_M008_BatchDosing.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-11-12  0.0.0     SY       Creation
## -- 2023-11-12  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-11-12)

This module provides a default implementation of a module of the LS-BGLP, which is a Batch Dosing
station.
"""


from mlpro_mpps.mpps import *
from mlpro_mpps.pool.comps.PS003_C008_Silo_30L import *
from mlpro_mpps.pool.comps.PS003_C023_Inventory import *
from mlpro_mpps.pool.comps.PS003_C013_VacuumPump3 import *
from mlpro_mpps.pool.comps.PS003_C025_DosingUnit import *


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class BatchDosingStation(Module):


## -------------------------------------------------------------------------------------------------
    def _setup_module(self):

        silo = Silo30(p_name='SiloH')
        inventory = FinishedGoodsInventory(p_name='Inventory')
        act1 = VacuumPump3(p_name='ActH1')
        act2 = DosingUnit(p_name='ActH2')
        
        self._add_component(p_component=silo)
        self._add_component(p_component=inventory)
        self._add_component(p_component=act1)
        self._add_component(p_component=act2)
    
