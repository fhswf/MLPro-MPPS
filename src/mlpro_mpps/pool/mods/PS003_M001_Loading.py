## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.mods
## -- Module  : PS003_M001_Loading.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-11-12  0.0.0     SY       Creation
## -- 2023-11-12  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-11-12)

This module provides a default implementation of a module of the LS-BGLP, which is a Loading station.
"""


from mlpro_mpps.mpps import *
from mlpro_mpps.pool.comps.PS003_C022_SiloLoading import *
from mlpro_mpps.pool.comps.PS003_C002_Hopper_9L import *
from mlpro_mpps.pool.comps.PS003_C009_ConveyorBelt1 import *


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class LoadingStation(Module):


## -------------------------------------------------------------------------------------------------
    def _setup_module(self):
        
        silo = SiloLoading(p_name='SiloA')
        hopper = Hopper9(p_name='HopperA')
        act1 = ConveyorBelt1(p_name='ActA1')
        
        self._add_component(p_component=silo)
        self._add_component(p_component=hopper)
        self._add_component(p_component=act1)
    
