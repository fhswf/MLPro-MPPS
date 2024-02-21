## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.mods
## -- Module  : PS001_M001_Loading.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-12-29  0.0.0     SY       Creation
## -- 2022-12-29  1.0.0     SY       Release of first version
## -- 2022-12-30  1.0.1     SY       Update Silo A component with SiloLoading
## -- 2023-02-01  1.0.2     SY       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.2 (2023-02-01)

This module provides a default implementation of a module of the BGLP, which is a Loading station
"""


from mlpro_mpps.mpps import *
from mlpro_mpps.pool.comps.PS001_C010_SiloLoading import *
from mlpro_mpps.pool.comps.PS001_C002_Hopper import *
from mlpro_mpps.pool.comps.PS001_C003_ConveyorBelt import *


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class LoadingStation(Module):


## -------------------------------------------------------------------------------------------------
    def _setup_module(self):
        """
        Loading station consists of a silo, a hopper, and a conveyor belt.
        """
        silo = SiloLoading(p_name='SiloA')
        hopper = Hopper(p_name='HopperA')
        belt = ConveyorBelt(p_name='BeltA')
        
        self._add_component(p_component=silo)
        self._add_component(p_component=hopper)
        self._add_component(p_component=belt)
    
