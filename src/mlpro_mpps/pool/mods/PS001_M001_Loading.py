## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.mods
## -- Module  : PS001_M001_Loading.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-12-29  0.0.0     SY       Creation
## -- 2022-12-29  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2022-12-29)

This module provides a default implementation of a module of the BGLP, which is a Loading station
"""


from mlpro_mpps.mpps import *
from mlpro_mpps.pool.comps.PS001_C001_Silo import *
from mlpro_mpps.pool.comps.PS001_C002_Hopper import *
from mlpro_mpps.pool.comps.PS001_C003_ConveyorBelt import *


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class LoadingStation(Module):


## -------------------------------------------------------------------------------------------------
    def setup_module(self):
        """
        Loading station consists of a silo, a hopper, and a conveyor belt.
        """
        silo = Silo(p_name='SiloA')
        hopper = Hopper(p_name='HopperA')
        belt = ConveyorBelt(p_name='BeltA')
        
        self.add_component(p_component=silo)
        self.add_component(p_component=hopper)
        self.add_component(p_component=belt)
    
