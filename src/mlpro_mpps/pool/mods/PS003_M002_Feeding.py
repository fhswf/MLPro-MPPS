## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.mods
## -- Module  : PS003_M002_Feeding.py
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
from mlpro_mpps.pool.comps.PS003_C004_Hopper_10L import *
from mlpro_mpps.pool.comps.PS003_C011_VacuumPump1 import *
from mlpro_mpps.pool.comps.PS003_C014_ScrewConveyor1 import *


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class LoadingStation(Module):


## -------------------------------------------------------------------------------------------------
    def _setup_module(self):
        """
        Loading station consists of a silo, a hopper, and a conveyor belt.
        """
        silo = Silo15(p_name='SiloB')
        hopper = Hopper10(p_name='HopperB')
        vc = VacuumPump1(p_name='ActB1')
        belt = ScrewConveyor1(p_name='ActB2')
        
        self._add_component(p_component=silo)
        self._add_component(p_component=hopper)
        self._add_component(p_component=vc)
        self._add_component(p_component=belt)
    
