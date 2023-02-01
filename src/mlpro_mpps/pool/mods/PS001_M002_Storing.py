## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.mods
## -- Module  : PS001_M002_Storing.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-12-29  0.0.0     SY       Creation
## -- 2022-12-29  1.0.0     SY       Release of first version
## -- 2023-02-01  1.0.1     SY       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.1 (2023-02-01)

This module provides a default implementation of a module of the BGLP, which is a Storing station
"""


from mlpro_mpps.mpps import *
from mlpro_mpps.pool.comps.PS001_C001_Silo import *
from mlpro_mpps.pool.comps.PS001_C002_Hopper import *
from mlpro_mpps.pool.comps.PS001_C004_VibratoryConveyor import *
from mlpro_mpps.pool.comps.PS001_C006_VacuumPump1 import *


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class StoringStation(Module):


## -------------------------------------------------------------------------------------------------
    def _setup_module(self):
        """
        Storing station consists of a silo, a hopper, a vibratory conveyor, and a vacuum pump.
        """
        hopper = Hopper(p_name='HopperB')
        silo = Silo(p_name='SiloB')
        vac = VacuumPump1(p_name='VacB')
        belt = VibratoryConveyor(p_name='BeltB')
        
        self._add_component(p_component=silo)
        self._add_component(p_component=hopper)
        self._add_component(p_component=vac)
        self._add_component(p_component=belt)
    
