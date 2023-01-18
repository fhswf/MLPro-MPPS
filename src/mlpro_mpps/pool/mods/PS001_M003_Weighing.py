## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.mods
## -- Module  : PS001_M003_Weighing.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-12-29  0.0.0     SY       Creation
## -- 2022-12-29  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2022-12-29)

This module provides a default implementation of a module of the BGLP, which is a Weighing station
"""


from mlpro_mpps.mpps import *
from mlpro_mpps.pool.comps.PS001_C001_Silo import *
from mlpro_mpps.pool.comps.PS001_C002_Hopper import *
from mlpro_mpps.pool.comps.PS001_C005_RotaryFeeder import *
from mlpro_mpps.pool.comps.PS001_C007_VacuumPump2 import *


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WeighingStation(Module):


## -------------------------------------------------------------------------------------------------
    def setup_module(self):
        """
        Storing station consists of a silo, a hopper, a rotary feeder, and a vacuum pump.
        """
        hopper = Hopper(p_name='HopperC')
        silo = Silo(p_name='SiloC')
        vac = VacuumPump2(p_name='VacC')
        belt = RotaryFeeder(p_name='BeltC')
        
        self.add_component(p_component=silo)
        self.add_component(p_component=hopper)
        self.add_component(p_component=vac)
        self.add_component(p_component=belt)
    
