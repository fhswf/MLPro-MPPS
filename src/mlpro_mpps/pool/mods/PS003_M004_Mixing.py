## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.mods
## -- Module  : PS003_M004_Mixing.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-11-12  0.0.0     SY       Creation
## -- 2023-11-12  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-11-12)

This module provides a default implementation of a module of the LS-BGLP, which is a Mixing station.
"""


from mlpro_mpps.mpps import *
from mlpro_mpps.pool.comps.PS003_C006_MixingSilo_17L import *
from mlpro_mpps.pool.comps.PS003_C007_Hopper_8L import *
from mlpro_mpps.pool.comps.PS003_C012_VacuumPump2 import *
from mlpro_mpps.pool.comps.PS003_C015_ScrewConveyor2 import *


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MixingStation(Module):


## -------------------------------------------------------------------------------------------------
    def _setup_module(self):

        silo = MixingSilo17(p_name='SiloC')
        hopper = Hopper8(p_name='HopperC')
        act1 = VacuumPump2(p_name='ActC1')
        act2 = ScrewConveyor2(p_name='ActC2')
        
        self._add_component(p_component=silo)
        self._add_component(p_component=hopper)
        self._add_component(p_component=act1)
        self._add_component(p_component=act2)
    
