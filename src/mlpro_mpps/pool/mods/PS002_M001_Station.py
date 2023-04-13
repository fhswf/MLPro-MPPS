## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.mods
## -- Module  : PS002_M001_Station.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-01-19  0.0.0     ML       Creation
## -- 2023-03-28  1.0.0     ML/SY    Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-03-28)

This module provides a default implementation of single module Liquid Station
"""


from mlpro_mpps.mpps import *
from mlpro_mpps.pool.comps.PS002_C001_Tank import Tank
from mlpro_mpps.pool.comps.PS002_C002_Pump1 import Pump1
from mlpro_mpps.pool.comps.PS002_C003_Pump2 import Pump2
from mlpro_mpps.pool.comps.PS002_C004_Pump3 import Pump3


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Station(Module):


## -------------------------------------------------------------------------------------------------
    def _setup_module(self):
        """
        The station consists of a tank, one inflow pump, two outflow pumps.
        """
        tank = Tank(p_name='Tank')
        pump_in_1 = Pump1(p_name='Pump_in_1')
        pump_out_2 = Pump2(p_name='Pump_out_2')
        pump_out_3 = Pump3(p_name='Pump_out_3')
        
        self._add_component(p_component=tank)
        self._add_component(p_component=pump_in_1)
        self._add_component(p_component=pump_out_2)
        self._add_component(p_component=pump_out_3)
    
