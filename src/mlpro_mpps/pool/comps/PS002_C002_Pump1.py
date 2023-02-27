## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.comps
## -- Module  : PS002_C002_Pump1.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-01-19  0.0.0     ML       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-01-19)

This module provides a default implementation of a component of the Liquid Station, which is a Pump.
This pump is located on the input side of the Liquid Station to fill liquid into the tank.
"""


from mlpro_mpps.mpps import *
from mlpro.bf.systems import TransferFunction
from mlpro.bf.math import *
import sys


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PC1TransportedLiquid(SimState):
    """
    This class serves as a component state to calculate the transported liquid.
    """

    C_TYPE = 'SimState'
    C_NAME = 'PC1TransportedLiquid'
  
    
## -------------------------------------------------------------------------------------------------      
    def setup_function(self) -> TransferFunction:
        _func = TF_Pump(p_name='TF_Pump',
                              p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                              p_dt=0.05,
                              coef=[0.464, 0.0332])
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TF_Pump(TransferFunction):
  
    
## -------------------------------------------------------------------------------------------------      
    def set_function_parameters(self, p_args) -> bool:
        if self.get_type() == self.C_TRF_FUNC_CUSTOM:
            try:
                self.coef = p_args['coef']
            except:
                raise NotImplementedError('One/More parameters for this function is missing.')           
        return True
  
    
## -------------------------------------------------------------------------------------------------      
    def custom_function(self, p_input, p_range=None):
        """
        To measure the transported material.

        Parameters
        ----------
        p_input : list
            [0] = Turn-on duration
            [1] = Status of the actuator
        p_range : float
            period of measuring the transported material in seconds.

        Returns
        -------
        float
            The transported material.
        """
        if p_input[1]:
            if p_range is None:
                mass_transport = (2*self.coef[1])+self.coef[0]
            else:
                if p_input[0] <= p_range:
                    mass_transport = ((2*self.coef[1])+self.coef[0])*p_input[0]
                else:
                    mass_transport = ((2*self.coef[1])+self.coef[0])*p_range
            return mass_transport
        else:
            return 0


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class PC1PowerConsumption(SimState):
    """
    This class serves as a component state to calculate the power consumption of a pump.
    """

    C_TYPE = 'SimState'
    C_NAME = 'PC1PowerConsumption'
  
    
## -------------------------------------------------------------------------------------------------      
    def setup_function(self) -> TransferFunction:
        _func = TF_PowerPump(p_name='TF_PowerPump',
                                   p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                   p_dt=0.05,
                                   min_power = 0,
                                   max_power = 305,
                                   min_duration = 0.567,
                                   max_duration = 4.575)
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TF_PowerPump(TransferFunction):
  
    
## -------------------------------------------------------------------------------------------------      
    def set_function_parameters(self, p_args) -> bool:
        if self.get_type() == self.C_TRF_FUNC_CUSTOM:
            try:
                self.min_power = p_args['min_power']
                self.max_power = p_args['max_power']
                self.min_duration = p_args['min_duration']
                self.max_duration = p_args['max_duration']
            except:
                raise NotImplementedError('One/More parameters for this function is missing.')           
        return True
  
    
## -------------------------------------------------------------------------------------------------      
    def custom_function(self, p_input, p_range=None):
        """
        To measure the power consumption.

        Parameters
        ----------
        p_input : list
            [0] = Turn-on duration
            [1] = Status of the actuator
        p_range : float
            period of measuring the power consumption in seconds.

        Returns
        -------
        float
            The power consumption in kW.
        """
        if p_input[1]:
            if p_range is None:
                power = self.max_power
            else:
                if p_input[0] <= p_range:
                    power = self.max_power*p_input[0]
                else:
                    power = self.max_power*p_range
            return power/1000.0
        else:
            return 0


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Pump1(Component):


## -------------------------------------------------------------------------------------------------
    def setup_component(self):
        """
        A pump consists of an actuator and two states components.
        """
        timer = SimActuator(p_name_short='Timer1',
                            p_base_set=Dimension.C_BASE_SET_R,
                            p_unit='s',
                            p_boundaries=[0.567, 4.575])
        transported_liquid = PC1TransportedLiquid(p_name_short='PC1TransportedMaterial',
                                                      p_base_set=Dimension.C_BASE_SET_R,
                                                      p_unit='L',
                                                      p_boundaries=[0,sys.maxsize])
        power_consumption = PC1PowerConsumption(p_name_short='PC1PowerConsumption',
                                                p_base_set=Dimension.C_BASE_SET_R,
                                                p_unit='kW',
                                                p_boundaries=[0,sys.maxsize])
        
        self.add_actuator(p_actuator=timer)
        self.add_component_states(p_comp_states=transported_liquid)
        self.add_component_states(p_comp_states=power_consumption)
    
