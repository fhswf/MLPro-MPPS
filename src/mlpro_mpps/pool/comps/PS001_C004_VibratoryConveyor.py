## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.comps
## -- Module  : PS001_C004_VibratoryConveyor.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-12-29  0.0.0     SY       Creation
## -- 2022-12-29  1.0.0     SY       Release of first version
## -- 2023-01-11  1.0.1     SY       - Debugging (sys.maxsize related issue)
## --                                - Updating TF_PowerBelt_Cont
## -- 2023-01-15  1.0.2     SY       Debugging
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.2 (2023-01-15)

This module provides a default implementation of a component of the BGLP, which is a Vibratory
Conveyor.
A vibratory conveyor is located on Module 2 of the BGLP to transport materials from Silo B to
Hopper B.
"""


from mlpro_mpps.mpps import *
from mlpro_at_basis.bf import *
from mlpro.bf.math import *
import sys


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class VCTransportedMaterial(SimState):
    """
    This class serves as a component state to calculate the transported material.
    """

    C_TYPE = 'SimState'
    C_NAME = 'VCTransportedMaterial'
  
    
## -------------------------------------------------------------------------------------------------      
    def setup_function(self) -> TransferFunction:
        _func = TF_TransBelt_Binary(p_name='TF_TransBelt_Binary',
                                    p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                    p_dt=0.05,
                                    coef=0.40)
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TF_TransBelt_Binary(TransferFunction):
  
    
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
        p_input : boolean
            Status of the actuator
        p_range : float
            period of measuring the transported material in seconds.

        Returns
        -------
        float
            The transported material.
        """
        if p_input:
            if p_range is None:
                mass_transport = self.coef
            else:
                mass_transport = self.coef*p_range
            return mass_transport
        else:
            return 0


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class VCPowerConsumption(SimState):
    """
    This class serves as a component state to calculate the power consumption.
    """

    C_TYPE = 'SimState'
    C_NAME = 'VCPowerConsumption'
  
    
## -------------------------------------------------------------------------------------------------      
    def setup_function(self) -> TransferFunction:
        _func = TF_PowerBelt_Binary(p_name='TF_PowerBelt_Binary',
                                    p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                    power = 26.9)
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TF_PowerBelt_Binary(TransferFunction):
  
    
## -------------------------------------------------------------------------------------------------      
    def set_function_parameters(self, p_args) -> bool:
        if self.get_type() == self.C_TRF_FUNC_CUSTOM:
            try:
                self.power = p_args['power']
            except:
                raise NotImplementedError('One/More parameters for this function is missing.')           
        return True
  
    
## -------------------------------------------------------------------------------------------------      
    def custom_function(self, p_input, p_range=None):
        """
        To measure the power consumption.

        Parameters
        ----------
        p_input : boolean
            Status of the actuator
        p_range : float
            period of measuring the power consumption in seconds.

        Returns
        -------
        float
            The power consumption in kW.
        """
        if p_input:
            if p_range is None:
                power  = self.power
            else:
                power  = self.power*p_range
            return power/1000.0
        else:
            return 0


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class VibratoryConveyor(Component):


## -------------------------------------------------------------------------------------------------
    def setup_component(self):
        """
        A vibratory conveyor consists of an actuator and two states components.
        """
        switch = SimActuator(p_name_short='Switch',
                             p_base_set=Dimension.C_BASE_SET_Z,
                             p_unit='',
                             p_boundaries=[0,1])
        transported_material = VCTransportedMaterial(p_name_short='VCTransportedMaterial',
                                                     p_base_set=Dimension.C_BASE_SET_R,
                                                     p_unit='L',
                                                     p_boundaries=[0,sys.maxsize])
        power_consumption = VCPowerConsumption(p_name_short='VCPowerConsumption',
                                               p_base_set=Dimension.C_BASE_SET_R,
                                               p_unit='kW',
                                               p_boundaries=[0,sys.maxsize])
        
        self.add_actuator(p_actuator=switch)
        self.add_component_states(p_comp_states=transported_material)
        self.add_component_states(p_comp_states=power_consumption)
    
