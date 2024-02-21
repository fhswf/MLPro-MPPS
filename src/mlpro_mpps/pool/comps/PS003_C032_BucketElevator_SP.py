## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.comps
## -- Module  : PS003_C032_BucketElevator_SP.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-11-13  0.0.0     SY       Creation
## -- 2023-11-13  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-11-13)

This module provides a default implementation of a component of the LS-BGLP, which is a Bucket
Elevator for serial-parallel processes.
"""


from mlpro_mpps.mpps import *
from mlpro.bf.physics import TransferFunction
from mlpro.bf.math import *
from mlpro_mpps.pool.comps.PS003_C019_VibratoryConveyor import *
import sys


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class BuESP_TransportedMaterial(SimState):
    """
    This class serves as a component state to calculate the transported material.
    """

    C_TYPE = 'SimState'
    C_NAME = 'BuESP_TransportedMaterial'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_TransBeltSP_Binary(p_name='TF_TransBuESP_Binary',
                                      p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                      p_dt=0.05,
                                      coef=0.265)
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TF_TransBeltSP_Binary(TransferFunction):
  
    
## -------------------------------------------------------------------------------------------------      
    def _set_function_parameters(self, p_args) -> bool:
        if self.get_type() == self.C_TRF_FUNC_CUSTOM:
            try:
                self.coef = p_args['coef']
            except:
                raise NotImplementedError('One/More parameters for this function is missing.')           
        return True
  
    
## -------------------------------------------------------------------------------------------------      
    def _custom_function(self, p_input, p_range=None):
        """
        To measure the transported material.

        Parameters
        ----------
        p_input : boolean
            [0] Status of the actuator
            [1] Fill-level of the previous buffer
            [2] Transported material by the parallel actuator
        p_range : float
            period of measuring the transported material in seconds.

        Returns
        -------
        float
            The transported material.
        """
        if p_input[0]:
            if p_range is None:
                mass_transport = self.coef
            else:
                mass_transport = self.coef*p_range
        
            if mass_transport > (p_input[1]-p_input[2]):
                return p_input[1]-p_input[2]
            else:
                return mass_transport
        else:
            return 0


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class BuESP_PowerConsumption(SimState):
    """
    This class serves as a component state to calculate the power consumption.
    """

    C_TYPE = 'SimState'
    C_NAME = 'BuESP_PowerConsumption'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_PowerBelt_Binary(p_name='TF_PowerBuE_Binary',
                                    p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                    power=22.50)
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class BucketElevatorSP(Component):


## -------------------------------------------------------------------------------------------------
    def _setup_component(self):
        """
        A bucket elevator consists of an actuator and two states components.
        """
        switch = SimActuator(p_name_short='Switch',
                             p_base_set=Dimension.C_BASE_SET_Z,
                             p_unit='',
                             p_boundaries=[0,1])
        transported_material = BuESP_TransportedMaterial(p_name_short='BuESP_TransportedMaterial',
                                                         p_base_set=Dimension.C_BASE_SET_R,
                                                         p_unit='L',
                                                         p_boundaries=[0,sys.maxsize])
        power_consumption = BuESP_PowerConsumption(p_name_short='BuESP_PowerConsumption',
                                                   p_base_set=Dimension.C_BASE_SET_R,
                                                   p_unit='kW',
                                                   p_boundaries=[0,sys.maxsize])
        
        self._add_actuator(p_actuator=switch)
        self._add_component_states(p_comp_states=transported_material)
        self._add_component_states(p_comp_states=power_consumption)
    
