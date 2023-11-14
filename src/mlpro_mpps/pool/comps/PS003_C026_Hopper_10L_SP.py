## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.comps
## -- Module  : PS003_C026_Hopper_10L_SP.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-11-12  0.0.0     SY       Creation
## -- 2023-11-13  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-11-13)

This module provides a default implementation of a component of the BGLP, which is a 10L Mini Hopper
for serial-parallel processes.
A hopper is a component to temporary store materials that consists of a sensor.
"""


from mlpro_mpps.mpps import *
from mlpro.bf.physics import TransferFunction
from mlpro.bf.math import *
from mlpro_mpps.pool.comps.PS003_C001_Silo_17L import *
import sys


         
            
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Hopper10SP_Sensor(SimSensor):
    """
    This class serves as a sensor in the upper side of the hopper to indicate whether the fill-level
    of the hopper closes to overflow or not.
    """

    C_TYPE = 'SimSensor'
    C_NAME = 'Hopper10SP_Sensor'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_BufferSensor(p_name='TF_Hopper10SP_Sensor',
                                p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                p_dt=0,
                                theta=0.8*10.0) # 80% of the maximum fill-level
        return _func


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Hopper10SP_FillLevel(SimState):
    """
    This class serves as a component state to calculate the actual fill-level of the hopper.
    """

    C_TYPE = 'SimState'
    C_NAME = 'Hopper10SP_FillLevel'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_FillLevel_2Outputs(p_name='TF_Hopper10SP_FillLevel',
                                      p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                      p_dt=0,
                                      max_vol=10.0,
                                      min_vol=0)
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TF_FillLevel_2Outputs(TransferFunction):
  
    
## -------------------------------------------------------------------------------------------------      
    def _set_function_parameters(self, p_args) -> bool:
        if self.get_type() == self.C_TRF_FUNC_CUSTOM:
            try:
                self.max_vol = p_args['max_vol']
                self.min_vol = p_args['min_vol']
            except:
                raise NotImplementedError('One/More parameters for this function is missing.')           
        return True
  
    
## -------------------------------------------------------------------------------------------------      
    def _custom_function(self, p_input, p_range=None):
        """
        To measure the current fill-level.

        Parameters
        ----------
        p_input : list
            [0] = Actual fill-level
            [1] = Volume in
            [2] = Volume out from actuator 1
            [3] = Volume out from actuator 2

        Returns
        -------
        float
            Actual fill-level.
        """
        output = p_input[0]+p_input[1]-p_input[2]-p_input[3]
        
        if output >= self.max_vol:
            return self.max_vol
        elif output <= self.min_vol:
            return self.min_vol
        else:
            return output


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Hopper10SP_Overflow(SimState):
    """
    This class serves as a component state to calculate the overflow level of the hopper.
    """

    C_TYPE = 'SimState'
    C_NAME = 'Hopper10SP_Overflow'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_Overflow_2Outputs(p_name='TF_Hopper10SP_Overflow',
                                     p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                     p_dt=0,
                                     max_vol=10.0)
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TF_Overflow_2Outputs(TransferFunction):
  
    
## -------------------------------------------------------------------------------------------------      
    def _set_function_parameters(self, p_args) -> bool:
        if self.get_type() == self.C_TRF_FUNC_CUSTOM:
            try:
                self.max_vol = p_args['max_vol']
            except:
                raise NotImplementedError('One/More parameters for this function is missing.')           
        return True
  
    
## -------------------------------------------------------------------------------------------------      
    def _custom_function(self, p_input, p_range=None):
        """
        To measure the current overflow level.

        Parameters
        ----------
        p_input : list
            [0] = Actual fill-level
            [1] = Volume in
            [2] = Volume out from actuator 1
            [3] = Volume out from actuator 2

        Returns
        -------
        float
            Actual fill-level.
        """
        cur_level = p_input[0]+p_input[1]-p_input[2]-p_input[3]
        
        if cur_level > self.max_vol:
            return cur_level-self.max_vol
        else:
            return 0


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Hopper10SP(Component):


## -------------------------------------------------------------------------------------------------
    def _setup_component(self):
        """
        A silo consists of two sensors and two states components.
        """
        hopper_sensor = Hopper10SP_Sensor(p_name_short='Hopper10SP_Sensor1',
                                          p_base_set=Dimension.C_BASE_SET_Z,
                                          p_boundaries=[0,1])
        hopper_fill_level = Hopper10SP_FillLevel(p_name_short='Hopper10SP_FillLevel',
                                                 p_base_set=Dimension.C_BASE_SET_R,
                                                 p_unit='L',
                                                 p_boundaries=[0,10.0])
        hopper_overflow = Hopper10SP_Overflow(p_name_short='Hopper10SP_Overflow',
                                              p_base_set=Dimension.C_BASE_SET_R,
                                              p_unit='L',
                                              p_boundaries=[0,sys.maxsize])
        
        self._add_sensor(p_sensor=hopper_sensor)
        self._add_component_states(p_comp_states=hopper_overflow)
        self._add_component_states(p_comp_states=hopper_fill_level)
    
