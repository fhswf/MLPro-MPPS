## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.comps
## -- Module  : PS002_C001_Tank.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-01-19  0.0.0     ML       Creation
## -- 2023-03-28  1.0.0     ML/SY    Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-03-28)

This module provides a default implementation of a component of the Liquid Station, which is a Tank.
A Tank is a component to temporary store liquid that consists of three sensors.
"""


from mlpro_mpps.mpps import *
from mlpro.bf.physics import TransferFunction
from mlpro.bf.math import *
import sys


         
            
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TankSensor1(SimSensor):
    """
    This class serves as a sensor in the upper side of the tank to indicate whether the fill-level
    of the tank closes to overflow or not.
    """

    C_TYPE = 'SimSensor'
    C_NAME = 'TankSensor1'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_BufferSensor(p_name='TF_TankSensor1',
                                p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                p_dt=0,
                                theta = 0.75*250) # 75% of the maximum tank volume - 187,5 l
        return _func




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TankSensor2(SimSensor):
    """
    This class serves as a sensor in the middle side of the tank to indicate whether the fill-level
    of the tank is in the middle.
    """

    C_TYPE = 'SimSensor'
    C_NAME = 'TankSensor3'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_BufferSensor(p_name='TF_TankSensor3',
                                p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                p_dt=0,
                                theta = 0.5*250) # 50% of the maximum tank volume - 125 l
        return _func



                    
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TankSensor3(SimSensor):
    """
    This class serves as a sensor in the lower side of the tank to indicate whether the fill-level
    of the tank closes to empty or not.
    """

    C_TYPE = 'SimSensor'
    C_NAME = 'TankSensor3'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_BufferSensor(p_name='TF_TankSensor3',
                                p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                                p_dt=0,
                                theta = 0.25*250) # 25% of the maximum tank volume - 62,5 l
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TF_BufferSensor(TransferFunction):
  
    
## -------------------------------------------------------------------------------------------------      
    def _set_function_parameters(self, p_args) -> bool:
        if self.get_type() == self.C_TRF_FUNC_CUSTOM:
            try:
                self.theta = p_args['theta']
            except:
                raise NotImplementedError('One/More parameters for this function is missing.')           
        return True
  
    
## -------------------------------------------------------------------------------------------------      
    def _custom_function(self, p_input, p_range=None):
        """
        If the fill-level is above the sensor, then the sensor returns True. Otherwise False.

        Parameters
        ----------
        p_input : float
            actual fill level of the silo.

        Returns
        -------
        bool
            True means on, False means off.
        """
        if p_input >= self.theta:
            return True
        else:
            return False


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TankFillLevel(SimState):
    """
    This class serves as a component state to calculate the actual fill-level of the Tank.
    """

    C_TYPE = 'SimState'
    C_NAME = 'TankFillLevel'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_FillLevel(p_name='TF_FillLevel',
                             p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                             p_dt=0,
                             max_vol = 250,
                             min_vol = 0)
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TF_FillLevel(TransferFunction):
  
    
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
            [1] = Volume in by pump 1
            [2] = Volume out by pump 2
            [3] = Volume out by pump 3

        Returns
        -------
        float
            Actual fill-level.
        """
        output = p_input[0] + p_input[1] - p_input[2] - p_input[3]
        
        # max value - full
        if output >= self.max_vol:
            return self.max_vol
        # min value - empty
        elif output <= self.min_vol:
            return self.min_vol
        # current fill value
        else:
            return output


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TankOverflow(SimState):
    """
    This class serves as a component state to calculate the overflow level of the Tank.
    Demage is posible by this component
    """

    C_TYPE = 'SimState'
    C_NAME = 'TankOverflow'
  
    
## -------------------------------------------------------------------------------------------------      
    def _setup_function(self) -> TransferFunction:
        _func = TF_Overflow(p_name='TF_Overflow',
                            p_type=TransferFunction.C_TRF_FUNC_CUSTOM,
                            p_dt=0,
                            max_vol = 250)
        return _func


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class TF_Overflow(TransferFunction):
  
    
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
            [1] = Volume in by pump 1
            [2] = Volume out by pump 2
            [3] = Volume out by pump 3

        Returns
        -------
        float
            Actual overflow.
        """
        cur_level = p_input[0] + p_input[1] - p_input[2] - p_input[3]
        
        # overflow
        if cur_level > self.max_vol:
            return cur_level - self.max_vol
        # no overflow
        else:
            return 0


                 
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Tank(Component):


## -------------------------------------------------------------------------------------------------
    def _setup_component(self):
        """
        A Tank consists of three sensors and two states components.
        """
        tank_sensor_1 = TankSensor1(p_name_short='TankSensor1',
                                    p_base_set=Dimension.C_BASE_SET_Z,
                                    p_boundaries=[0,1])
        
        tank_sensor_2 = TankSensor2(p_name_short='TankSensor2',
                                    p_base_set=Dimension.C_BASE_SET_Z,
                                    p_boundaries=[0,1])
        
        tank_sensor_3 = TankSensor3(p_name_short='TankSensor3',
                                    p_base_set=Dimension.C_BASE_SET_Z,
                                    p_boundaries=[0,1])
        
        tank_fill_level = TankFillLevel(p_name_short='TankFillLevel',
                                        p_base_set=Dimension.C_BASE_SET_R,
                                        p_unit='L',
                                        p_boundaries=[0,250])
        
        tank_overflow = TankOverflow(p_name_short='TankOverflow',
                                     p_base_set=Dimension.C_BASE_SET_R,
                                     p_unit='L',
                                     p_boundaries=[0,sys.maxsize])
        
        self._add_sensor(p_sensor=tank_sensor_1)
        self._add_sensor(p_sensor=tank_sensor_2)
        self._add_sensor(p_sensor=tank_sensor_3)

        self._add_component_states(p_comp_states=tank_overflow)
        self._add_component_states(p_comp_states=tank_fill_level)
    
