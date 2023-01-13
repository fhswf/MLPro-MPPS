## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps
## -- Module  : mpps.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-12-22  0.0.0     SY/ML    Creation
## -- 2022-12-29  1.0.0     SY       Release of first version
## -- 2023-01-11  1.0.1     SY       Add p_setup on Component Class, debugging, restructuring
## -- 2023-01-13  1.0.2     SY       Add documentation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.2 (2023-01-13)

This module provides a multi-purpose environment for continuous and batch production systems with
modular setting and high-flexibility.

The users are able to develop and simulate their own production systems including setting up own
actuators, reservoirs, processes, modules/stations, and components.
We also provide the default implementations of actuators, reservoirs, components, processes and
modules, which can be found in the pool of objects.

To be noted, the usage of this simulation is not limited to machine learning tasks, but it also
can be as a testing environment for any kind of simulations, including GT tasks, evolutionary
algorithms, supervised learning, model predictive control, domain learning, transfer learning,
and many more.
"""


from mlpro.rl.models import *
from mlpro.bf.various import *
from mlpro.bf.math import *
from mlpro.bf.systems import *
from mlpro_at_basis.bf import *
import numpy as np
import random
import uuid
import math
import matplotlib.pyplot as plt





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SimActuator(Actuator, ScientificObject):
    """
    This class serves as a base class of actuators in MPPS, which provides the main attributes of an
    actuator in a simulation mode.
    An actuator is a component of a machine that is responsible for moving and controlling a mechanism
    or system.
    
    Parameters
    ----------
    p_name_short : str
        Short name of dimension
    p_base_set 
        Base set of dimension. See constants C_BASE_SET_*. Default = C_BASE_SET_R.
    p_name_long :str
        Long name of dimension (optional)
    p_name_latex : str
        LaTeX name of dimension (optional)
    p_unit : str
        Unit (optional)
    p_unit_latex : str
        LaTeX code of unit (optional)
    p_boundaries : List
        List with minimum and maximum value (optional)
    p_description : str
        Description of dimension (optional)
    p_symmetrical : bool
        Information about the symmetry of the dimension (optional, default is False)
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_kwargs : dict
        Further keyword arguments
        
    Attributes
    ----------
    C_TYPE : str
        Type of the base class. Default: 'SimActuator'.
    C_NAME : str
        Name of the actuator. Default:''.
    """

    C_TYPE = 'SimActuator'
    C_NAME = ''

    
## -------------------------------------------------------------------------------------------------
    def __init__(self, 
                 p_name_short, 
                 p_base_set=Dimension.C_BASE_SET_R, 
                 p_name_long='', 
                 p_name_latex='', 
                 p_unit='',
                 p_unit_latex='', 
                 p_boundaries:list=[], 
                 p_description='',
                 p_symmetrical:bool=False,
                 p_logging=Log.C_LOG_NOTHING,
                 **p_kwargs):
              
        Actuator.__init__(self,
                          p_name_short=p_name_short, 
                          p_base_set=p_base_set, 
                          p_name_long=p_name_long, 
                          p_name_latex=p_name_latex, 
                          p_unit=p_unit,
                          p_unit_latex=p_unit_latex, 
                          p_boundaries=p_boundaries, 
                          p_description=p_description,
                          p_symmetrical=p_symmetrical,
                          p_logging=p_logging,
                          p_kwargs=p_kwargs)
        self.value = None
        self.status = False
            

## -------------------------------------------------------------------------------------------------
    def set_value(self, p_input) -> bool:
        """
        This method provides a functionality to set a value of the actuator.

        Parameters
        ----------
        p_input : float
            input parameters to set a value.

        Returns
        -------
        bool
            if set value is successful, then True. Otherwise False.
        """
        if p_input >= self.get_boundaries()[0] and p_input <= self.get_boundaries()[1]:
            self.value = p_input
            self.status = True
            self.log(Log.C_LOG_TYPE_I, 'Actuator ' + self.get_name_short() + ' is updated.')
            return True
        else:
            self.deactivate()
            self.log(Log.C_LOG_TYPE_E, 'Actuator ' + self.get_name_short() + ' fails to be updated.')
            return False
            

## -------------------------------------------------------------------------------------------------
    def deactivate(self) -> bool:
        """
        This method provides a functionality to deactivate the actuator.

        Returns
        -------
        bool
            if deactivate is successful, then True. Otherwise False.
        """
        self.value = None
        self.status = False
        self.log(Log.C_LOG_TYPE_I, 'Actuator ' + self.get_name_short() + ' is deactivated.')
        return True
  
    
## -------------------------------------------------------------------------------------------------      
    def get_status(self) -> bool:
        """
        This method provides a functionality to get the status of the actuator.

        Returns
        -------
        bool
            Status is on/off. True means on, false means off.
        """
        return self.status
  
    
## -------------------------------------------------------------------------------------------------      
    def get_value(self):
        """
        This method provides a functionality to get the actual value of the actuator.

        Returns
        -------
        value
            The actual value of the actuator.
        """
        return self.value




    
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SimSensor(Sensor, ScientificObject):
    """
    This class serves as a base class of sensors in MPPS, which provides the main attributes of a
    sensor in a simulation mode.
    A sensor is a component of a machine that is responsible for sensing the actual state of the
    system.
    
    Parameters
    ----------
    p_name_short : str
        Short name of dimension
    p_base_set 
        Base set of dimension. See constants C_BASE_SET_*. Default = C_BASE_SET_R.
    p_name_long :str
        Long name of dimension (optional)
    p_name_latex : str
        LaTeX name of dimension (optional)
    p_unit : str
        Unit (optional)
    p_unit_latex : str
        LaTeX code of unit (optional)
    p_boundaries : List
        List with minimum and maximum value (optional)
    p_description : str
        Description of dimension (optional)
    p_symmetrical : bool
        Information about the symmetry of the dimension (optional, default is False)
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_kwargs : dict
        Further keyword arguments
        
    Attributes
    ----------
    C_TYPE : str
        Type of the base class. Default: 'SimSensor'.
    C_NAME : str
        Name of the sensor. Default:''.
    """

    C_TYPE = 'SimSensor'
    C_NAME = ''

## -------------------------------------------------------------------------------------------------
    def __init__(self, 
                 p_name_short, 
                 p_base_set=Dimension.C_BASE_SET_R, 
                 p_name_long='', 
                 p_name_latex='', 
                 p_unit='',
                 p_unit_latex='', 
                 p_boundaries:list=[], 
                 p_description='',
                 p_symmetrical:bool=False,
                 p_logging=Log.C_LOG_NOTHING,
                 **p_kwargs):
              
        Sensor.__init__(self,
                        p_name_short=p_name_short, 
                        p_base_set=p_base_set, 
                        p_name_long=p_name_long, 
                        p_name_latex=p_name_latex, 
                        p_unit=p_unit,
                        p_unit_latex=p_unit_latex, 
                        p_boundaries=p_boundaries, 
                        p_description=p_description,
                        p_symmetrical=p_symmetrical,
                        p_logging=p_logging,
                        p_kwargs=p_kwargs)
        self.value = None
        self.status = True
        self._function = self.setup_function()
            

## -------------------------------------------------------------------------------------------------
    def set_value(self, p_input) -> bool:
        """
        This method provides a functionality to set a value of the sensor to be read.

        Parameters
        ----------
        p_input : float
            input parameters to set a value.

        Returns
        -------
        bool
            if set value is successful, then True. Otherwise False.
        """
        if p_input >= self.get_boundaries()[0] and p_input <= self.get_boundaries()[1]:
            self.value = p_input
            self.status = True
            self.log(Log.C_LOG_TYPE_I, 'Sensor ' + self.get_name_short() + ' is updated.')
            return True
        else:
            self.deactivate()
            self.log(Log.C_LOG_TYPE_E, 'Sensor ' + self.get_name_short() + ' fails to be updated.')
            return False
            

## -------------------------------------------------------------------------------------------------
    def deactivate(self) -> bool:
        """
        This method provides a functionality to deactivate the sensor.

        Returns
        -------
        bool
            if deactivate is successful, then True. Otherwise False.
        """
        self.value = None
        self.status = False
        self.log(Log.C_LOG_TYPE_I, 'Sensor ' + self.get_name_short() + ' is deactivated.')
        return True
  
    
## -------------------------------------------------------------------------------------------------      
    def get_status(self) -> bool:
        """
        This method provides a functionality to get the status of the related components.

        Returns
        -------
        bool
            Status is on/off. True means on, false means off.

        """
        return self.status
  
    
## -------------------------------------------------------------------------------------------------      
    def get_value(self):
        """
        This method provides a functionality to get the actual value read by the sensor.

        Returns
        -------
        value
            The actual value of the state.
        """
        return self.value
  
    
## -------------------------------------------------------------------------------------------------      
    def setup_function(self) -> TransferFunction:
        """
        This is a custom method, in which a specific TransferFunction object is incorporated.
        The TransferFunction object describes how the sensor obtain the actual information with
        respect to input signals according to a mathematical calculation.

        Returns
        -------
        TransferFunction
            TransferFunction object.
        """
        raise NotImplementedError
  
    
## -------------------------------------------------------------------------------------------------      
    def simulate(self, p_input_signal, p_range=None) -> bool:
        """
        This method provides a functionality to simulate the sensor.

        Returns
        -------
        bool
            True means successful, otherwise False.
        """
        output = self._function.call(p_input_signal, p_range)
        self.set_value(output)
        return True



    

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SimState(Dimension, ScientificObject):
    """
    This class serves as a base class of states (extra informations) for a component in MPPS,
    which provides the main attributes of a component state in a simulation mode.
    This could be included the dynamics of the systems that can not be read by the sensors but
    important for the simulations or triggering other components.
    
    Parameters
    ----------
    p_name_short : str
        Short name of dimension
    p_base_set 
        Base set of dimension. See constants C_BASE_SET_*. Default = C_BASE_SET_R.
    p_name_long :str
        Long name of dimension (optional)
    p_name_latex : str
        LaTeX name of dimension (optional)
    p_unit : str
        Unit (optional)
    p_unit_latex : str
        LaTeX code of unit (optional)
    p_boundaries : List
        List with minimum and maximum value (optional)
    p_description : str
        Description of dimension (optional)
    p_symmetrical : bool
        Information about the symmetry of the dimension (optional, default is False)
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_kwargs : dict
        Further keyword arguments
        
    Attributes
    ----------
    C_TYPE : str
        Type of the base class. Default: 'SimState'.
    C_NAME : str
        Name of the component state. Default:''.
    """

    C_TYPE = 'SimState'
    C_NAME = ''


## -------------------------------------------------------------------------------------------------
    def __init__(self, 
                 p_name_short, 
                 p_base_set=Dimension.C_BASE_SET_R, 
                 p_name_long='', 
                 p_name_latex='', 
                 p_unit='',
                 p_unit_latex='', 
                 p_boundaries:list=[], 
                 p_description='',
                 p_symmetrical:bool=False,
                 p_logging=Log.C_LOG_NOTHING,
                 **p_kwargs):
              
        Dimension.__init__(self,
                           p_name_short=p_name_short, 
                           p_base_set=p_base_set, 
                           p_name_long=p_name_long, 
                           p_name_latex=p_name_latex, 
                           p_unit=p_unit,
                           p_unit_latex=p_unit_latex, 
                           p_boundaries=p_boundaries, 
                           p_description=p_description,
                           p_symmetrical=p_symmetrical,
                           p_logging=p_logging,
                           p_kwargs=p_kwargs)
        self.value = None
        self._function = self.setup_function()
            

## -------------------------------------------------------------------------------------------------
    def set_value(self, p_input) -> bool:
        """
        This method provides a functionality to set a value of the state.

        Parameters
        ----------
        p_input : float
            input parameters to set a value.

        Returns
        -------
        bool
            if set value is successful, then True. Otherwise False.
        """
        if p_input >= self.get_boundaries()[0] and p_input <= self.get_boundaries()[1]:
            self.value = p_input
            self.log(Log.C_LOG_TYPE_I, 'State ' + self.get_name_short() + ' is updated.')
            return True
        else:
            self.log(Log.C_LOG_TYPE_E, 'State ' + self.get_name_short() + ' fails to be updated.')
            return False
  
    
## -------------------------------------------------------------------------------------------------      
    def get_value(self):
        """
        This method provides a functionality to get the actual value of the state.

        Returns
        -------
        value
            The actual value of the state.
        """
        return self.value
  
    
## -------------------------------------------------------------------------------------------------      
    def setup_function(self) -> TransferFunction:
        """
        This is a custom method, in which a specific TransferFunction object is incorporated.
        The TransferFunction object describes what is the current state of the component with
        respect to input signals according to a mathematical calculation.

        Returns
        -------
        TransferFunction
            TransferFunction object.
        """
        raise NotImplementedError
  
    
## -------------------------------------------------------------------------------------------------      
    def simulate(self, p_input_signal, p_range=None) -> bool:
        """
        This method provides a functionality to simulate the sensor.

        Returns
        -------
        bool
            True means successful, otherwise False.
        """
        output = self._function.call(p_input_signal, p_range)
        self.set_value(output)
        return True

    



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Component(Label, EventManager, ScientificObject):
    """
    This class serves as a base class of components in MPPS, which provides the main attributes of a
    component in a simulation mode.
    A component can consist of actuators, sensors, and component states.
    
    Parameters
    ----------
    p_name_short : str
        Short name of a component
    p_id : int
        Unique id. Default: None
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_setup : bool
        Whether setup component is required. Default: True
    p_kwargs : dict
        Further keyword arguments
        
    Attributes
    ----------
    C_TYPE : str
        Type of the base class. Default: 'Component'.
    C_NAME : str
        Name of the component. Default:''.
    """

    C_TYPE = 'Component'
    C_NAME = ''


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name:str,
                 p_id:int=None,
                 p_logging=Log.C_LOG_ALL,
                 p_setup:bool=True,
                 **p_kwargs):
        
        self._kwargs = p_kwargs.copy()
        self._sensors = Set()
        self._actuators = Set()
        self._states = Set()
        
        Label.__init__(self, p_name, p_id)
        EventManager.__init__(self, p_logging=p_logging)
        if p_setup:
            self.setup_component()


## -------------------------------------------------------------------------------------------------
    def get_name_short(self) -> str:
        """
        This method provides a functionality to get the unique name of the related component.

        Returns
        -------
        str
            The unique name of the related component.

        """
        return self._name
            

## -------------------------------------------------------------------------------------------------
    def deactivate(self) -> bool:
        """
        This method provides a functionality to deactivate the component.

        Returns
        -------
        bool
            if deactivate is successful, then True. Otherwise False.
        """
        for ids in self._sensors.get_dim_ids():
            self.get_sensor(p_id=ids).deactivate()

        for ids in self._actuators.get_dim_ids():
            self.get_actuator(p_id=ids).deactivate()

        return True


## -------------------------------------------------------------------------------------------------
    def add_sensor(self, p_sensor:SimSensor):
        """
        This method provides a functionality to add a sensor to the component.

        Parameters
        ----------
        p_sensor : SimSensor
            SimSensor object to be added.
        """
        self._sensors.add_dim(p_dim=p_sensor)

    
## -------------------------------------------------------------------------------------------------
    def get_sensors(self) -> Set:
        """
        This method provides a functionality to return the internal set of sensors.

        Returns
        -------
        sensors : Set
            Set of sensors.
        """
        return self._sensors


## -------------------------------------------------------------------------------------------------
    def get_sensor(self, p_id) -> SimSensor:
        """
        This method provides a functionality to return a specific sensor according to the desired id.

        Returns
        -------
        sensor : SimSensor
            The sensor with the specific id.
        """
        return self._sensors.get_dim(p_id=p_id)


## -------------------------------------------------------------------------------------------------
    def get_sensor_value(self, p_id):
        """
        This method provides a functionality to determine the value of a sensor.

        Parameters
        ----------
        p_id
            Id of the sensor.

        Returns
        -------
            Current value of the sensor or None. None means that the sensor is deactivated or the
            value has not been set.
        """
        return self.get_sensor(p_id).get_value()


## -------------------------------------------------------------------------------------------------
    def get_sensor_status(self, p_id):
        """
        This method provides a functionality to determine the status of a sensor.

        Parameters
        ----------
        p_id
            Id of the sensor.

        Returns
        -------
            Sensor is switched on or off.
        """
        return self.get_sensor(p_id).get_status()


## -------------------------------------------------------------------------------------------------
    def set_sensor_value(self, p_id, p_value) -> bool:
        """
        This method provides a functionality to set the value of a sensor.

        Parameters
        ----------
        p_id
            Id of the sensor.
        p_value
            New sensor value.

        Returns
        -------
            Set value is succeesful or fails.
        """
        try:
            self.get_sensor(p_id).set_value(p_value)
            return True
        except:
            return False


## -------------------------------------------------------------------------------------------------
    def add_actuator(self, p_actuator:SimActuator):
        """
        This method provides a functionality to add an actuator to the component.

        Parameters
        ----------
        p_actuator : SimActuator
            SimActuator object to be added.
        """
        self._actuators.add_dim(p_dim=p_actuator)


## -------------------------------------------------------------------------------------------------
    def get_actuators(self) -> Set:
        """
        This method provides a functionality to return the internal set of actuators.

        Returns
        -------
        actuators : Set
            Set of actuators.
        """
        return self._actuators


## -------------------------------------------------------------------------------------------------
    def get_actuator(self, p_id) -> SimActuator:
        """
        This method provides a functionality to return a specific actuator according to the desired id.

        Returns
        -------
        actuator : SimActuator
            The actuator with the specific id.
        """
        return self._actuators.get_dim(p_id=p_id)


## -------------------------------------------------------------------------------------------------
    def get_actuator_value(self, p_id):
        """
        This method provides a functionality to determine the value of an actuator.

        Parameters
        ----------
        p_id
            Id of the actuator.

        Returns
        -------
            Current value of the actuator or None. None means that the actuator is deactivated or the
            value has not been set.
        """
        return self.get_actuator(p_id).get_value()


## -------------------------------------------------------------------------------------------------
    def get_actuator_status(self, p_id):
        """
        This method provides a functionality to determine the status of an actuator.

        Parameters
        ----------
        p_id
            Id of the actuator.

        Returns
        -------
            Actuator is switched on or off.
        """
        return self.get_actuator(p_id).get_status()


## -------------------------------------------------------------------------------------------------
    def set_actuator_value(self, p_id, p_value) -> bool:
        """
        This method provides a functionality to set the value of an actuator.

        Parameters
        ----------
        p_id
            Id of the actuator.
        p_value
            New actuator value.

        Returns
        -------
            Set value is succeesful or fails.
        """
        try:
            self.get_actuator(p_id).set_value(p_value)
            return True
        except:
            return False


## -------------------------------------------------------------------------------------------------
    def add_component_states(self, p_comp_states:SimState):
        """
        This method provides a functionality to add a simulatable state to the component.

        Parameters
        ----------
        p_comp_states : SimState
            SimState object to be added.
        """
        self._states.add_dim(p_dim=p_comp_states)


## -------------------------------------------------------------------------------------------------
    def get_component_states(self) -> Set:
        """
        This method provides a functionality to return the internal set of simulatable states.

        Returns
        -------
        states : Set
            Set of simulatable states.
        """
        return self._states


## -------------------------------------------------------------------------------------------------
    def get_component_state(self, p_id) -> SimState:
        """
        This method provides a functionality to return a specific simulatable state according to the
        desired id.

        Returns
        -------
        state : SimState
            The simulatable state with the specific id.
        """
        return self._states.get_dim(p_id=p_id)


## -------------------------------------------------------------------------------------------------
    def get_component_state_value(self, p_id):
        """
        This method provides a functionality to determine the value of a simulatable state.

        Parameters
        ----------
        p_id
            Id of the simulatable state.

        Returns
        -------
            Current value of the simulatable state or None.
        """
        return self.get_component_state(p_id).get_value()


## -------------------------------------------------------------------------------------------------
    def set_component_state_value(self, p_id, p_value) -> bool:
        """
        This method provides a functionality to set the value of a simulatable state.

        Parameters
        ----------
        p_id
            Id of the simulatable state.
        p_value
            New simulatable state value.

        Returns
        -------
            Set value is succeesful or fails.
        """
        try:
            self.get_component_state(p_id).set_value(p_value)
            return True
        except:
            return False


## -------------------------------------------------------------------------------------------------
    def setup_component(self):
        """
        Custom method to setup a component. An howto and documentation related to setting up
        a component will be available soon.
        """

        # self.add_actuator(...)
        # self.add_component_states(...)
        # self.add_sensor(...)

        raise NotImplementedError



        

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Module(Component):
    """
    This class serves as a base class of modules in MPPS, which provides the main attributes of a
    module in a simulation mode.
    A module can consist of components.
    
    Parameters
    ----------
    p_name_short : str
        Short name of a module
    p_id : int
        Unique id. Default: None
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_kwargs : dict
        Further keyword arguments
        
    Attributes
    ----------
    C_TYPE : str
        Type of the base class. Default: 'Module'.
    C_NAME : str
        Name of the module. Default:''.
    """

    C_TYPE = 'Module'
    C_NAME = ''


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name:str,
                 p_id:int=None,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):
        
        self._components = Set()
        
        Component.__init__(self,
                           p_name=p_name,
                           p_id=p_id,
                           p_logging=p_logging,
                           p_setup=False,
                           p_kwargs=p_kwargs)
        self.setup_module()
            

## -------------------------------------------------------------------------------------------------
    def deactivate(self) -> bool:
        """
        This method provides a functionality to deactivate the modules.

        Returns
        -------
        bool
            if deactivate is successful, then True. Otherwise False.
        """
        for ids in self._components.get_dim_ids():
            self.get_component(p_id=ids).deactivate()

        return True


## -------------------------------------------------------------------------------------------------
    def add_component(self, p_component:Component):
        """
        This method provides a functionality to add a component to the module.

        Parameters
        ----------
        p_sensor : Component
            Component object to be added.
        """
        self._components.add_dim(p_dim=p_component)

    
## -------------------------------------------------------------------------------------------------
    def get_components(self) -> Set:
        """
        This method provides a functionality to return the internal set of components.

        Returns
        -------
        components : Set
            Set of components.
        """
        return self._components


## -------------------------------------------------------------------------------------------------
    def get_component(self, p_id) -> Component:
        """
        This method provides a functionality to return a specific component according to the
        desired id.

        Returns
        -------
        component : Component
            The component with the specific id.
        """
        return self._components.get_dim(p_id=p_id)

    
## -------------------------------------------------------------------------------------------------
    def get_sensors(self) -> list:
        """
        This method provides a functionality to return the internal sets of sensors.

        Returns
        -------
        _sensors : list
            List of set of sensors.
        """
        _sensors = []

        for ids in self._components.get_dim_ids():
            _sensors.extend(self.get_component(p_id=ids).get_sensors().get_dims())
            
        return _sensors


## -------------------------------------------------------------------------------------------------
    def get_sensor(self, p_id) -> SimSensor:
        """
        This method provides a functionality to return a specific sensor according to the desired id.

        Returns
        -------
        _sensor : SimSensor
            The sensor with the specific id.
        """
        for sens in self.get_sensors():
            try:
                _sensor = sens.get_dim(p_id=p_id)
                return _sensor
            except:
                pass
        
        return False


## -------------------------------------------------------------------------------------------------
    def get_actuators(self) -> list:
        """
        This method provides a functionality to return the internal sets of actuators.

        Returns
        -------
        actuators : list
            List of set of actuators.
        """
        _actuators = []

        for ids in self._components.get_dim_ids():
            _actuators.extend(self.get_component(p_id=ids).get_actuators().get_dims())
            
        return _actuators


## -------------------------------------------------------------------------------------------------
    def get_actuator(self, p_id) -> SimActuator:
        """
        This method provides a functionality to return a specific actuator according to the desired id.

        Returns
        -------
        actuator : SimActuator
            The actuator with the specific id.
        """
        for acts in self.get_actuators():
            try:
                _actuator = acts.get_dim(p_id=p_id)
                return _actuator
            except:
                pass
        
        return False


## -------------------------------------------------------------------------------------------------
    def get_component_states(self) -> list:
        """
        This method provides a functionality to return the internal sets of simulatable states.

        Returns
        -------
        states : list
            List of set of simulatable states.
        """
        _states = []

        for ids in self._components.get_dim_ids():
            _states.extend(self.get_component(p_id=ids).get_component_states().get_dims())
            
        return _states


## -------------------------------------------------------------------------------------------------
    def get_component_state(self, p_id) -> SimState:
        """
        This method provides a functionality to return a specific simulatable state according
        to the desired id.

        Returns
        -------
        state : SimState
            The simulatable state with the specific id.
        """
        for st in self.get_actuators():
            try:
                _state = st.get_dim(p_id=p_id)
                return _state
            except:
                pass
        
        return False


## -------------------------------------------------------------------------------------------------
    def setup_module(self):
        """
        Custom method to setup a module. An howto and documentation related to setting up
        a module will be available soon.
        """

        # self.add_component(...)

        raise NotImplementedError



    

## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SimMPPS(FctSTrans, Label, ScientificObject):
    """
    This class serves as a base class of SimMPPS, which provides the main attributes of a
    MPPS in a simulation mode.
    
    [Description of SimMPPS] -- to be added
    
    Parameters
    ----------
    p_name_short : str
        Short name of a module
    p_id : int
        Unique id. Default: None
    p_logging
        Log level (see constants of class Log). Default: Log.C_LOG_ALL
    p_kwargs : dict
        Further keyword arguments
        
    Attributes
    ----------
    C_TYPE : str
        Type of the base class. Default: 'SimMPPS'.
    C_NAME : str
        Name of the SimMPPS. Default:''.
    """

    C_TYPE = 'SimMPPS'
    C_NAME = ''


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_name:str,
                 p_id:int=None,
                 p_logging=Log.C_LOG_ALL,
                 **p_kwargs):
        
        self._elements = Set()
        self._kwargs = p_kwargs.copy()
        self._actions_in_order = False
        
        Label.__init__(self, p_name, p_id)
        FctSTrans.__init__(self, p_logging)
        self._actions_in_order, self._signals = self.setup_mpps()


## -------------------------------------------------------------------------------------------------
    def add_element(self, p_elem:Component):
        """
        This method provides a functionality to add an element to the MPPS.

        Parameters
        ----------
        p_elem : Component
            Component object to be added. Module object can also be added due to inheritance from 
            Component object.
        """
        self._elements.add_dim(p_dim=p_elem)

    
## -------------------------------------------------------------------------------------------------
    def get_elements(self) -> Set:
        """
        This method provides a functionality to return the internal set of elements.

        Returns
        -------
        _elements : Set
            Set of elements.
        """
        return self._elements


## -------------------------------------------------------------------------------------------------
    def get_element(self, p_id) -> Component:
        """
        This method provides a functionality to return a specific element according to the desired id.

        Returns
        -------
        element : Component
            The element with the specific id.
        """
        return self._elements.get_dim(p_id=p_id)

    
## -------------------------------------------------------------------------------------------------
    def get_sensors(self) -> list:
        """
        This method provides a functionality to return the internal sets of sensors.

        Returns
        -------
        _sensors : list
            List of set of sensors.
        """
        _sensors = []

        for ids in self._elements.get_dim_ids():
            if isinstance(self.get_element(p_id=ids), Module):
                _sensors.extend(self.get_element(p_id=ids).get_sensors())
            else:
                _sensors.append(self.get_element(p_id=ids).get_sensors())
            
        return _sensors


## -------------------------------------------------------------------------------------------------
    def get_sensor(self, p_id) -> SimSensor:
        """
        This method provides a functionality to return a specific sensor according to the desired id.

        Returns
        -------
        _sensor : SimSensor
            The sensor with the specific id.
        """
        for sens in self.get_sensors():
            try:
                _sensor = sens.get_dim(p_id=p_id)
                return _sensor
            except:
                pass
        
        return False


## -------------------------------------------------------------------------------------------------
    def get_actuators(self) -> list:
        """
        This method provides a functionality to return the internal sets of actuators.

        Returns
        -------
        actuators : list
            List of set of actuators.
        """
        _actuators = []

        for ids in self._elements.get_dim_ids():
            if isinstance(self.get_element(p_id=ids), Module):
                _actuators.extend(self.get_element(p_id=ids).get_actuators())
            else:
                _actuators.append(self.get_element(p_id=ids).get_actuators())
            
        return _actuators


## -------------------------------------------------------------------------------------------------
    def get_actuator(self, p_id) -> SimActuator:
        """
        This method provides a functionality to return a specific actuator according to the
        desired id.

        Returns
        -------
        actuator : SimActuator
            The actuator with the specific id.
        """
        for acts in self.get_actuators():
            try:
                _actuator = acts.get_dim(p_id=p_id)
                return _actuator
            except:
                pass
        
        return False


## -------------------------------------------------------------------------------------------------
    def get_component_states(self) -> list:
        """
        This method provides a functionality to return the internal sets of simulatable states.

        Returns
        -------
        states : list
            List of set of simulatable states.
        """
        _states = []

        for ids in self._elements.get_dim_ids():
            if isinstance(self.get_element(p_id=ids), Module):
                _states.extend(self.get_element(p_id=ids).get_component_states())
            else:
                _states.append(self.get_element(p_id=ids).get_component_states())
            
        return _states


## -------------------------------------------------------------------------------------------------
    def get_component_state(self, p_id) -> SimState:
        """
        This method provides a functionality to return a specific simulatable state according to
        the desired id.

        Returns
        -------
        state : SimState
            The simulatable state with the specific id.
        """
        for st in self.get_actuators():
            try:
                _state = st.get_dim(p_id=p_id)
                return _state
            except:
                pass
        
        return False


## -------------------------------------------------------------------------------------------------
    def setup_mpps(self):
        """
        Custom method to setup a mpps. An howto and documentation related to setting up MPPS will be
        available soon.
        """
        
        # 1. Add elements
        # self.add_element(p_elem:Component)

        # 2. Setup which actions connected to which actuators

        # Option 1: The actions are sorted in the same order as self.get_actuators() 
        # _actions_in_order = True

        # Option 2: The actions are not sorted in any orders
        # _actions_in_order = False

        # 3. Setup input signals for updating sensors or component states values

        # _signals = []
        # _sensors = self.get_sensors()
        # _actuators = self.get_actuators()
        # _comp_states = self.get_component_states()
        # _signals.append([sensor/state id, input signal 1, input signal 2, ...., input signal x])
        # _signals.append([_sensors[0], _comp_states[0].get_value])
        # _signals.append([_comp_states[0], _actuators[0].get_value, _actuators[1].get_value])

        # 4. Return _actions_in_order and _signals
        # return _actions_in_order, _signals

        raise NotImplementedError



## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State, p_action: Action) -> State:
        """
        Custom method for a simulated state transition. See method simulate_reaction() for further
        details.

        Parameters
        ----------
        p_state : State
            System state.
        p_action : Action
            Action to be processed.

        Returns
        -------
        new_state : State
            Result state after state transition.
        """
        # 1. Set values to actuators
        if self._actions_in_order:
            actions = Action.get_sorted_values()
            for idx, acts in enumerate(self.get_actuators()):
                acts.set_value(actions[idx])
        else:
            raise NotImplementedError
        
        # 2. Update values of the sensors and component states
        for sig in self._signals:
            if len(sig[1:]) == 1:
                input = sig[1]()
            else:
                input = []
                for x in range(len(sig[1:])):
                    input.append(sig[x+1]())
            sig[0].simulate(input, p_range=None)

        # 3. Return the resulted states in the form of State object
        raise NotImplementedError
        

