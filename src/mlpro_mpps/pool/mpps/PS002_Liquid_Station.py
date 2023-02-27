## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.mpps
## -- Module  : PS002_Liquid_Station.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-01-19  0.0.0     ML       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-01-19)

This module provides a default implementation of the Liquid Station in MLPro-MPPS.
"""


from mlpro_mpps.mpps import *
from mlpro_mpps.pool.mods.PS002_M001_Station import Station


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class Liquid_Station(SimMPPS):


## -------------------------------------------------------------------------------------------------
    def setup_mpps(self, p_auto_adjust_names=True):
        
        # 0. Add elements
        station = Station(p_name='LiquidStation')
        
        self.add_element(p_elem=station)


       # 1. Check duplications of the elements names
        while not self._elements_names_checker():
            if p_auto_adjust_names:
                self._elements_names_auto_adjust()
            else:
                raise NameError('There are duplications of the elements names. You can just simply set p_auto_adjust_names to True.')

        # 2. Setup which actions connected to which actuators
        self._actions_in_order = False

        # 3. Setup input signals for updating sensors or component states values
        _sens = self.get_sensors()
        _acts = self.get_actuators()
        _sts = self.get_component_states()
        
        # 3.1. Actuators-related states
        self._add_signal(
            _sts['PC1TransportedMaterial'],             # transport matirial pump 1
            _acts['Timer1'].get_value,                  # timer value pump 1
            _acts['Timer1'].get_status,                 # status pump 1
            )
        
        # energy pump 1 
        # timer value pump 1
        # status pump 1

        # same pump 2
        self._add_signal(
            _sts['PC1TransportedMaterial'],             # transport matirial pump 1
            _acts['Timer2'].get_value,                  # timer value pump 1
            _acts['Timer2'].get_status,                 # status pump 1
            _sts['TankFillLevel'].get_value,            # fill level tank
            )

        # same pump 3

        # tank 


        self._add_signal(
            _sts['CBPowerConsumption'],
            _acts['Motor'].get_value, 
            _acts['Motor'].get_status
            )
        
        self._add_signal(
            _sts['VC1TransportedMaterial'],
            _acts['Timer'].get_value, 
            _acts['Timer'].get_status, 
            _sts['HopperFillLevel'].get_value
            )
        
        self._add_signal(
            _sts['VC1PowerConsumption'],
            _acts['Timer'].get_value, 
            _acts['Timer'].get_status
            )
        
        self._add_signal(
            _sts['VCTransportedMaterial'],
            _acts['Switch'].get_status,
            _sts['SiloFillLevel'].get_value
            )
        
        self._add_signal(
            _sts['VCPowerConsumption'],
            _acts['Switch'].get_status
            )
        
        self._add_signal(
            _sts['VC2TransportedMaterial'],
            _acts['Timer_1'].get_value, 
            _acts['Timer_1'].get_status, 
            _sts['HopperFillLevel_1'].get_value
            )
        
        self._add_signal(
            _sts['VC2PowerConsumption'],
            _acts['Timer_1'].get_value,
            _acts['Timer_1'].get_status
            )
        
        self._add_signal(
            _sts['RFTransportedMaterial'],
            _acts['Motor_1'].get_value, 
            _acts['Motor_1'].get_status, 
            _sts['SiloFillLevel_1'].get_value
            )
        
        self._add_signal(
            _sts['RFPowerConsumption'],
            _acts['Motor_1'].get_value, 
            _acts['Motor_1'].get_status
            )
        
        self._add_signal(
            _sts['VC1TransportedMaterial_1'],
            _acts['Switch_1'].get_status,
            _sts['HopperFillLevel_2'].get_value
            )
        
        # 3.2. Buffers-related states
        self._add_signal(
            _sts['SiloLoadingOverflow'],
            _sts['SiloLoadingFillLevel'].get_value, 
            _sts['CBTransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['SiloLoadingFillLevel'],
            _sts['SiloLoadingFillLevel'].get_value, 
            _sts['CBTransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['HopperOverflow'], 
            _sts['HopperFillLevel'].get_value, 
            _sts['CBTransportedMaterial'].get_value,
            _sts['VC1TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['HopperFillLevel'],
            _sts['HopperFillLevel'].get_value,
            _sts['CBTransportedMaterial'].get_value, 
            _sts['VC1TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['SiloOverflow'], 
            _sts['SiloFillLevel'].get_value, 
            _sts['VC1TransportedMaterial'].get_value, 
            _sts['VCTransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['SiloFillLevel'], 
            _sts['SiloFillLevel'].get_value, 
            _sts['VC1TransportedMaterial'].get_value, 
            _sts['VCTransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['HopperOverflow_1'], 
            _sts['HopperFillLevel_1'].get_value, 
            _sts['VCTransportedMaterial'].get_value, 
            _sts['VC2TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['HopperFillLevel_1'], 
            _sts['HopperFillLevel_1'].get_value, 
            _sts['VCTransportedMaterial'].get_value, 
            _sts['VC2TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['SiloOverflow_1'], 
            _sts['SiloFillLevel_1'].get_value, 
            _sts['VC2TransportedMaterial'].get_value, 
            _sts['RFTransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['SiloFillLevel_1'], 
            _sts['SiloFillLevel_1'].get_value, 
            _sts['VC2TransportedMaterial'].get_value, 
            _sts['RFTransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['HopperOverflow_2'], 
            _sts['HopperFillLevel_2'].get_value, 
            _sts['RFTransportedMaterial'].get_value, 
            _sts['VC1TransportedMaterial_1'].get_value
            )
        
        self._add_signal(
            _sts['HopperFillLevel_2'], 
            _sts['HopperFillLevel_2'].get_value, 
            _sts['RFTransportedMaterial'].get_value, 
            _sts['VC1TransportedMaterial_1'].get_value
            )
        
        self._add_signal(
            _sts['InventoryLevel'], 
            _sts['InventoryLevel'].get_value, 
            _sts['VC1TransportedMaterial_1'].get_value
            )        
                
        # 3.3. Buffers-related sensor
        self._add_signal(
            _sens['SiloSensor1'],
            _sts['SiloLoadingFillLevel'].get_value
            )
        
        self._add_signal(
            _sens['SiloSensor2'], 
            _sts['SiloLoadingFillLevel'].get_value
            )
        
        self._add_signal(
            _sens['SiloSensor1_1'], 
            _sts['HopperFillLevel'].get_value
            )
   
        self._add_signal(
            _sens['SiloSensor1_2'], 
            _sts['SiloFillLevel'].get_value
            )
   
        self._add_signal(
            _sens['SiloSensor2_1'], 
            _sts['SiloFillLevel'].get_value
            )
        
        self._add_signal(
            _sens['SiloSensor1_3'], 
            _sts['HopperFillLevel_1'].get_value
            )
        
        self._add_signal(
            _sens['SiloSensor1_4'], 
            _sts['SiloFillLevel_1'].get_value
            )
   
        self._add_signal(
            _sens['SiloSensor2_2'], 
            _sts['SiloFillLevel_1'].get_value
            )
   
        self._add_signal(
            _sens['SiloSensor1_5'], 
            _sts['HopperFillLevel_2'].get_value
            )




## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State, p_action: Action) -> State:
        
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
    
