## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.mpps
## -- Module  : PS001_BGLP.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2022-12-30  0.0.0     SY       Creation
## -- 2022-12-30  1.0.0     SY       Release of first version
## -- 2023-01-11  1.0.1     SY       Debugging on setup_mpps, adjusting sensors' indices
## -- 2023-01-16  1.0.2     SY       Change order between fill-level and overflow as comp. states
## -- 2023-01-18  1.0.3     SY       Adjustment due to updated transported material functions
## -- 2023-02-01  1.0.4     SY       Refactoring
## -- 2023-02-02  1.0.5     SY       Refactoring
## -- 2023-02-27  1.0.6     SY       Refactoring
## -- 2023-11-14  1.0.7     SY       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.7 (2023-11-14)

This module provides a default implementation of the BGLP in MLPro-MPPS.
"""


from mlpro_mpps.mpps import *
from mlpro_mpps.pool.mods.PS001_M001_Loading import *
from mlpro_mpps.pool.mods.PS001_M002_Storing import *
from mlpro_mpps.pool.mods.PS001_M003_Weighing import *
from mlpro_mpps.pool.mods.PS001_M004_Filling import *


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class BGLP(SimMPPS):


## -------------------------------------------------------------------------------------------------
    def _setup_mpps(self, p_auto_adjust_names=True):
        
        # 0. Add reference
        self.C_SCIREF_TYPE    = self.C_SCIREF_TYPE_ARTICLE
        self.C_SCIREF_AUTHOR  = "Dorothea Schwung, Steve Yuwono, Andreas Schwung, Steven X. Ding"
        self.C_SCIREF_TITLE   = "Decentralized learning of energy optimal production policies using PLC-informed reinforcement learning"
        self.C_SCIREF_JOURNAL = "Computers & Chemical Engineering"
        self.C_SCIREF_YEAR    = "2021"
        self.C_SCIREF_MONTH   = "05"
        self.C_SCIREF_DAY     = "28"
        self.C_SCIREF_VOLUME  = "152"
        self.C_SCIREF_DOI     = "10.1016/j.compchemeng.2021.107382"
        
        # 1. Add elements
        loading = LoadingStation(p_name='LoadingStation')
        storing = StoringStation(p_name='StoringStation')
        weighing = WeighingStation(p_name='WeighingStation')
        filling = FillingStation(p_name='FillingStation')
        
        self._add_element(p_elem=loading)
        self._add_element(p_elem=storing)
        self._add_element(p_elem=weighing)
        self._add_element(p_elem=filling)
        
        # 2. Check duplications of the elements names
        while not self._elements_names_checker():
            if p_auto_adjust_names:
                self._elements_names_auto_adjust()
            else:
                raise NameError('There are duplications of the elements names. You can just simply set p_auto_adjust_names to True.')

        # 3. Setup which actions connected to which actuators
        self._actions_in_order = False

        # 4. Setup input signals for updating sensors or component states values
        _sens = self.get_sensors()
        _acts = self.get_actuators()
        _sts = self.get_component_states()
        
        # 4.1. Actuators-related states
        self._add_signal(
            _sts['CBTransportedMaterial'],              # p_updated_elem
            _acts['Motor'].get_value,                   # p_input_fcts[0]
            _acts['Motor'].get_status,                  # p_input_fcts[1]
            _sts['SiloLoadingFillLevel'].get_value      # p_input_fcts[2]
            )
        
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
        
        # 4.2. Buffers-related states
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
                
        # 4.3. Buffers-related sensor
        self._add_signal(
            _sens['SiloSensor1'],
            _sts['SiloLoadingFillLevel'].get_value
            )
        
        self._add_signal(
            _sens['SiloSensor2'], 
            _sts['SiloLoadingFillLevel'].get_value
            )
        
        self._add_signal(
            _sens['HopperSensor'], 
            _sts['HopperFillLevel'].get_value
            )
   
        self._add_signal(
            _sens['SiloSensor1_1'], 
            _sts['SiloFillLevel'].get_value
            )
   
        self._add_signal(
            _sens['SiloSensor2_1'], 
            _sts['SiloFillLevel'].get_value
            )
        
        self._add_signal(
            _sens['HopperSensor_1'], 
            _sts['HopperFillLevel_1'].get_value
            )
        
        self._add_signal(
            _sens['SiloSensor1_2'], 
            _sts['SiloFillLevel_1'].get_value
            )
   
        self._add_signal(
            _sens['SiloSensor2_2'], 
            _sts['SiloFillLevel_1'].get_value
            )
   
        self._add_signal(
            _sens['HopperSensor_2'], 
            _sts['HopperFillLevel_2'].get_value
            )



## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State, p_action: Action) -> State:
        
        # 1. Set values to actuators
        if self._actions_in_order:
            actions = Action.get_sorted_values()
            for idx, (_, acts) in enumerate(self.get_actuators().items()):
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