## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.mpps
## -- Module  : PS003_LS_BGLP.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-11-09  0.0.0     SY       Creation
## -- 2023-11-12  1.0.0     SY       Release of first version, LS-BGLP Type 1
## -- 2023-11-13  2.0.0     SY       Release of second version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-11-13)

This module provides implementations of the LS-BGLP in MLPro-MPPS in three different settings,
such as:

1. LS_BGLP: based on http://dx.doi.org/10.1109/ETFA54631.2023.10275577

2. LS_BGLP_SP: serial-parallel production system

"""


from mlpro_mpps.mpps import *
from mlpro_mpps.pool.mods.PS003_M001_Loading import *
from mlpro_mpps.pool.mods.PS003_M002_Feeding import *
from mlpro_mpps.pool.mods.PS003_M003_Transporting import *
from mlpro_mpps.pool.mods.PS003_M004_Mixing import *
from mlpro_mpps.pool.mods.PS003_M005_Storing import *
from mlpro_mpps.pool.mods.PS003_M006_Weighing import *
from mlpro_mpps.pool.mods.PS003_M007_Filling import *
from mlpro_mpps.pool.mods.PS003_M008_BatchDosing import *
from mlpro_mpps.pool.mods.PS003_M009_Feeding_SP import *
from mlpro_mpps.pool.mods.PS003_M010_Storing_SP import *
from mlpro_mpps.pool.mods.PS003_M011_BatchDosing_SP import *
from mlpro_mpps.pool.mods.PS003_M012_Mixing_SP import *
from mlpro_mpps.pool.mods.PS003_M013_Filling_SP import *


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class LS_BGLP(SimMPPS):


## -------------------------------------------------------------------------------------------------
    def _setup_mpps(self, p_auto_adjust_names=True):
        
        # 0. Add reference
        self.C_SCIREF_TYPE          = self.C_SCIREF_TYPE_INPROCEEDINGS
        self.C_SCIREF_AUTHOR        = "Steve Yuwono, Andreas Schwung"
        self.C_SCIREF_TITLE         = "A Model-Based Deep Learning Approach for Self-Learning in Smart Production Systems"
        self.C_SCIREF_CONFERENCE    = "2023 IEEE 28th International Conference on Emerging Technologies and Factory Automation (ETFA)"
        self.C_SCIREF_YEAR          = "2023"
        self.C_SCIREF_CITY          = "Sinaia"
        self.C_SCIREF_COUNTRY       = "Romania"
        self.C_SCIREF_DOI           = "10.1109/ETFA54631.2023.10275577"
        
        # 1. Add elements
        loading = LoadingStation(p_name='LoadingStation')
        feeding = FeedingStation(p_name='FeedingStation')
        transporting = TransportingStation(p_name='TransportingStation')
        mixing = MixingStation(p_name='MixingStation')
        storing = StoringStation(p_name='StoringStation')
        weighing = WeighingStation(p_name='WeighingStation')
        filling = FillingStation(p_name='FillingStation')
        dosing = BatchDosingStation(p_name='BatchDosingStation')
        
        self._add_element(p_elem=loading)
        self._add_element(p_elem=feeding)
        self._add_element(p_elem=transporting)
        self._add_element(p_elem=mixing)
        self._add_element(p_elem=storing)
        self._add_element(p_elem=weighing)
        self._add_element(p_elem=filling)
        self._add_element(p_elem=dosing)
        
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
            _sts['CB1_TransportedMaterial'],            # p_updated_elem
            _acts['Motor'].get_value,                   # p_input_fcts[0]
            _acts['Motor'].get_status,                  # p_input_fcts[1]
            _sts['SiloLoadingFillLevel'].get_value      # p_input_fcts[2]
            )
        
        self._add_signal(
            _sts['CB1_PowerConsumption'],
            _acts['Motor'].get_value, 
            _acts['Motor'].get_status
            )
        
        self._add_signal(
            _sts['VC1_TransportedMaterial'],
            _acts['Timer'].get_value, 
            _acts['Timer'].get_status, 
            _sts['Hopper9_FillLevel'].get_value
            )
        
        self._add_signal(
            _sts['VC1_PowerConsumption'],
            _acts['Timer'].get_value, 
            _acts['Timer'].get_status
            )
        
        self._add_signal(
            _sts['SC1_TransportedMaterial'],
            _acts['Motor_1'].get_value,
            _acts['Motor_1'].get_status,
            _sts['Silo15_FillLevel'].get_value
            )
        
        self._add_signal(
            _sts['SC1_PowerConsumption'],
            _acts['Motor_1'].get_value, 
            _acts['Motor_1'].get_status
            )
        
        self._add_signal(
            _sts['BE1_TransportedMaterial'],
            _acts['Motor_2'].get_value,
            _acts['Motor_2'].get_status,
            _sts['Hopper10_FillLevel'].get_value
            )
        
        self._add_signal(
            _sts['BE1_PowerConsumption'],
            _acts['Motor_2'].get_value, 
            _acts['Motor_2'].get_status
            )
        
        self._add_signal(
            _sts['CB2_TransportedMaterial'],
            _acts['Motor_3'].get_value,
            _acts['Motor_3'].get_status,
            _sts['Silo12_FillLevel'].get_value
            )
        
        self._add_signal(
            _sts['CB2_PowerConsumption'],
            _acts['Motor_3'].get_value, 
            _acts['Motor_3'].get_status
            )
        
        self._add_signal(
            _sts['VC2_TransportedMaterial'],
            _acts['Timer_1'].get_value, 
            _acts['Timer_1'].get_status, 
            _sts['Hopper9_FillLevel_1'].get_value
            )
        
        self._add_signal(
            _sts['VC2_PowerConsumption'],
            _acts['Timer_1'].get_value, 
            _acts['Timer_1'].get_status
            )
        
        self._add_signal(
            _sts['SC2_TransportedMaterial'],
            _acts['Motor_4'].get_value,
            _acts['Motor_4'].get_status,
            _sts['MixingSilo17_FillLevel'].get_value
            )
        
        self._add_signal(
            _sts['SC2_PowerConsumption'],
            _acts['Motor_4'].get_value, 
            _acts['Motor_4'].get_status
            )
        
        self._add_signal(
            _sts['VC1_TransportedMaterial_1'],
            _acts['Timer_2'].get_value, 
            _acts['Timer_2'].get_status, 
            _sts['Hopper8_FillLevel'].get_value
            )
        
        self._add_signal(
            _sts['VC1_PowerConsumption_1'],
            _acts['Timer_2'].get_value, 
            _acts['Timer_2'].get_status
            )
        
        self._add_signal(
            _sts['ViC_TransportedMaterial'],
            _acts['Switch'].get_status,
            _sts['Silo17_FillLevel'].get_value
            )
        
        self._add_signal(
            _sts['ViC_PowerConsumption'],
            _acts['Switch'].get_status
            )
        
        self._add_signal(
            _sts['BE2_TransportedMaterial'],
            _acts['Motor_5'].get_value,
            _acts['Motor_5'].get_status,
            _sts['Hopper10_FillLevel_1'].get_value
            )
        
        self._add_signal(
            _sts['BE2_PowerConsumption'],
            _acts['Motor_5'].get_value, 
            _acts['Motor_5'].get_status
            )
        
        self._add_signal(
            _sts['RF_TransportedMaterial'],
            _acts['Motor_6'].get_value,
            _acts['Motor_6'].get_status,
            _sts['Silo15_FillLevel_1'].get_value
            )
        
        self._add_signal(
            _sts['RF_PowerConsumption'],
            _acts['Motor_6'].get_value, 
            _acts['Motor_6'].get_status
            )
        
        self._add_signal(
            _sts['BuE_TransportedMaterial'],
            _acts['Switch_1'].get_status,
            _sts['Hopper9_FillLevel_2'].get_value
            )
        
        self._add_signal(
            _sts['BuE_PowerConsumption'],
            _acts['Switch_1'].get_status
            )
        
        self._add_signal(
            _sts['DV_TransportedMaterial'],
            _acts['Switch_2'].get_status,
            _sts['Silo17_FillLevel_1'].get_value
            )
        
        self._add_signal(
            _sts['DV_PowerConsumption'],
            _acts['Switch_2'].get_status
            )
        
        self._add_signal(
            _sts['VC3_TransportedMaterial'],
            _acts['Timer_3'].get_value, 
            _acts['Timer_3'].get_status, 
            _sts['Hopper12_FillLevel'].get_value
            )
        
        self._add_signal(
            _sts['VC3_PowerConsumption'],
            _acts['Timer_3'].get_value, 
            _acts['Timer_3'].get_status
            )
        
        self._add_signal(
            _sts['DU_TransportedMaterial'],
            _acts['Switch_3'].get_status,
            _sts['Silo30_FillLevel'].get_value
            )
        
        # 4.2. Buffers-related states
        self._add_signal(
            _sts['SiloLoadingOverflow'],
            _sts['SiloLoadingFillLevel'].get_value, 
            _sts['CB1_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['SiloLoadingFillLevel'],
            _sts['SiloLoadingFillLevel'].get_value, 
            _sts['CB1_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Hopper9_Overflow'], 
            _sts['Hopper9_FillLevel'].get_value, 
            _sts['CB1_TransportedMaterial'].get_value,
            _sts['VC1_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Hopper9_FillLevel'],
            _sts['Hopper9_FillLevel'].get_value,
            _sts['CB1_TransportedMaterial'].get_value, 
            _sts['VC1_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Silo15_Overflow'], 
            _sts['Silo15_FillLevel'].get_value, 
            _sts['VC1_TransportedMaterial'].get_value, 
            _sts['SC1_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Silo15_FillLevel'], 
            _sts['Silo15_FillLevel'].get_value, 
            _sts['VC1_TransportedMaterial'].get_value, 
            _sts['SC1_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Hopper10_Overflow'], 
            _sts['Hopper10_FillLevel'].get_value, 
            _sts['SC1_TransportedMaterial'].get_value,
            _sts['BE1_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Hopper10_FillLevel'],
            _sts['Hopper10_FillLevel'].get_value,
            _sts['SC1_TransportedMaterial'].get_value, 
            _sts['BE1_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Silo12_Overflow'], 
            _sts['Silo12_FillLevel'].get_value, 
            _sts['BE1_TransportedMaterial'].get_value, 
            _sts['CB2_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Silo12_FillLevel'], 
            _sts['Silo12_FillLevel'].get_value, 
            _sts['BE1_TransportedMaterial'].get_value, 
            _sts['CB2_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Hopper9_Overflow_1'], 
            _sts['Hopper9_FillLevel_1'].get_value, 
            _sts['CB2_TransportedMaterial'].get_value,
            _sts['VC2_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Hopper9_FillLevel_1'],
            _sts['Hopper9_FillLevel_1'].get_value,
            _sts['CB2_TransportedMaterial'].get_value, 
            _sts['VC2_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['MixingSilo17_Overflow'], 
            _sts['MixingSilo17_FillLevel'].get_value, 
            _sts['VC2_TransportedMaterial'].get_value, 
            _sts['SC2_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['MixingSilo17_FillLevel'], 
            _sts['MixingSilo17_FillLevel'].get_value, 
            _sts['VC2_TransportedMaterial'].get_value, 
            _sts['SC2_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Hopper8_Overflow'], 
            _sts['Hopper8_FillLevel'].get_value, 
            _sts['SC2_TransportedMaterial'].get_value,
            _sts['VC1_TransportedMaterial_1'].get_value
            )
        
        self._add_signal(
            _sts['Hopper8_FillLevel'],
            _sts['Hopper8_FillLevel'].get_value,
            _sts['SC2_TransportedMaterial'].get_value, 
            _sts['VC1_TransportedMaterial_1'].get_value
            )
        
        self._add_signal(
            _sts['Silo17_Overflow'], 
            _sts['Silo17_FillLevel'].get_value, 
            _sts['VC1_TransportedMaterial_1'].get_value, 
            _sts['ViC_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Silo17_FillLevel'], 
            _sts['Silo17_FillLevel'].get_value, 
            _sts['VC1_TransportedMaterial_1'].get_value, 
            _sts['ViC_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Hopper10_Overflow_1'], 
            _sts['Hopper10_FillLevel_1'].get_value, 
            _sts['ViC_TransportedMaterial'].get_value,
            _sts['BE2_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Hopper10_FillLevel_1'],
            _sts['Hopper10_FillLevel_1'].get_value,
            _sts['ViC_TransportedMaterial'].get_value, 
            _sts['BE2_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Silo15_Overflow_1'], 
            _sts['Silo15_FillLevel_1'].get_value, 
            _sts['BE2_TransportedMaterial'].get_value, 
            _sts['RF_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Silo15_FillLevel_1'], 
            _sts['Silo15_FillLevel_1'].get_value, 
            _sts['BE2_TransportedMaterial'].get_value, 
            _sts['RF_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Hopper9_Overflow_2'], 
            _sts['Hopper9_FillLevel_2'].get_value, 
            _sts['RF_TransportedMaterial'].get_value,
            _sts['BuE_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Hopper9_FillLevel_2'],
            _sts['Hopper9_FillLevel_2'].get_value,
            _sts['RF_TransportedMaterial'].get_value, 
            _sts['BuE_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Silo17_Overflow_1'], 
            _sts['Silo17_FillLevel_1'].get_value, 
            _sts['BuE_TransportedMaterial'].get_value, 
            _sts['DV_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Silo17_FillLevel_1'], 
            _sts['Silo17_FillLevel_1'].get_value, 
            _sts['BuE_TransportedMaterial'].get_value, 
            _sts['DV_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Hopper12_Overflow'], 
            _sts['Hopper12_FillLevel'].get_value, 
            _sts['DV_TransportedMaterial'].get_value,
            _sts['VC3_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Hopper12_FillLevel'],
            _sts['Hopper12_FillLevel'].get_value,
            _sts['DV_TransportedMaterial'].get_value, 
            _sts['VC3_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Silo30_Overflow'], 
            _sts['Silo30_FillLevel'].get_value, 
            _sts['VC3_TransportedMaterial'].get_value, 
            _sts['DU_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Silo30_FillLevel'], 
            _sts['Silo30_FillLevel'].get_value, 
            _sts['VC3_TransportedMaterial'].get_value, 
            _sts['DU_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['InventoryLevel'], 
            _sts['InventoryLevel'].get_value, 
            _sts['DU_TransportedMaterial'].get_value
            )       
                
        # 4.3. Buffers-related sensor
        self._add_signal(
            _sens['SiloLoadingSensor1'],
            _sts['SiloLoadingFillLevel'].get_value
            )
        
        self._add_signal(
            _sens['SiloLoadingSensor2'],
            _sts['SiloLoadingFillLevel'].get_value
            )
        
        self._add_signal(
            _sens['Hopper9_Sensor1'],
            _sts['Hopper9_FillLevel'].get_value
            )
        
        self._add_signal(
            _sens['Silo15_Sensor1'],
            _sts['Silo15_FillLevel'].get_value
            )
        
        self._add_signal(
            _sens['Silo15_Sensor2'],
            _sts['Silo15_FillLevel'].get_value
            )
        
        self._add_signal(
            _sens['Hopper10_Sensor1'],
            _sts['Hopper10_FillLevel'].get_value
            )
        
        self._add_signal(
            _sens['Silo12_Sensor1'],
            _sts['Silo12_FillLevel'].get_value
            )
        
        self._add_signal(
            _sens['Silo12_Sensor2'],
            _sts['Silo12_FillLevel'].get_value
            )
        
        self._add_signal(
            _sens['Hopper9_Sensor1_1'],
            _sts['Hopper9_FillLevel_1'].get_value
            )
        
        self._add_signal(
            _sens['MixingSilo17_Sensor1'],
            _sts['MixingSilo17_FillLevel'].get_value
            )
        
        self._add_signal(
            _sens['MixingSilo17_Sensor2'],
            _sts['MixingSilo17_FillLevel'].get_value
            )
        
        self._add_signal(
            _sens['Hopper8_Sensor1'],
            _sts['Hopper8_FillLevel'].get_value
            )
        
        self._add_signal(
            _sens['Silo17_Sensor1'],
            _sts['Silo17_FillLevel'].get_value
            )
        
        self._add_signal(
            _sens['Silo17_Sensor2'],
            _sts['Silo17_FillLevel'].get_value
            )
        
        self._add_signal(
            _sens['Hopper10_Sensor1_1'],
            _sts['Hopper10_FillLevel_1'].get_value
            )
        
        self._add_signal(
            _sens['Silo15_Sensor1_1'],
            _sts['Silo15_FillLevel_1'].get_value
            )
        
        self._add_signal(
            _sens['Silo15_Sensor2_1'],
            _sts['Silo15_FillLevel_1'].get_value
            )
        
        self._add_signal(
            _sens['Hopper9_Sensor1_2'],
            _sts['Hopper9_FillLevel_2'].get_value
            )
        
        self._add_signal(
            _sens['Silo17_Sensor1_1'],
            _sts['Silo17_FillLevel_1'].get_value
            )
        
        self._add_signal(
            _sens['Silo17_Sensor2_1'],
            _sts['Silo17_FillLevel_1'].get_value
            )
        
        self._add_signal(
            _sens['Hopper12_Sensor1'],
            _sts['Hopper12_FillLevel'].get_value
            )
        
        self._add_signal(
            _sens['Silo30_Sensor1'],
            _sts['Silo30_FillLevel'].get_value
            )
        
        self._add_signal(
            _sens['Silo30_Sensor2'],
            _sts['Silo30_FillLevel'].get_value
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


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class LS_BGLP_SP(SimMPPS):


## -------------------------------------------------------------------------------------------------
    def _setup_mpps(self, p_auto_adjust_names=True):
        
        # 1. Add elements
        loading = LoadingStation(p_name='LoadingStation')
        feeding = FeedingStation_SP(p_name='FeedingStation')
        transporting = TransportingStation(p_name='TransportingStation')
        mixing = MixingStation_SP(p_name='MixingStation')
        storing = StoringStation_SP(p_name='StoringStation')
        weighing = WeighingStation(p_name='WeighingStation')
        filling = FillingStation_SP(p_name='FillingStation')
        dosing = BatchDosingStation_SP(p_name='BatchDosingStation')
        
        self._add_element(p_elem=loading)
        self._add_element(p_elem=feeding)
        self._add_element(p_elem=transporting)
        self._add_element(p_elem=mixing)
        self._add_element(p_elem=storing)
        self._add_element(p_elem=weighing)
        self._add_element(p_elem=filling)
        self._add_element(p_elem=dosing)
        
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
            _sts['CB1_TransportedMaterial'],            # p_updated_elem
            _acts['Motor'].get_value,                   # p_input_fcts[0]
            _acts['Motor'].get_status,                  # p_input_fcts[1]
            _sts['SiloLoadingFillLevel'].get_value      # p_input_fcts[2]
            )
        
        self._add_signal(
            _sts['CB1_PowerConsumption'],
            _acts['Motor'].get_value, 
            _acts['Motor'].get_status
            )
        
        self._add_signal(
            _sts['VC1_TransportedMaterial'],
            _acts['Timer'].get_value, 
            _acts['Timer'].get_status, 
            _sts['Hopper9_FillLevel'].get_value
            )
        
        self._add_signal(
            _sts['VC1_PowerConsumption'],
            _acts['Timer'].get_value, 
            _acts['Timer'].get_status
            )
        
        self._add_signal(
            _sts['SC1_TransportedMaterial'],
            _acts['Motor_1'].get_value,
            _acts['Motor_1'].get_status,
            _sts['Silo15_FillLevel'].get_value
            )
        
        self._add_signal(
            _sts['SC1_PowerConsumption'],
            _acts['Motor_1'].get_value, 
            _acts['Motor_1'].get_status
            )
        
        self._add_signal(
            _sts['BE1_TransportedMaterial'],
            _acts['Motor_2'].get_value,
            _acts['Motor_2'].get_status,
            _sts['Hopper10SP_FillLevel'].get_value
            )
        
        self._add_signal(
            _sts['BE1_PowerConsumption'],
            _acts['Motor_2'].get_value, 
            _acts['Motor_2'].get_status
            )
        
        self._add_signal(
            _sts['CB2_TransportedMaterial'],
            _acts['Motor_3'].get_value,
            _acts['Motor_3'].get_status,
            _sts['Silo12_FillLevel'].get_value
            )
        
        self._add_signal(
            _sts['CB2_PowerConsumption'],
            _acts['Motor_3'].get_value, 
            _acts['Motor_3'].get_status
            )
        
        self._add_signal(
            _sts['VC2SP_TransportedMaterial'],
            _acts['Timer_1'].get_value, 
            _acts['Timer_1'].get_status, 
            _sts['Hopper10SP_FillLevel'].get_value,
            _sts['BE1_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['VC2SP_PowerConsumption'],
            _acts['Timer_1'].get_value, 
            _acts['Timer_1'].get_status
            )
        
        self._add_signal(
            _sts['SC2_TransportedMaterial'],
            _acts['Motor_4'].get_value,
            _acts['Motor_4'].get_status,
            _sts['MixingSilo17_FillLevel'].get_value
            )
        
        self._add_signal(
            _sts['SC2_PowerConsumption'],
            _acts['Motor_4'].get_value, 
            _acts['Motor_4'].get_status
            )
        
        self._add_signal(
            _sts['VC1SP_TransportedMaterial'],
            _acts['Timer_2'].get_value, 
            _acts['Timer_2'].get_status, 
            _sts['Hopper9_FillLevel_1'].get_value, 
            _sts['Hopper8_FillLevel'].get_value
            )
        
        self._add_signal(
            _sts['VC1SP_PowerConsumption'],
            _acts['Timer_2'].get_value, 
            _acts['Timer_2'].get_status
            )
        
        self._add_signal(
            _sts['ViC_TransportedMaterial'],
            _acts['Switch'].get_status,
            _sts['Silo17SP_FillLevel'].get_value
            )
        
        self._add_signal(
            _sts['ViC_PowerConsumption'],
            _acts['Switch'].get_status
            )
        
        self._add_signal(
            _sts['BE2_TransportedMaterial'],
            _acts['Motor_5'].get_value,
            _acts['Motor_5'].get_status,
            _sts['Hopper10SP_FillLevel_1'].get_value
            )
        
        self._add_signal(
            _sts['BE2_PowerConsumption'],
            _acts['Motor_5'].get_value, 
            _acts['Motor_5'].get_status
            )
        
        self._add_signal(
            _sts['RF_TransportedMaterial'],
            _acts['Motor_6'].get_value,
            _acts['Motor_6'].get_status,
            _sts['Silo15_FillLevel_1'].get_value
            )
        
        self._add_signal(
            _sts['RF_PowerConsumption'],
            _acts['Motor_6'].get_value, 
            _acts['Motor_6'].get_status
            )
        
        self._add_signal(
            _sts['BuESP_TransportedMaterial'],
            _acts['Switch_1'].get_status,
            _sts['Hopper10SP_FillLevel_1'].get_value,
            _sts['BE2_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['BuESP_PowerConsumption'],
            _acts['Switch_1'].get_status
            )
        
        self._add_signal(
            _sts['DV_TransportedMaterial'],
            _acts['Switch_2'].get_status,
            _sts['Silo17_FillLevel'].get_value
            )
        
        self._add_signal(
            _sts['DV_PowerConsumption'],
            _acts['Switch_2'].get_status
            )
        
        self._add_signal(
            _sts['VC3SP_TransportedMaterial'],
            _acts['Timer_3'].get_value, 
            _acts['Timer_3'].get_status, 
            _sts['Hopper9_FillLevel_2'].get_value, 
            _sts['Hopper12_FillLevel'].get_value
            )
        
        self._add_signal(
            _sts['VC3SP_PowerConsumption'],
            _acts['Timer_3'].get_value, 
            _acts['Timer_3'].get_status
            )
        
        self._add_signal(
            _sts['DU_TransportedMaterial'],
            _acts['Switch_3'].get_status,
            _sts['Silo30SP_FillLevel'].get_value
            )
        
        # 4.2. Buffers-related states
        self._add_signal(
            _sts['SiloLoadingOverflow'],
            _sts['SiloLoadingFillLevel'].get_value, 
            _sts['CB1_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['SiloLoadingFillLevel'],
            _sts['SiloLoadingFillLevel'].get_value, 
            _sts['CB1_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Hopper9_Overflow'], 
            _sts['Hopper9_FillLevel'].get_value, 
            _sts['CB1_TransportedMaterial'].get_value,
            _sts['VC1_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Hopper9_FillLevel'],
            _sts['Hopper9_FillLevel'].get_value,
            _sts['CB1_TransportedMaterial'].get_value, 
            _sts['VC1_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Silo15_Overflow'], 
            _sts['Silo15_FillLevel'].get_value, 
            _sts['VC1_TransportedMaterial'].get_value, 
            _sts['SC1_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Silo15_FillLevel'], 
            _sts['Silo15_FillLevel'].get_value, 
            _sts['VC1_TransportedMaterial'].get_value, 
            _sts['SC1_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Hopper10SP_Overflow'], 
            _sts['Hopper10SP_FillLevel'].get_value, 
            _sts['SC1_TransportedMaterial'].get_value,
            _sts['BE1_TransportedMaterial'].get_value,
            _sts['VC2SP_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Hopper10SP_FillLevel'],
            _sts['Hopper10SP_FillLevel'].get_value,
            _sts['SC1_TransportedMaterial'].get_value, 
            _sts['BE1_TransportedMaterial'].get_value,
            _sts['VC2SP_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Silo12_Overflow'], 
            _sts['Silo12_FillLevel'].get_value, 
            _sts['BE1_TransportedMaterial'].get_value, 
            _sts['CB2_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Silo12_FillLevel'], 
            _sts['Silo12_FillLevel'].get_value, 
            _sts['BE1_TransportedMaterial'].get_value, 
            _sts['CB2_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Hopper9_Overflow_1'], 
            _sts['Hopper9_FillLevel_1'].get_value, 
            _sts['CB2_TransportedMaterial'].get_value,
            _sts['VC2SP_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Hopper9_FillLevel_1'],
            _sts['Hopper9_FillLevel_1'].get_value,
            _sts['CB2_TransportedMaterial'].get_value, 
            _sts['VC2SP_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['MixingSilo17_Overflow'], 
            _sts['MixingSilo17_FillLevel'].get_value, 
            _sts['VC2SP_TransportedMaterial'].get_value, 
            _sts['SC2_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['MixingSilo17_FillLevel'], 
            _sts['MixingSilo17_FillLevel'].get_value, 
            _sts['VC2SP_TransportedMaterial'].get_value, 
            _sts['SC2_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Hopper8_Overflow'], 
            _sts['Hopper8_FillLevel'].get_value, 
            _sts['SC2_TransportedMaterial'].get_value,
            _sts['VC1SP_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Hopper8_FillLevel'],
            _sts['Hopper8_FillLevel'].get_value,
            _sts['SC2_TransportedMaterial'].get_value, 
            _sts['VC1SP_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Silo17SP_Overflow'], 
            _sts['Silo17SP_FillLevel'].get_value, 
            _sts['VC1SP_TransportedMaterial'].get_value, 
            _sts['ViC_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Silo17SP_FillLevel'], 
            _sts['Silo17SP_FillLevel'].get_value, 
            _sts['VC1SP_TransportedMaterial'].get_value, 
            _sts['ViC_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Hopper10SP_Overflow_1'], 
            _sts['Hopper10SP_FillLevel_1'].get_value, 
            _sts['ViC_TransportedMaterial'].get_value,
            _sts['BE2_TransportedMaterial'].get_value,
            _sts['BuESP_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Hopper10SP_FillLevel_1'],
            _sts['Hopper10SP_FillLevel_1'].get_value,
            _sts['ViC_TransportedMaterial'].get_value, 
            _sts['BE2_TransportedMaterial'].get_value,
            _sts['BuESP_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Silo15_Overflow_1'], 
            _sts['Silo15_FillLevel_1'].get_value, 
            _sts['BE2_TransportedMaterial'].get_value, 
            _sts['RF_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Silo15_FillLevel_1'], 
            _sts['Silo15_FillLevel_1'].get_value, 
            _sts['BE2_TransportedMaterial'].get_value, 
            _sts['RF_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Hopper9_Overflow_2'], 
            _sts['Hopper9_FillLevel_2'].get_value, 
            _sts['RF_TransportedMaterial'].get_value,
            _sts['BuESP_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Hopper9_FillLevel_2'],
            _sts['Hopper9_FillLevel_2'].get_value,
            _sts['RF_TransportedMaterial'].get_value, 
            _sts['BuESP_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Silo17_Overflow'], 
            _sts['Silo17_FillLevel'].get_value, 
            _sts['BuESP_TransportedMaterial'].get_value, 
            _sts['DV_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Silo17_FillLevel'], 
            _sts['Silo17_FillLevel'].get_value, 
            _sts['BuESP_TransportedMaterial'].get_value, 
            _sts['DV_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Hopper12_Overflow'], 
            _sts['Hopper12_FillLevel'].get_value, 
            _sts['DV_TransportedMaterial'].get_value,
            _sts['VC3SP_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Hopper12_FillLevel'],
            _sts['Hopper12_FillLevel'].get_value,
            _sts['DV_TransportedMaterial'].get_value, 
            _sts['VC3SP_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Silo30SP_Overflow'], 
            _sts['Silo30SP_FillLevel'].get_value, 
            _sts['VC3SP_TransportedMaterial'].get_value, 
            _sts['DU_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['Silo30SP_FillLevel'], 
            _sts['Silo30SP_FillLevel'].get_value, 
            _sts['VC3SP_TransportedMaterial'].get_value, 
            _sts['DU_TransportedMaterial'].get_value
            )
        
        self._add_signal(
            _sts['InventoryLevel'], 
            _sts['InventoryLevel'].get_value, 
            _sts['DU_TransportedMaterial'].get_value
            )       
                
        # 4.3. Buffers-related sensor
        self._add_signal(
            _sens['SiloLoadingSensor1'],
            _sts['SiloLoadingFillLevel'].get_value
            )
        
        self._add_signal(
            _sens['SiloLoadingSensor2'],
            _sts['SiloLoadingFillLevel'].get_value
            )
        
        self._add_signal(
            _sens['Hopper9_Sensor1'],
            _sts['Hopper9_FillLevel'].get_value
            )
        
        self._add_signal(
            _sens['Silo15_Sensor1'],
            _sts['Silo15_FillLevel'].get_value
            )
        
        self._add_signal(
            _sens['Silo15_Sensor2'],
            _sts['Silo15_FillLevel'].get_value
            )
        
        self._add_signal(
            _sens['Hopper10SP_Sensor1'],
            _sts['Hopper10SP_FillLevel'].get_value
            )
        
        self._add_signal(
            _sens['Silo12_Sensor1'],
            _sts['Silo12_FillLevel'].get_value
            )
        
        self._add_signal(
            _sens['Silo12_Sensor2'],
            _sts['Silo12_FillLevel'].get_value
            )
        
        self._add_signal(
            _sens['Hopper9_Sensor1_1'],
            _sts['Hopper9_FillLevel_1'].get_value
            )
        
        self._add_signal(
            _sens['MixingSilo17_Sensor1'],
            _sts['MixingSilo17_FillLevel'].get_value
            )
        
        self._add_signal(
            _sens['MixingSilo17_Sensor2'],
            _sts['MixingSilo17_FillLevel'].get_value
            )
        
        self._add_signal(
            _sens['Hopper8_Sensor1'],
            _sts['Hopper8_FillLevel'].get_value
            )
        
        self._add_signal(
            _sens['Silo17SP_Sensor1'],
            _sts['Silo17SP_FillLevel'].get_value
            )
        
        self._add_signal(
            _sens['Silo17SP_Sensor2'],
            _sts['Silo17SP_FillLevel'].get_value
            )
        
        self._add_signal(
            _sens['Hopper10SP_Sensor1_1'],
            _sts['Hopper10SP_FillLevel_1'].get_value
            )
        
        self._add_signal(
            _sens['Silo15_Sensor1_1'],
            _sts['Silo15_FillLevel_1'].get_value
            )
        
        self._add_signal(
            _sens['Silo15_Sensor2_1'],
            _sts['Silo15_FillLevel_1'].get_value
            )
        
        self._add_signal(
            _sens['Hopper9_Sensor1_2'],
            _sts['Hopper9_FillLevel_2'].get_value
            )
        
        self._add_signal(
            _sens['Silo17_Sensor1'],
            _sts['Silo17_FillLevel'].get_value
            )
        
        self._add_signal(
            _sens['Silo17_Sensor2'],
            _sts['Silo17_FillLevel'].get_value
            )
        
        self._add_signal(
            _sens['Hopper12_Sensor1'],
            _sts['Hopper12_FillLevel'].get_value
            )
        
        self._add_signal(
            _sens['Silo30SP_Sensor1'],
            _sts['Silo30SP_FillLevel'].get_value
            )
        
        self._add_signal(
            _sens['Silo30SP_Sensor2'],
            _sts['Silo30SP_FillLevel'].get_value
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