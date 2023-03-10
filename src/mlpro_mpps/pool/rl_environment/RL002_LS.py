## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.pool.rl_environment
## -- Module  : RL002_LS.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-03-10  0.0.0     ML       Creation
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-03-10)

This module provides a default implementation of the BGLP in MLPro-MPPS as RL Environment.
"""


from mlpro_mpps.pool.mpps.PS002_Liquid_Station import Liquid_Station
from mlpro.bf.math import *
from mlpro.rl.models import *


                     

                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class LS4RL(Liquid_Station):


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_name:str, p_id:int=None, p_logging=Log.C_LOG_ALL, **p_kwargs):
        super().__init__(p_name=p_name, 
                         p_id=p_id, 
                         p_logging=p_logging, 
                         p_kwargs=p_kwargs
                         )
        
        try:
            self.parent = p_kwargs['p_parent']
        except:
            raise NotImplementedError('Please input the parent class of this class as p_parent')
            


## -------------------------------------------------------------------------------------------------
    def _simulate_reaction(self, p_state: State, p_action: Action) -> State:
        
        # 1. Set values to actuators
        action = []
        for agent_id in p_action.get_agent_ids():
            action_elem = p_action.get_elem(agent_id)
            for action_id in action_elem.get_dim_ids():
                action.append(action_elem.get_value(action_id))
                
        for idx, (_, acts) in enumerate(self.get_actuators().items()):
            if idx != len(self.get_actuators())-1:
                boundaries = acts.get_boundaries()
                final_action = action[idx]*(boundaries[1]-boundaries[0])+boundaries[0]
                acts.set_value(final_action)
            else:
                acts.set_value(True)
        
        # 2. Update values of the sensors and component states
        init_inventory_level = self.get_component_states()['InventoryLevel'].get_value()
        for sig in self._signals:
            if len(sig[1:]) == 1:
                input = sig[1]()
            else:
                input = []
                for x in range(len(sig[1:])):
                    input.append(sig[x+1]())
            sig[0].simulate(input, p_range=self.parent.t_set)

        # 3. Return the resulted states in the form of State object
        self.parent._state = self.parent.get_states()
        self.parent._state.set_success(False)
        self.parent._state.set_broken(False)
        
        self.parent.t += self.parent.t_set
        current_volume = self.get_component_states()['InventoryLevel'].get_value()
        overlfow = sum(self.parent.get_overflow())
        power = sum(self.parent.get_power())
        self.parent.current_demand = self.parent.get_demand(init_inventory_level, current_volume)
        self.parent.prod_reached += (current_volume-init_inventory_level)
        
        self.parent.data_storing.memorize("time",str(self.parent.data_frame), self.parent.t)
        self.parent.data_storing.memorize("overflow",str(self.parent.data_frame), overlfow/self.parent.t_set)
        self.parent.data_storing.memorize("power",str(self.parent.data_frame), power/self.parent.t_set)
        self.parent.data_storing.memorize("demand",str(self.parent.data_frame), self.parent.current_demand/self.parent.t_set)
        
        return self.parent._state


                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class LS_RLEnv(Environment):

    C_TYPE = 'Environment'
    C_NAME = 'MPPS-based Liquid Station - RL Environment'
    C_CYCLE_LIMIT = 0  # Recommended cycle limit for training episodes


## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_reward_type=Reward.C_TYPE_EVERY_AGENT,
                 p_logging=Log.C_LOG_ALL,
                 t_set=10.0,
                 demand=0.1,
                 lr_margin=1.0,
                 lr_demand=4.0,
                 lr_power=0.0010, 
                 margin_p=[0.2,0.8,4],
                 prod_target=10000,
                 prod_scenario='continuous',
                 cycle_limit=0):

        self.num_envs = 5                                                 # Number of internal sub-environments
        self.reward_type = p_reward_type
        
        super().__init__(p_mode = Mode.C_MODE_SIM, 
                         p_latency = None, 
                         p_fct_strans = LS4RL(p_name='BGLP_RL',
                                              p_logging=p_logging,
                                              p_parent=self
                                              ), 
                         p_fct_reward = None, 
                         p_fct_success = None, 
                         p_fct_broken = None, 
                         p_visualize = False, 
                         p_logging = p_logging)
        
        self.C_SCIREF_TYPE    = self.C_SCIREF_TYPE_ARTICLE
        self.C_SCIREF_AUTHOR  = "Marlon Löppenberg, Andreas Schwung"
        self.C_SCIREF_TITLE   = "Self Optimisation and Automatic Code Generation by Evolutionary Algorithms in PLC based Controlling Processes"
        self.C_SCIREF_JOURNAL = ""
        self.C_SCIREF_YEAR    = ""
        self.C_SCIREF_MONTH   = ""
        self.C_SCIREF_DAY     = ""
        self.C_SCIREF_VOLUME  = ""
        self.C_SCIREF_DOI     = ""
        
        self.C_CYCLE_LIMIT = cycle_limit
        self.t = 0
        self.t_set = t_set
        self.demand = demand
        self.lr_margin = lr_margin
        self.lr_demand = lr_demand
        self.lr_power = lr_power
        self.prod_target = prod_target
        self.prod_scenario = prod_scenario
        self.margin_p = margin_p
        
        self.data_lists = ["time","overflow","power","demand"]
        self.data_storing = DataStoring(self.data_lists)
        self.data_frame = None
        
        # tank overflow 
        self.set_overflow = ['TankOverflow',
                             ]
        
        # tank level
        self.set_fill_levels = ['TankFillLevel',
                                ]
        
        # pump liquid transport
        self.set_transport_liquid = ['PC1TransportedMaterial',
                                     'PC2TransportedMaterial',
                                     'PC3TransportedMaterial'
                                     ]
        
        # pump energy consumption
        self.set_power = ['PC1PowerConsumption',
                          'PC2PowerConsumption',
                          'PC3PowerConsumption',
                          ]
        
        self.reset()


## -------------------------------------------------------------------------------------------------
    @staticmethod
    def setup_spaces():
        state_space = ESpace()
        action_space = ESpace()

        # define state space
        state_space.add_dim(Dimension('R-1 LvlTank', 'R', 'Res-1 Level of Tank', '', '', '', [0, 250]))
        
        # define action space
        action_space.add_dim(Dimension('A-1 Act', 'R', 'Act-1 Belt Conveyor A', '', '', '', [0, 1]))
        action_space.add_dim(Dimension('A-2 Act', 'R', 'Act-2 Vacuum Pump B', '', '', '', [0, 1]))
        action_space.add_dim(Dimension('A-3 Act', 'Z', 'Act-3 Vibratory Conveyor B', '', '', '', [0, 1]))

        return state_space, action_space


## -------------------------------------------------------------------------------------------------
    def get_states(self) -> State:
        state = State(self._state_space)
        ids = state.get_dim_ids()
        
        for x in range(len(ids)):
            fill_level = self._fct_strans.get_component_states()[self.set_fill_levels[x]].get_value()
            boundaries = self._fct_strans.get_component_states()[self.set_fill_levels[x]].get_boundaries()
            norm_fill_level = (fill_level-boundaries[0])/(boundaries[1]-boundaries[0])
            state.set_value(ids[x], norm_fill_level) 
        return state


## -------------------------------------------------------------------------------------------------
    def get_margin(self) -> list:
        margin = []
        
        for x in range(len(self.set_fill_levels)):
            fill_level = self._fct_strans.get_component_states()[self.set_fill_levels[x]].get_value()
            boundaries = self._fct_strans.get_component_states()[self.set_fill_levels[x]].get_boundaries()
            norm_fill_level = (fill_level-boundaries[0])/(boundaries[1]-boundaries[0])
            if norm_fill_level < self.margin_p[0]:
                m = (0-self.margin_p[2])/(self.margin_p[0])*(norm_fill_level-self.margin_p[0])*self.t_set
            elif norm_fill_level > self.margin_p[1]:
                m = self.margin_p[2]/(1-self.margin_p[1])*(norm_fill_level-self.margin_p[1])*self.t_set
            else:
                m = 0.0
            margin.append(m)
        return margin


## -------------------------------------------------------------------------------------------------
    def get_overflow(self) -> list:
        total_overflow = []
        
        for x in range(len(self.set_fill_levels)):
            overflow = self._fct_strans.get_component_states()[self.set_overflow[x]].get_value()
            total_overflow.append(overflow)
        return total_overflow


## -------------------------------------------------------------------------------------------------
    def get_power(self) -> list:
        total_power = []
        
        for x in range(len(self.set_power)):
            power = self._fct_strans.get_component_states()[self.set_power[x]].get_value()
            total_power.append(power)
        return total_power


## -------------------------------------------------------------------------------------------------
    def get_demand(self, init_volume, cur_volume) -> list:
        delta = cur_volume-init_volume
        
        if (self.demand*self.t_set) > delta:
            total_demand = delta-self.demand*self.t_set
        else:
            total_demand = 0
        return total_demand


## -------------------------------------------------------------------------------------------------
    def _compute_reward(self, p_state_old: State = None, p_state_new: State = None) -> Reward:
        reward = Reward(self.reward_type)

        if self.reward_type == Reward.C_TYPE_OVERALL:
            r_overall = 0
            r_overall = r_overall + sum(self.calc_reward())
            reward.set_overall_reward(r_overall)
        
        elif self.reward_type == Reward.C_TYPE_EVERY_AGENT:
           for agent_id in self._last_action.get_agent_ids():
               r_reward = self.calc_reward()
               reward.add_agent_reward(agent_id, r_reward[int(agent_id)])
               
        else:
           for agent_id in self._last_action.get_agent_ids():
                agent_action_elem = self._last_action.get_elem(agent_id)
                agent_action_ids = agent_action_elem.get_dim_ids()
                r_reward = self.calc_reward()
                action_idx = 0
                for action_id in agent_action_ids:
                    r_action = r_reward[action_idx]
                    action_idx += 1
                    reward.add_action_reward(agent_id, action_id, r_action)
                    
        return reward


## -------------------------------------------------------------------------------------------------
    def _compute_success(self, p_state: State) -> bool:        
        if self.prod_scenario == 'continuous':
            return False
        else:
            if self.prod_reached >= self.prod_target:
                self._state.set_terminal(True)
                return True
            else:
                return False


## -------------------------------------------------------------------------------------------------
    def _compute_broken(self, p_state: State) -> bool:
        return False


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None) -> None:

        # set seed
        random.seed(p_seed)

        # deactivate all actuators
        for acts in self._fct_strans.get_actuators():
            self._fct_strans.get_actuators()[acts].deactivate()

        # deactivate all sensors
        for sens in self._fct_strans.get_sensors():
            self._fct_strans.get_sensors()[sens].deactivate()
            
        # init tank fill level 
        for st in range(len(self.set_fill_levels)):
            buffer = self._fct_strans.get_component_states()[self.set_fill_levels[st]]          # get tank 
            boundaries = buffer.get_boundaries()                                                # get boundaries
            levels_init = random.uniform(0,1)                                                   # init uniform distribution
            fill_level = levels_init*(boundaries[1]-boundaries[0])+boundaries[0]                # compute fill level
            buffer.set_value(fill_level)                                                        # set fill level
        
        
        self.t = 0                              # reset time set
        self._state = self.get_states()         # set state
        self._state.set_success(False)          # set success
        self._state.set_broken(False)           # set broken
        
        # counte frame
        if self.data_frame == None:         
            self.data_frame = 0
        else:
            self.data_frame += 1
        self.data_storing.add_frame(str(self.data_frame))
            

## -------------------------------------------------------------------------------------------------
    def calc_reward(self):
        reward = []
        margin = self.get_margin()
        power = self.get_power()
        demand = self.current_demand/self.t_set
        
        for actnum, pwr in enumerate(self.set_power):
            try:
                power_max = self._fct_strans.get_component_states()[pwr]._function.max_power
            except:
                power_max = self._fct_strans.get_component_states()[pwr]._function.power
                
            reward.append(1/(1+self.lr_margin*margin[actnum]))
            reward[actnum] += 1/(1+self.lr_power*power[actnum]/(power_max/1000.0))
            if actnum == len(self.set_power)-1:
                reward[actnum] += 1/(1-self.lr_demand*demand)
            else:
                reward[actnum] += 1/(1+self.lr_margin*margin[actnum+1])
        return reward





