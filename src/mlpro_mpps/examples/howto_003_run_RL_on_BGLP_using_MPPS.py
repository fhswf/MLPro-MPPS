## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.examples
## -- Module  : howto_003_run_RL_on_BGLP_using_MPPS.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-01-03  0.0.0     SY       Creation
## -- 2023-01-16  1.0.0     SY       Release of first version
## -- 2023-02-01  1.0.1     SY       Refactoring
## -- 2023-02-13  1.0.2     SY       Renaming module and refactoring
## -- 2023-02-15  1.0.3     SY       Incorporating SB3 algorithm via SB3 Wrapper of MLPro
## -- 2023-02-17  1.0.4     SY       Optimizing for paper
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.3 (2023-02-17)

This example demonstrates the implementation of the MPPS-based BGLP as an RL Environment.

You will learn:
    
    1) How to set up a built-in MPPS to a RL environment.
    
    2) How to set up RL scenario and training, including agents, policies, etc.
    
    3) How to incoporate SB3 algorithm into the RL training
    
"""


from mlpro_mpps.pool.mpps.PS001_bglp import BGLP
from mlpro.bf.math import *
from mlpro.rl.models import *
from stable_baselines3 import PPO, A2C
from mlpro.wrappers.sb3 import WrPolicySB32MLPro
from copy import deepcopy
import torch
import random
from pathlib import Path



                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class BGLP4RL(BGLP):


## -------------------------------------------------------------------------------------------------
    def __init__(self, p_name:str, p_id:int=None, p_logging=Log.C_LOG_ALL, **p_kwargs):
        super().__init__(p_name=p_name, p_id=p_id, p_logging=p_logging, p_kwargs=p_kwargs)
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
class BGLP_RLEnv(Environment):

    C_TYPE = 'Environment'
    C_NAME = 'MPPS-based BGLP - RL Environment'
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
                         p_fct_strans = BGLP4RL(p_name='BGLP_RL',
                                                p_logging=p_logging,
                                                p_parent=self), 
                         p_fct_reward = None, 
                         p_fct_success = None, 
                         p_fct_broken = None, 
                         p_visualize = False, 
                         p_logging = p_logging)
        
        self.C_SCIREF_TYPE    = self.C_SCIREF_TYPE_ARTICLE
        self.C_SCIREF_AUTHOR  = "Dorothea Schwung, Steve Yuwono, Andreas Schwung, Steven X. Ding"
        self.C_SCIREF_TITLE   = "Decentralized learning of energy optimal production policies using PLC-informed reinforcement learning"
        self.C_SCIREF_JOURNAL = "Computers & Chemical Engineering"
        self.C_SCIREF_YEAR    = "2021"
        self.C_SCIREF_MONTH   = "05"
        self.C_SCIREF_DAY     = "28"
        self.C_SCIREF_VOLUME  = "152"
        self.C_SCIREF_DOI     = "10.1016/j.compchemeng.2021.107382"
        
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
        
        self.set_overflow = ['SiloLoadingOverflow',
                             'HopperOverflow',
                             'SiloOverflow',
                             'HopperOverflow_1',
                             'SiloOverflow_1',
                             'HopperOverflow_2']
        self.set_fill_levels = ['SiloLoadingFillLevel',
                                'HopperFillLevel',
                                'SiloFillLevel',
                                'HopperFillLevel_1',
                                'SiloFillLevel_1',
                                'HopperFillLevel_2']
        self.set_power = ['CBPowerConsumption',
                          'VC1PowerConsumption',
                          'VCPowerConsumption',
                          'VC2PowerConsumption',
                          'RFPowerConsumption']
        
        self.reset()


## -------------------------------------------------------------------------------------------------
    @staticmethod
    def setup_spaces():
        state_space = ESpace()
        action_space = ESpace()

        state_space.add_dim(Dimension('R-1 LvlSiloA', 'R', 'Res-1 Level of Silo A', '', '', '', [0, 1]))
        state_space.add_dim(Dimension('R-2 LvlHopperA', 'R', 'Res-2 Level of Hopper A', '', '', '', [0, 1]))
        state_space.add_dim(Dimension('R-3 LvlSiloB', 'R', 'Res-3 Level of Silo B', '', '', '', [0, 1]))
        state_space.add_dim(Dimension('R-4 LvlHopperB', 'R', 'Res-4 Level of Hopper B', '', '', '', [0, 1]))
        state_space.add_dim(Dimension('R-5 LvlSiloC', 'R', 'Res-5 Level of Silo C', '', '', '', [0, 1]))
        state_space.add_dim(Dimension('R-6 LvlHopperC', 'R', 'Res-6 Level of Hopper C', '', '', '', [0, 1]))
        
        action_space.add_dim(Dimension('A-1 Act', 'R', 'Act-1 Belt Conveyor A', '', '', '', [0, 1]))
        action_space.add_dim(Dimension('A-2 Act', 'R', 'Act-2 Vacuum Pump B', '', '', '', [0, 1]))
        action_space.add_dim(Dimension('A-3 Act', 'Z', 'Act-3 Vibratory Conveyor B', '', '', '', [0, 1]))
        action_space.add_dim(Dimension('A-4 Act', 'R', 'Act-4 Vacuum Pump C', '', '', '', [0, 1]))
        action_space.add_dim(Dimension('A-5 Act', 'R', 'Act-5 Rotary Feeder C', '', '', '', [0, 1]))

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
        random.seed(p_seed)
        self._fct_strans.get_component_states()['VC1TransportedMaterial_1']._function.prod_target = self.demand
        
        for acts in self._fct_strans.get_actuators():
            self._fct_strans.get_actuators()[acts].deactivate()
        for sens in self._fct_strans.get_sensors():
            self._fct_strans.get_sensors()[sens].deactivate()
            
        for st in range(len(self.set_fill_levels)):
            buffer = self._fct_strans.get_component_states()[self.set_fill_levels[st]]
            boundaries = buffer.get_boundaries()
            levels_init = random.uniform(0,1)
            fill_level = levels_init*(boundaries[1]-boundaries[0])+boundaries[0]
            buffer.set_value(fill_level)
        self._fct_strans.get_component_states()['InventoryLevel'].set_value(0)
        
        self.t = 0
        self.prod_reached = 0
        self._state = self.get_states()
        self._state.set_success(False)
        self._state.set_broken(False)
        
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





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MyBGLP(RLScenario):

    C_NAME = 'My_BGLP'
    

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada, p_visualize, p_logging):
        self._env = BGLP_RLEnv(p_logging=p_logging)
        self._agent = MultiAgent(p_name='SB3 Policy', p_ada=1, p_logging=p_logging)
        state_space = self._env.get_state_space()
        action_space = self._env.get_action_space()
        
        policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                              net_arch=[dict(pi=[128, 128], vf=[128, 128])])
        
        policy_sb3 = A2C(
            policy="MlpPolicy",
            n_steps=100,
            env=None,
            _init_setup_model=False,
            policy_kwargs=policy_kwargs,
            device="cpu",
            seed=2)
        
        # Agent 1
        _name         = 'BELT_CONVEYOR_A'
        _id           = 0
        _ospace       = state_space.spawn([state_space.get_dim_ids()[0],state_space.get_dim_ids()[1]])
        _aspace       = action_space.spawn([action_space.get_dim_ids()[0]])
        
        _policy_wrapped = WrPolicySB32MLPro(
            p_sb3_policy=deepcopy(policy_sb3),
            p_cycle_limit=self._cycle_limit,
            p_observation_space=_ospace,
            p_action_space=_aspace,
            p_ada=p_ada,
            p_visualize=p_visualize,
            p_logging=p_logging)
        
        self._agent.add_agent(
            p_agent=Agent(
                p_policy=_policy_wrapped,
                p_envmodel=None,
                p_name=_name,
                p_id=_id,
                p_ada=True,
                p_logging=True),
            p_weight=1.0
            )
        
        
        # Agent 2
        _name         = 'VACUUM_PUMP_B'
        _id           = 1
        _ospace       = state_space.spawn([state_space.get_dim_ids()[1],state_space.get_dim_ids()[2]])
        _aspace       = action_space.spawn([action_space.get_dim_ids()[1]])
        
        _policy_wrapped = WrPolicySB32MLPro(
            p_sb3_policy=deepcopy(policy_sb3),
            p_cycle_limit=self._cycle_limit,
            p_observation_space=_ospace,
            p_action_space=_aspace,
            p_ada=p_ada,
            p_visualize=p_visualize,
            p_logging=p_logging)
        
        self._agent.add_agent(
            p_agent=Agent(
                p_policy=_policy_wrapped,
                p_envmodel=None,
                p_name=_name,
                p_id=_id,
                p_ada=True,
                p_logging=True),
            p_weight=1.0
            )
        
        
        # Agent 3
        _name         = 'VIBRATORY_CONVEYOR_B'
        _id           = 2
        _ospace       = state_space.spawn([state_space.get_dim_ids()[2],state_space.get_dim_ids()[3]])
        _aspace       = action_space.spawn([action_space.get_dim_ids()[2]])
        
        _policy_wrapped = WrPolicySB32MLPro(
            p_sb3_policy=deepcopy(policy_sb3),
            p_cycle_limit=self._cycle_limit,
            p_observation_space=_ospace,
            p_action_space=_aspace,
            p_ada=p_ada,
            p_visualize=p_visualize,
            p_logging=p_logging)
        
        self._agent.add_agent(
            p_agent=Agent(
                p_policy=_policy_wrapped,
                p_envmodel=None,
                p_name=_name,
                p_id=_id,
                p_ada=True,
                p_logging=True),
            p_weight=1.0
            )
        
        
        # Agent 4
        _name         = 'VACUUM_PUMP_C'
        _id           = 3
        _ospace       = state_space.spawn([state_space.get_dim_ids()[3],state_space.get_dim_ids()[4]])
        _aspace       = action_space.spawn([action_space.get_dim_ids()[3]])
        
        _policy_wrapped = WrPolicySB32MLPro(
            p_sb3_policy=deepcopy(policy_sb3),
            p_cycle_limit=self._cycle_limit,
            p_observation_space=_ospace,
            p_action_space=_aspace,
            p_ada=p_ada,
            p_visualize=p_visualize,
            p_logging=p_logging)
        
        self._agent.add_agent(
            p_agent=Agent(
                p_policy=_policy_wrapped,
                p_envmodel=None,
                p_name=_name,
                p_id=_id,
                p_ada=True,
                p_logging=True),
            p_weight=1.0
            )
        
        
        # Agent 5
        _name         = 'ROTARY_FEEDER_C'
        _id           = 4
        _ospace       = state_space.spawn([state_space.get_dim_ids()[4],state_space.get_dim_ids()[5]])
        _aspace       = action_space.spawn([action_space.get_dim_ids()[4]])
        
        _policy_wrapped = WrPolicySB32MLPro(
            p_sb3_policy=deepcopy(policy_sb3),
            p_cycle_limit=self._cycle_limit,
            p_observation_space=_ospace,
            p_action_space=_aspace,
            p_ada=p_ada,
            p_visualize=p_visualize,
            p_logging=p_logging)
        
        self._agent.add_agent(
            p_agent=Agent(
                p_policy=_policy_wrapped,
                p_envmodel=None,
                p_name=_name,
                p_id=_id,
                p_ada=True,
                p_logging=True),
            p_weight=1.0
            )
        
        return self._agent





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    logging         = Log.C_LOG_ALL
    visualize       = False
    dest_path       = str(Path.home())
    cycle_limit     = 200000
    cycle_per_ep    = 1000
    eval_freq       = 0
    eval_grp_size   = 0
    adapt_limit     = 0
    stagnant_limit  = 0
    score_ma_hor    = 0
else:
    logging         = Log.C_LOG_NOTHING
    visualize       = False
    dest_path       = str(Path.home())
    cycle_limit     = 10
    cycle_per_ep    = 10
    eval_freq       = 0
    eval_grp_size   = 0
    adapt_limit     = 0
    stagnant_limit  = 0
    score_ma_hor    = 0
    
training = RLTraining(
    p_scenario_cls=MyBGLP,
    p_cycle_limit=cycle_limit,
    p_cycles_per_epi_limit=cycle_per_ep,
    p_eval_frequency=eval_freq,
    p_eval_grp_size=eval_grp_size,
    p_adaptation_limit=adapt_limit,
    p_stagnation_limit=stagnant_limit,
    p_score_ma_horizon=score_ma_hor,
    p_collect_states=True,
    p_collect_actions=True,
    p_collect_rewards=True,
    p_collect_training=True,
    p_visualize=visualize,
    p_path=dest_path,
    p_logging=logging)

training.run()
if __name__ == "__main__":
    training._scenario.get_env().data_storing.save_data(training._root_path, 'bglp')

