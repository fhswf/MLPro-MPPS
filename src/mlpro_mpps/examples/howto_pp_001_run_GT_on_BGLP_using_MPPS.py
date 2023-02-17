## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.examples
## -- Module  : howto_pp_001_run_GT_on_BGLP_using_MPPS.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-02-17  0.0.0     SY       Creation
## -- 2023-02-17  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-02-17)

This example demonstrates the implementation of the MPPS-based BGLP as an GT Game Board using SbPG
for paper.
    
"""


from mlpro_mpps.pool.rl_environment.RL001_bglp import BGLP_RLEnv
from mlpro.bf.various import *
from mlpro.bf.math import *
from mlpro.bf.ml import *
import random
import numpy as np
import math as m
import torch
from mlpro.gt.models import *
import random
from pathlib import Path





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class SbPG_GlobI(Policy):

    C_NAME      = 'SbPG_GlobI'
    

## -------------------------------------------------------------------------------------------------
    def __init__(self, p_observation_space:MSpace, p_action_space:MSpace, p_buffer_size=1, p_ada=1.0, p_logging=True):
        super().__init__(p_observation_space=p_observation_space, p_action_space=p_action_space, p_buffer_size=p_buffer_size, p_ada=p_ada, p_logging=p_logging)
        self.additional_buffer_element  = {}
        
        self.C_SCIREF_TYPE    = self.C_SCIREF_TYPE_ARTICLE
        self.C_SCIREF_AUTHOR  = "Dorothea Schwung, Andreas Schwung, Steven X. Ding"
        self.C_SCIREF_TITLE   = "Distributed Self-Optimization of Modular Production Units: A State-Based Potential Game Approach"
        self.C_SCIREF_JOURNAL = "IEEE Transactions on Cybernetics"
        self.C_SCIREF_YEAR    = "2020"
        self.C_SCIREF_PAGES   = "1-12"
        self.C_SCIREF_DOI     = "10.1109/TCYB.2020.3006620"
        
        self.levels_current     = np.zeros(p_observation_space.get_num_dim())
        self.levels_current_con = np.zeros(p_observation_space.get_num_dim())
        self.levels_last        = np.zeros(p_observation_space.get_num_dim(), dtype=int)
        self.levels_last_con    = np.zeros(p_observation_space.get_num_dim())
        self.action_last        = np.zeros(p_action_space.get_num_dim())
        self.action_current     = np.zeros(p_action_space.get_num_dim())
        self.action_base_set    = p_action_space.get_dim(p_action_space.get_dim_ids()[-1]).get_base_set()
        self.exploration        = 1


## -------------------------------------------------------------------------------------------------
    def _init_hyperparam(self):
        self._hyperparam_space.add_dim(HyperParam(0,'num_states','Z'))
        self._hyperparam_space.add_dim(HyperParam(1,'smoothing','R'))
        self._hyperparam_space.add_dim(HyperParam(2,'lvl_max_silo','R'))
        self._hyperparam_space.add_dim(HyperParam(3,'lvl_max_hopper','R'))
        self._hyperparam_space.add_dim(HyperParam(4,'exp_decay','R'))
        self._hyperparam_tuple = HyperParamTuple(self._hyperparam_space)
        
        ids_ = self._hyperparam_tuple.get_dim_ids()
        self._hyperparam_tuple.set_value(ids_[0], 40)
        self._hyperparam_tuple.set_value(ids_[1], 0.000035)
        self._hyperparam_tuple.set_value(ids_[2], 17.42)
        self._hyperparam_tuple.set_value(ids_[3], 9.10)
        self._hyperparam_tuple.set_value(ids_[4], 0.999999)

        self.num_states         = self._hyperparam_tuple.get_value(ids_[0])
        self.levels_max         = [self._hyperparam_tuple.get_value(ids_[2]),
                                   self._hyperparam_tuple.get_value(ids_[3]),
                                   self._hyperparam_tuple.get_value(ids_[2]),
                                   self._hyperparam_tuple.get_value(ids_[3]),
                                   self._hyperparam_tuple.get_value(ids_[2]),
                                   self._hyperparam_tuple.get_value(ids_[3])]
        self.exp_decay          = self._hyperparam_tuple.get_value(ids_[4])
        self.smoothing          = self._hyperparam_tuple.get_value(ids_[1])
        self.map_utility        = torch.zeros(int(self.num_states), int(self.num_states))
        self.map_action         = torch.zeros(int(self.num_states), int(self.num_states))
        self.grid               = (torch.arange(int(self.num_states)).float()+1)/(int(self.num_states))-1/(int(self.num_states)*2)
        self.grid_center_x      = torch.zeros(int(self.num_states), int(self.num_states))
        self.grid_center_y      = torch.zeros(int(self.num_states), int(self.num_states))
        for x in range(int(self.num_states)):
            for y in range(int(self.num_states)):
                self.grid_center_x[x,y] = self.grid[x]
                self.grid_center_y[x,y] = self.grid[y]


## -------------------------------------------------------------------------------------------------
    def calc_current_states(self, levels):
        for i in range(self._observation_space.get_num_dim()):
            levels_cur = levels[i]*self.levels_max[i]
            self.levels_current[i] = min(m.floor(self.num_states*levels_cur/self.levels_max[i]),self.num_states-1)


## -------------------------------------------------------------------------------------------------
    def memorize_levels(self):
        for i in range(self._observation_space.get_num_dim()):
            self.levels_last[i] = self.levels_current[i]
            self.levels_last_con[i] = self.levels_current_con[i]


## -------------------------------------------------------------------------------------------------
    def compute_action(self, p_state: State) -> Action:
        self.action_last[0] = self.action_current[0]
        self.memorize_levels()
        states = p_state.get_values()
        for i in range(self._observation_space.get_num_dim()):
            self.levels_current_con[i] = states[i]
        self.calc_current_states(states)
        if random.uniform(0,1) <= self.exploration and self._adaptivity:
            self.action_current[0] = random.uniform(0,1)
        else:
            self.action_current[0] = self.interpolate_map(states[0],states[1])
        if self.action_base_set == "Z":
            if self.action_current[0] >= 0.5:
                self.action_current[0] = 1
            else:
                self.action_current[0] = 0
        return Action(self._id, self._action_space, self.action_current)


## -------------------------------------------------------------------------------------------------
    def _adapt(self, **p_args) -> bool:
        self.add_buffer(p_args['p_sars_elem'])
        
        if not self._buffer.is_full():
            return False
        
        sar_data = self._buffer.get_all()
        for reward in sar_data["reward"]:
            rwd = reward.get_agent_reward(self._id)
        self.update_maps(self.action_last[0], rwd, self.levels_last)
        self.exploration = self.exploration*self.exp_decay
        self.log(self.C_LOG_TYPE_I, 'Performance map is updated')
        return True


## -------------------------------------------------------------------------------------------------
    def update_maps(self, action, utility, levels):
        if utility > self.map_utility[levels[1], levels[0]].item():
            self.update_area(levels, action, utility)


## -------------------------------------------------------------------------------------------------
    def update_area(self, levels, action, utility):
        self.map_action[levels[1], levels[0]] = action
        self.map_utility[levels[1], levels[0]] = utility


## -------------------------------------------------------------------------------------------------
    def interpolate_map(self, pos_y, pos_x):
        distances = torch.sqrt((pos_x.item()-self.grid_center_x)**2+(pos_y.item()-self.grid_center_y)**2)
        distances[distances == 0] = 0.0001
        weights = 1/(distances**2)+self.smoothing
        DistancesTotal = sum(sum(weights))
        outputs = weights/DistancesTotal*self.map_action
        return (sum(sum(outputs))).item()


## -------------------------------------------------------------------------------------------------
    def add_buffer(self, p_buffer_element: SARSElement):
        """
        Intended to save the data to the buffer. By default it save the SARS data.
        
        """
        buffer_element = self._add_additional_buffer(p_buffer_element)
        self._buffer.add_element(buffer_element)


## -------------------------------------------------------------------------------------------------
    def clear_buffer(self):
        self._buffer.clear()


## -------------------------------------------------------------------------------------------------
    def _add_additional_buffer(self, p_buffer_element: SARSElement):
        p_buffer_element.add_value_element(self.additional_buffer_element)
        return p_buffer_element



                     
                        
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class BGLP_GTGameBoard(BGLP_RLEnv, GameBoard):
    """
    Game theoretical pendant for the reinforcement learning environment class BGLP.
    """

    C_NAME          = 'BGLP_GTGameBoard'

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
        BGLP_RLEnv.__init__(self,
                            p_reward_type=p_reward_type,
                            p_logging=p_logging,
                            t_set=t_set,
                            demand=demand,
                            lr_margin=lr_margin,
                            lr_demand=lr_demand,
                            lr_power=lr_power, 
                            margin_p=margin_p,
                            prod_target=prod_target,
                            prod_scenario=prod_scenario,
                            cycle_limit=cycle_limit)



                                                 
                                                    
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class MyGame(Game):

    C_NAME = 'MyGame'
    

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada, p_visualize, p_logging):
        self._env = BGLP_GTGameBoard(p_logging=p_logging)
        self._player = MultiPlayer(p_name='SbPG', p_ada=1, p_logging=p_logging)
        state_space = self._env.get_state_space()
        action_space = self._env.get_action_space()
        
        
        # Player 1
        _name         = 'BELT_CONVEYOR_A'
        _id           = 0
        _ospace       = state_space.spawn([state_space.get_dim_ids()[0],state_space.get_dim_ids()[1]])
        _aspace       = action_space.spawn([action_space.get_dim_ids()[0]])
        _policy       = SbPG_GlobI(p_observation_space=_ospace, p_action_space=_aspace, p_buffer_size=1, p_ada=1, p_logging=False)
        self._player.add_player(
            p_player=Agent(
                p_policy=_policy,
                p_envmodel=None,
                p_name=_name,
                p_id=_id,
                p_ada=True,
                p_logging=True),
            p_weight=1.0
            )
        
        
        # Player 2
        _name         = 'VACUUM_PUMP_B'
        _id           = 1
        _ospace       = state_space.spawn([state_space.get_dim_ids()[1],state_space.get_dim_ids()[2]])
        _aspace       = action_space.spawn([action_space.get_dim_ids()[1]])
        _policy       = SbPG_GlobI(p_observation_space=_ospace, p_action_space=_aspace, p_buffer_size=1, p_ada=1, p_logging=False)
        self._player.add_player(
            p_player=Agent(
                p_policy=_policy,
                p_envmodel=None,
                p_name=_name,
                p_id=_id,
                p_ada=True,
                p_logging=True),
            p_weight=1.0
            )
        
        
        # Player 3
        _name         = 'VIBRATORY_CONVEYOR_B'
        _id           = 2
        _ospace       = state_space.spawn([state_space.get_dim_ids()[2],state_space.get_dim_ids()[3]])
        _aspace       = action_space.spawn([action_space.get_dim_ids()[2]])
        _policy       = SbPG_GlobI(p_observation_space=_ospace, p_action_space=_aspace, p_buffer_size=1, p_ada=1, p_logging=False)
        self._player.add_player(
            p_player=Agent(
                p_policy=_policy,
                p_envmodel=None,
                p_name=_name,
                p_id=_id,
                p_ada=True,
                p_logging=True),
            p_weight=1.0
            )
        
        
        # Player 4
        _name         = 'VACUUM_PUMP_C'
        _id           = 3
        _ospace       = state_space.spawn([state_space.get_dim_ids()[3],state_space.get_dim_ids()[4]])
        _aspace       = action_space.spawn([action_space.get_dim_ids()[3]])
        _policy       = SbPG_GlobI(p_observation_space=_ospace, p_action_space=_aspace, p_buffer_size=1, p_ada=1, p_logging=False)
        self._player.add_player(
            p_player=Agent(
                p_policy=_policy,
                p_envmodel=None,
                p_name=_name,
                p_id=_id,
                p_ada=True,
                p_logging=True),
            p_weight=1.0
            )
        
        
        # Player 5
        _name         = 'ROTARY_FEEDER_C'
        _id           = 4
        _ospace       = state_space.spawn([state_space.get_dim_ids()[4],state_space.get_dim_ids()[5]])
        _aspace       = action_space.spawn([action_space.get_dim_ids()[4]])
        _policy       = SbPG_GlobI(p_observation_space=_ospace, p_action_space=_aspace, p_buffer_size=1, p_ada=1, p_logging=False)
        self._player.add_player(
            p_player=Agent(
                p_policy=_policy,
                p_envmodel=None,
                p_name=_name,
                p_id=_id,
                p_ada=True,
                p_logging=True),
            p_weight=1.0
            )
        
        return self._player





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
    
training = GTTraining(
    p_game_cls=MyGame,
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

