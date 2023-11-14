## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.examples
## -- Module  : howto_004_run_GT_on_BGLP_using_MPPS.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-02-13  0.0.0     SY       Creation
## -- 2023-02-13  1.0.0     SY       Release of first version
## -- 2023-02-17  1.0.1     SY       Refactoring
## -- 2023-02-28  1.0.2     SY       Update following new pool.ml
## -- 2023-11-09  1.0.3     SY       Refactoring
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.3 (2023-11-09)

This example demonstrates the implementation of the MPPS-based BGLP as an GT Game Board.

You will learn:
    
    1) How to set up a built-in MPPS to a GT gameboard from a RL environment.
    
    2) How to set up GT game and training, including players, random policies, etc.
    
"""


from mlpro_mpps.pool.ml.rl_environment.RL001_BGLP import BGLP_RLEnv
from mlpro.bf.math import *
from mlpro.gt.models import *
from mlpro.rl.pool.policies.randomgenerator import RandomGenerator
from pathlib import Path



                     
                        
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
        self._player = MultiPlayer(p_name='Random Policy', p_ada=1, p_logging=p_logging)
        state_space = self._env.get_state_space()
        action_space = self._env.get_action_space()
        
        
        # Player 1
        _name         = 'BELT_CONVEYOR_A'
        _id           = 0
        _ospace       = state_space.spawn([state_space.get_dim_ids()[0],state_space.get_dim_ids()[1]])
        _aspace       = action_space.spawn([action_space.get_dim_ids()[0]])
        _policy       = RandomGenerator(p_observation_space=_ospace, p_action_space=_aspace, p_buffer_size=1, p_ada=1, p_logging=False)
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
        _policy       = RandomGenerator(p_observation_space=_ospace, p_action_space=_aspace, p_buffer_size=1, p_ada=1, p_logging=False)
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
        _policy       = RandomGenerator(p_observation_space=_ospace, p_action_space=_aspace, p_buffer_size=1, p_ada=1, p_logging=False)
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
        _policy       = RandomGenerator(p_observation_space=_ospace, p_action_space=_aspace, p_buffer_size=1, p_ada=1, p_logging=False)
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
        _policy       = RandomGenerator(p_observation_space=_ospace, p_action_space=_aspace, p_buffer_size=1, p_ada=1, p_logging=False)
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
    cycle_limit     = 20000
    cycle_per_ep    = 100
    eval_freq       = 10
    eval_grp_size   = 5
    adapt_limit     = 0
    stagnant_limit  = 0
    score_ma_hor    = 5
else:
    logging         = Log.C_LOG_NOTHING
    visualize       = False
    dest_path       = str(Path.home())
    cycle_limit     = 10
    cycle_per_ep    = 10
    eval_freq       = 10
    eval_grp_size   = 1
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

