## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro_mpps.examples
## -- Module  : howto_008_run_GT_on_LS_BGLP_using_MPPS.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-11-09  0.0.0     SY       Creation
## -- 2023-11-14  1.0.0     SY       Release of first version
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2023-11-14)

This example demonstrates the implementation of the MPPS-based Larger-Scale BGLP as an GT Game Board.

You will learn:
    
    1) How to set up a built-in MPPS to a GT gameboard.
    
    2) How to set up GT game and training, including players, random policies, etc.
    
"""


from mlpro_mpps.pool.ml.gt_gameboard.GT001_LS_BGLP import LS_BGLP_GTGameBoard
from mlpro_mpps.pool.ml.gt_gameboard.GT002_LS_BGLP_SP import LS_BGLP_SP_GTGameBoard
from mlpro.bf.math import *
from mlpro.gt.models import *
from mlpro.rl.pool.policies.randomgenerator import RandomGenerator
from pathlib import Path



                                                 
                                                    
## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------

class MyGame(Game):

    C_NAME = 'MyGame'
    

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada, p_visualize, p_logging):
        self._env = LS_BGLP_GTGameBoard(p_logging=p_logging)
        self._player = MultiPlayer(p_name='Random Policy', p_ada=1, p_logging=p_logging)
        state_space = self._env.get_state_space()
        action_space = self._env.get_action_space()

        for x in range(14):
            _name         = 'ACT_' + str(x+1)
            _id           = x
            _ospace       = state_space.spawn([state_space.get_dim_ids()[x],state_space.get_dim_ids()[x+1]])
            _aspace       = action_space.spawn([action_space.get_dim_ids()[x]])
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

class MyGameSP(Game):

    C_NAME = 'MyGameSP'
    

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada, p_visualize, p_logging):
        self._env = LS_BGLP_SP_GTGameBoard(p_logging=p_logging)
        self._player = MultiPlayer(p_name='Random Policy', p_ada=1, p_logging=p_logging)
        state_space = self._env.get_state_space()
        action_space = self._env.get_action_space()

        for x in range(14):
            _name         = 'ACT_' + str(x+1)
            _id           = x
            _ospace       = state_space.spawn([state_space.get_dim_ids()[x],state_space.get_dim_ids()[x+1]])
            _aspace       = action_space.spawn([action_space.get_dim_ids()[x]])
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
    
    training = GTTraining(
        p_game_cls=MyGame,
        # p_game_cls=MyGameSP,
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
    training._scenario.get_env().data_storing.save_data(training._root_path, 'bglp')