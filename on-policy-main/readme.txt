python train_matrix_game.py --check_eval --experiment_name population_idv_mappo_turn_update --idv_para True --num_env_steps 4000000

python test_generation_matrix_game.py --use_eval --experiment_name population_idv_mappo_turn_update --idv_para True --num_env_steps 4000


# control policy eval
python test_generation_matrix_game.py --use_eval --eval_policy --experiment_name population_idv_mappo_turn_update --idv_para True --num_env_steps 4000 --eval_policy_dir /home/storm/Documents/workspace_m/research/codes/mappo_ctrl/results/matrix_game/3m/mappo/control_idv_mappo/run1/models --eval_policy_num 0


# turn policy eval
python test_generation_matrix_game.py --use_eval --eval_policy --experiment_name population_idv_mappo --idv_para True --num_env_steps 4000 --eval_policy_dir /home/storm/Documents/workspace_m/research/codes/mappo_turn_update/results/matrix_game/3m/mappo/population_idv_mappo_turn_update/run1/models --eval_policy_num 0



# mpe train

python train_mpe.py --check_eval --num_agents 4 --env_name MPE --experiment_name population_idv_mappo_turn_update --idv_para True --num_env_steps 6000000


# simple spread

python train_mpe.py --check_eval --num_agents 4 --env_name MPE --experiment_name population_idv_mappo_turn_update --idv_para True --num_env_steps 6000000 --scenario_name simple_spread


3-1
python test_generalization_mpe.py --num_agents 4 --env_name MPE --experiment_name population_idv_mappo_turn_update --idv_para True --num_env_steps 200

2-2
python test_generalization_mpe.py --num_agents 4 --env_name MPE --experiment_name population_idv_mappo_turn_update --idv_para True --num_env_steps 200 --my_policy 2