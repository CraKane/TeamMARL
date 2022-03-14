python train_matrix_game.py --experiment_name control_idv_mappo --idv_para True --num_env_steps 4000000

python test_generation_matrix_game.py --use_eval --experiment_name control_idv_mappo --idv_para True --num_env_steps 4000

# mappo policy eval
python test_generation_matrix_game.py --use_eval --eval_policy --experiment_name control_idv_mappo --idv_para True --num_env_steps 4000 --eval_policy_dir /home/storm/Documents/workspace_m/research/codes/mappo/results/matrix_game/3m/mappo/population_idv_mappo/run1/models --eval_policy_num 0


# turn policy eval
python test_generation_matrix_game.py --use_eval --eval_policy --experiment_name control_idv_mappo --idv_para True --num_env_steps 4000 --eval_policy_dir /home/storm/Documents/workspace_m/research/codes/mappo_turn_update/results/matrix_game/3m/mappo/population_idv_mappo_turn_update/run1/models --eval_policy_num 0





# mpe environment

python train_mpe.py --num_agents 4 --env_name MPE --experiment_name control_idv_mappo --idv_para True --num_env_steps 4000000
