### 2. calculate PPCF and OBSO ###
num_runs=1
max_parallel=2
command="CUDA_VISIBLE_DEVICES="5" python main.py --data statsbomb --game wc2022 --set_vel 5.0 --skip_load_rawdata --skip_compare_the_number_of_players --skip_verify_obso --skip_identify_optimal_positioning --skip_evaluate_team_defense --skip_show_results"
seq $num_runs | xargs -I {} -P $max_parallel sh -c "echo '[$(date)] Running command {}'; $command --count {}"
wait
