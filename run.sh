#!/bin/bash



### 1. load and convert statsbomb data ###
### 1.2. Compare the number of players in attacking-third and in non attacking-third area. ###
python main.py --data statsbomb --game wc2022 --skip_calculate_obso --skip_verify_obso --skip_identify_optimal_positioning --skip_evaluate_team_defense --skip_show_results

### 2. calculate PPCF and OBSO ###
#num_runs=180
#command="python main.py --data statsbomb --game wc2022 --set_vel 5.0 --skip_load_rawdata --skip_compare_the_number_of_players --skip_verify_obso --skip_identify_optimal_positioning --skip_evaluate_team_defense --skip_show_results"
#seq $num_runs | xargs -I {} -P $num_runs sh -c "echo '[$(date)] Running command {}'; $command --count {}"
#wait

### 3. Verify sample OBSO. ###
#python main.py --data statsbomb --game wc2022 --set_vel 5.0 --skip_load_rawdata --skip_compare_the_number_of_players --skip_calculate_obso --skip_identify_optimal_positioning --skip_evaluate_team_defense --skip_show_results

### 4. Identify optimal positioning strategies. ###
#num_runs=180
#command="python main.py --data statsbomb --game wc2022 --set_vel 5.0 --skip_load_rawdata --skip_compare_the_number_of_players --skip_calculate_obso --skip_verify_obso --skip_evaluate_team_defense --skip_show_results"
#seq $num_runs | xargs -I {} -P $num_runs sh -c "echo '[$(date)] Running command {}'; $command --count {}"
#wait

# ### 5. Evaluate team defense ###
# python main.py --count 0 --data statsbomb --game wc2022 --set_vel 5.0 --skip_load_rawdata --skip_compare_the_number_of_players --skip_calculate_obso --skip_verify_obso --skip_identify_optimal_positioning