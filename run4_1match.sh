#!/bin/bash

# 1試合分だけ実行するように変更
num_runs=1
max_parallel=1

# コマンドの定義
command="python main.py --data statsbomb --game wc2022 --set_vel 5.0 --skip_load_rawdata --skip_compare_the_number_of_players --skip_calculate_obso --skip_verify_obso --skip_evaluate_team_defense --skip_show_results"

# 1から1まで（つまり1回だけ）実行
seq $num_runs | xargs -I {} -P $max_parallel sh -c "echo '[$(date)] Running command {}'; $command --count {}"

wait
echo "Done."