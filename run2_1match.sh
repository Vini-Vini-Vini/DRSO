#!/bin/bash
### 2. calculate PPCF and OBSO ###
# 実行するコマンド
# --count 1 で最初の1試合を指定
# --skip_load_rawdata で26分のロードをスキップ
# 他の不要なフェーズは --skip フラグで止めています
python main.py --data statsbomb --game wc2022 --set_vel 5.0 \
  --count 1 \
  --skip_load_rawdata \
  --skip_compare_the_number_of_players \
  --skip_verify_obso \
  --skip_identify_optimal_positioning \
  --skip_evaluate_team_defense \
  --skip_show_results

echo "[$(date)] Match 1 processing finished."