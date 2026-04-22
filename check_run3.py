import pickle
import pandas as pd
import sys
import os

# モジュール読み込みパスの設定
sys.path.append(os.getcwd())

# ファイルパス
path_team = "../DRSO_data/data-statsbomb/wc2022/main/metrics/metrics_game_team_5.0.pkl"
path_result = "../DRSO_data/data-statsbomb/wc2022/main/metrics/metrics_result_5.0.pkl"

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

print("=== 1. metrics_game_team_5.0.pkl の確認 ===")
# 通常、試合ごとの各チームの守備指標が入っています
team_metrics = load_pkl(path_team)
if isinstance(team_metrics, pd.DataFrame):
    print(team_metrics.head())
    # 重要な指標（守備評価値など）の平均を表示
    print("\n[指標別の全チーム平均]")
    print(team_metrics.mean(numeric_only=True))
else:
    print(f"Type: {type(team_metrics)}")
    print(team_metrics)

print("\n" + "="*40 + "\n")

print("=== 2. metrics_result_5.0.pkl の確認 ===")
# 通常、全試合を通じたチームごとの最終ランキングや集計結果が入っています
result_metrics = load_pkl(path_result)
if isinstance(result_metrics, pd.DataFrame):
    # スコア（守備評価値）の高い順に並び替えて表示
    # カラム名は 'score' や 'defense_value' など、実際の出力に合わせて調整してください
    print(result_metrics.sort_values(by=result_metrics.columns[0], ascending=False))
else:
    print(result_metrics)