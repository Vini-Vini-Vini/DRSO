import sys
import os
import pickle
import numpy as np

# 1. カレントディレクトリを検索パスに追加（モジュールを読み込めるようにする）
sys.path.append(os.getcwd())

# 必要に応じて、pickleが参照しているモジュールをインポート可能な状態にします
# これにより、'Metrica_PitchControl' が見つからないエラーを回避できます。

# 2. 保存されたファイルのパスを指定
file_path = "../DRSO_data/data-statsbomb/wc2022/main/obso/3857256_5.0.pkl"

try:
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # 3. データの構成要素を確認
    print("--- Data Keys ---")
    print(data.keys())

    # 4. M-PPCF の数値チェック
    ppcf = data["PPCF"]
    print("\n--- M-PPCF Stats ---")
    print(f"Shape: {ppcf.shape}")
    print(f"Max value: {np.max(ppcf):.4f}")
    # 0以外の値（アタッキングサードの予測値）の平均
    if np.any(ppcf > 0):
        print(f"Mean (calculated area): {ppcf[ppcf > 0].mean():.4f}")

    # 5. M-OBSO の数値チェック
    obso = data["OBSO"]
    print("\n--- M-OBSO Stats ---")
    print(f"Max OBSO value in match: {np.max(obso):.4f}")

except Exception as e:
    print(f"Error loading pickle: {e}")

import sys
import os
import pickle
import numpy as np

# モジュール読み込みパスの追加
sys.path.append(os.getcwd())

file_path = "../DRSO_data/data-statsbomb/wc2022/main/obso/3857256_5.0.pkl"

def analyze_obso(path, label):
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    ppcf = data["PPCF"]
    obso = data["OBSO"]
    
    # 1. 各イベントごとの「合計OBSO」を計算（そのプレーの危険度）
    # 32x50のグリッドを足し合わせます
    event_obso_sums = np.sum(obso, axis=(1, 2))
    
    # 2. アタッキングサードで計算が行われたイベントの数
    active_events = np.where(np.max(ppcf, axis=(1, 2)) > 0)[0]
    
    print(f"\n=== Analysis for: {label} ===")
    print(f"Total events in match: {len(ppcf)}")
    print(f"Events in attacking third: {len(active_events)}")
    
    if len(active_events) > 0:
        # OBSOの統計
        print(f"\n[OBSO Statistics (Attacking Third Only)]")
        print(f"  Match Total OBSO (Sum): {np.sum(obso):.4f}")
        print(f"  Max OBSO in one event: {np.max(obso):.4f}")
        print(f"  Mean OBSO per active event: {np.mean(event_obso_sums[active_events]):.4f}")
        
        # トップ5イベントの特定
        top_5_idx = np.argsort(event_obso_sums)[-5:][::-1]
        print(f"\n[Top 5 Most Dangerous Events]")
        for i, idx in enumerate(top_5_idx):
            print(f"  {i+1}. Event ID: {idx:4d} | Value: {event_obso_sums[idx]:.4f}")

        # PPCF（支配確率）の統計
        print(f"\n[PPCF Statistics]")
        print(f"  Average Control Prob: {ppcf[ppcf > 0].mean():.4f}")
        print(f"  Max Control Prob: {np.max(ppcf):.4f}")
    else:
        print("No attacking third events found.")

# 実行（パスを自分の環境に合わせて書き換えてください）
analyze_obso(file_path, "Current Model (ML)")