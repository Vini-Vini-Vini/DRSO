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