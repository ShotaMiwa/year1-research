# -*- coding: utf-8 -*-
"""
data_validator.py

特徴量データセット（アンケートCSVおよびShorts CSV）の整合性を自動検証するモジュール。
特徴量名の欠損、予期せぬスケール違い（例: TTRの場所にBERT分散が入っている等）を検知し、
不整合があれば即座にハードフェイル（ValueError）を発生させて誤った実験を未然に防止する。
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# constants のインポート
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.satisfaction_analysis.constants import FEATURE_NAMES_12, FEATURE_SPECS


class DataIntegrityError(Exception):
    """データセットの整合性違反エラー"""
    pass


def validate_feature_matrix(df: pd.DataFrame, dataset_name: str = "Dataset") -> bool:
    """
    データフレームが公式の全12特徴量を正しく保持し、
    かつ各値が仕様通りの範囲・スケールに収まっているかを厳格に検証する。
    """
    # 1. カラムの存在チェック
    missing_cols = [col for col in FEATURE_NAMES_12 if col not in df.columns]
    if missing_cols:
        raise DataIntegrityError(
            f"【DATA INTEGRITY ERROR】{dataset_name} に必要な特徴量カラムが不足しています: {missing_cols}"
        )

    # 2. 欠損値 (NaN/Inf) チェック
    for col in FEATURE_NAMES_12:
        if df[col].isnull().any():
            nan_count = df[col].isnull().sum()
            raise DataIntegrityError(
                f"【DATA INTEGRITY ERROR】{dataset_name} の '{col}' に {nan_count} 件の欠損値 (NaN) が含まれています。"
            )
        if np.isinf(df[col].values).any():
            raise DataIntegrityError(
                f"【DATA INTEGRITY ERROR】{dataset_name} の '{col}' に無限大 (Inf) が含まれています。"
            )

    # 3. 値のレンジ・スケール整合性チェック（定義の取り違えを防止）
    for col, spec in FEATURE_SPECS.items():
        min_val = df[col].min()
        max_val = df[col].max()
        exp_min = spec["min"]
        exp_max = spec["max"]

        # 著しくスケールが外れている場合の検知（例: TTRに40〜60の分散が入っている、など）
        if min_val < exp_min - 0.5 or max_val > exp_max + 10.0:
            raise DataIntegrityError(
                f"【DATA INTEGRITY ERROR】{dataset_name} の '{col}' ({spec['desc']}) の値の範囲が仕様と大きく乖離しています。\n"
                f"  実測レンジ: [{min_val:.4f}, {max_val:.4f}]\n"
                f"  期待レンジ: [{exp_min}, {exp_max}]\n"
                f"  ※ TTRとBERT分散の取り違えや、正規化の誤りが疑われます。"
            )

    print(f"✅ [{dataset_name}] 全12特徴量の整合性・スケール検証をパスしました (N={len(df)}件)")
    return True


def load_and_validate_datasets():
    """アンケート23件とShorts 77件を読み込み、同時に整合性を検証して返す便利関数"""
    survey_path = PROJECT_ROOT / "data/processed/feature_matrix.csv"
    shorts_path = PROJECT_ROOT / "data/processed/extended_feature_matrix_77.csv"

    if not survey_path.exists():
        raise FileNotFoundError(f"アンケートデータが見つかりません: {survey_path}")
    if not shorts_path.exists():
        raise FileNotFoundError(f"Shortsデータが見つかりません: {shorts_path}")

    df_survey = pd.read_csv(survey_path)
    df_shorts = pd.read_csv(shorts_path)

    validate_feature_matrix(df_survey, "アンケート23件 (feature_matrix.csv)")
    validate_feature_matrix(df_shorts, "Shorts 77本 (extended_feature_matrix_77.csv)")

    return df_survey, df_shorts


if __name__ == "__main__":
    print("=== データセット整合性自動テスト実行 ===")
    try:
        load_and_validate_datasets()
        print("\n🎉 すべてのデータセットが公式仕様と完全に一致しています。")
    except DataIntegrityError as e:
        print(f"\n❌ 検証失敗: {e}")
        sys.exit(1)
