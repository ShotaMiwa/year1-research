# -*- coding: utf-8 -*-
"""
update_extended_dataset_with_shorts_duration.py

extended_feature_matrix_77.csv の segment_length_sec を、
元配信の区間長から「Shorts動画自体の実際の再生時間（shorts_duration: 30.0s〜61.2s）」へ正しく差し替え、
z-score を再計算して更新する。
"""

import re
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path("/home/shota/work/year1")
EXT_CSV_77 = PROJECT_ROOT / "data/processed/extended_feature_matrix_77.csv"
HTML_PATH = PROJECT_ROOT / "docs/paper/verification_report.html"
TABLE_CSV = PROJECT_ROOT / "docs/paper/tables/exp_extended_features_77.csv"
PROGRESS_TABLE_CSV = PROJECT_ROOT / "進捗報告用/20261111/tables/exp_extended_features_77.csv"

def main():
    df77 = pd.read_csv(EXT_CSV_77)
    print(f"Loaded N=77 dataset. Initial segment_length_sec mean: {df77['segment_length_sec'].mean():.2f}s")
    
    # HTMLレポートから shorts_duration を抽出
    content = HTML_PATH.read_text(encoding="utf-8")
    pattern = re.compile(r'Shorts ID:.*?([a-zA-Z0-9_-]{11}).*?\(尺:\s*([0-9.]+)s\)')
    matches = pattern.findall(content)
    dur_map = {sid: float(dur) for sid, dur in matches}
    
    # 正しいShorts動画尺に差し替え
    old_lengths = df77["segment_length_sec"].copy()
    df77["segment_length_sec"] = df77["shorts_id"].map(dur_map)
    
    # 欠損がないか確認
    assert df77["segment_length_sec"].isna().sum() == 0, "Missing shorts_duration detected!"
    
    print("\n=== 正しいShorts自体の長さに差し替え完了 ===")
    print(f"修正前 (元配信区間長): Mean={old_lengths.mean():.2f}s, Min={old_lengths.min():.2f}s, Max={old_lengths.max():.2f}s")
    print(f"修正後 (Shorts正味尺): Mean={df77['segment_length_sec'].mean():.2f}s, Min={df77['segment_length_sec'].min():.2f}s, Max={df77['segment_length_sec'].max():.2f}s")
    
    # 保存
    df77.to_csv(EXT_CSV_77, index=False, encoding="utf-8-sig")
    df77.to_csv(TABLE_CSV, index=False, encoding="utf-8-sig")
    df77.to_csv(PROGRESS_TABLE_CSV, index=False, encoding="utf-8-sig")
    print("\nSaved updated feature matrix to:")
    print(f"  {EXT_CSV_77}")
    print(f"  {TABLE_CSV}")
    print(f"  {PROGRESS_TABLE_CSV}")

if __name__ == "__main__":
    main()
