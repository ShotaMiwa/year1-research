import pandas as pd
from pathlib import Path

# パスの設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = PROJECT_ROOT / "data" / "processed" / "merged_survey_heatmap.csv"
OUT_CSV = PROJECT_ROOT / "outputs" / "correlation" / "descriptive_statistics.csv"

def main():
    print("📊 観測変数の基本統計量（平均値・標準偏差）を計算します...")
    
    # データの読み込み
    if not DATA_CSV.exists():
        print(f"❌ エラー: データファイルが見つかりません: {DATA_CSV}")
        return
        
    df = pd.read_csv(DATA_CSV)
    
    # 計算対象の観測変数リスト（8項目 + 総合満足度）
    target_columns = [
        "面白かった",
        "話に引き込まれた",
        "テンポが良かった",
        "新しい情報を得られた",
        "有益な内容だった",
        "誰かに共有したいと思った",
        "気軽に視聴できた",
        "気分転換になった",
        "この動画に満足した"
    ]
    
    # 存在するカラムのみに絞り込む
    valid_columns = [col for col in target_columns if col in df.columns]
    
    if len(valid_columns) != len(target_columns):
        missing = set(target_columns) - set(valid_columns)
        print(f"⚠️ 警告: 一部のカラムがデータ内に見つかりません: {missing}")
    
    # 平均値と標準偏差の計算
    stats_df = pd.DataFrame({
        "観測変数": valid_columns,
        "平均値 (Mean)": df[valid_columns].mean().values,
        "標準偏差 (Std Dev)": df[valid_columns].std().values
    })
    
    # 結果の保存
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    
    print(f"✅ 結果を保存しました: {OUT_CSV}")
    print("\n[ 計算結果 ]")
    print(stats_df.to_string(index=False))

if __name__ == "__main__":
    main()
