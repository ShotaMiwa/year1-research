import pandas as pd
import numpy as np
from pathlib import Path

# パスの設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = PROJECT_ROOT / "data" / "processed" / "merged_survey_heatmap.csv"

def cronbach_alpha(df):
    """
    クロンバックのα係数を計算する関数
    df: 該当する因子の観測項目のみを含んだDataFrame
    """
    # 項目数 N
    N = df.shape[1]
    
    # 各質問項目の分散の合計
    item_variances = df.var(ddof=1).sum()
    
    # 合計スコアの分散
    total_variance = df.sum(axis=1).var(ddof=1)
    
    # クロンバックのα係数の公式
    alpha = (N / (N - 1)) * (1 - (item_variances / total_variance))
    return alpha

def main():
    print("🧮 クロンバックのα係数（信頼性係数）を計算します...")
    
    if not DATA_CSV.exists():
        print(f"❌ エラー: データファイルが見つかりません: {DATA_CSV}")
        return
        
    df = pd.read_csv(DATA_CSV)
    
    # 因子分析で採択された構成項目
    factor1_items = [
        "有益な内容だった",
        "新しい情報を得られた",
        "誰かに共有したいと思った"
    ]
    
    factor2_items = [
        "気分転換になった",
        "気軽に視聴できた",
        "面白かった"
    ]
    
    # 因子1 (情報価値・共有) のα係数
    df_f1 = df[factor1_items].dropna()
    alpha_f1 = cronbach_alpha(df_f1)
    
    # 因子2 (娯楽・リラックス) のα係数
    df_f2 = df[factor2_items].dropna()
    alpha_f2 = cronbach_alpha(df_f2)
    
    print("\n[ クロンバックのα係数 計算結果 ]")
    print(f"■ 因子1 『情報価値・共有』 (項目数: {len(factor1_items)})")
    print(f"  構成項目: {factor1_items}")
    print(f"  α係数  : {alpha_f1:.4f}")
    
    print(f"\n■ 因子2 『娯楽・リラックス』 (項目数: {len(factor2_items)})")
    print(f"  構成項目: {factor2_items}")
    print(f"  α係数  : {alpha_f2:.4f}")
    
    # 結果の判定
    for name, val in [("因子1", alpha_f1), ("因子2", alpha_f2)]:
        if val >= 0.8:
            status = "非常に高い信頼性 (非常に一貫している)"
        elif val >= 0.7:
            status = "十分な信頼性 (一貫している)"
        elif val >= 0.6:
            status = "許容範囲内の信頼性"
        else:
            status = "警告: 信頼性が低いです (一貫していない)"
        print(f"  -> {name}の判定: {status}")

if __name__ == "__main__":
    main()
