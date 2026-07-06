import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── 日本語フォント設定 ─────────────────────────────────────
import japanize_matplotlib

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "analysis_questionnaire"
MERGED_SURVEY_PATH = DATA_DIR / "merged_survey_heatmap.csv"
SENTIMENT_RESULTS_PATH = PROJECT_ROOT / "outputs" / "run_20260624_071441" / "segment_sentiment_comparison_results.csv"
OUTPUT_COMP_PATH = DATA_DIR / "group_quantitative_comparison.csv"
OUTPUT_IMG_PATH = DATA_DIR / "group_sentiment_comparison.png"

def main():
    print("📊 4グループ間の定量データ（感情・コメント）の比較分析を開始します...")
    
    # データの読み込み
    df_survey_hm = pd.read_csv(MERGED_SURVEY_PATH)
    df_sentiment = pd.read_csv(SENTIMENT_RESULTS_PATH)
    
    # マージ
    df_compare = pd.merge(df_survey_hm, df_sentiment, on="セグメント", suffixes=('', '_sent'))
    
    # 1秒あたりのコメント数 (コメント速度) を算出
    df_compare["コメント速度(件/秒)"] = df_compare["コメント数"] / df_compare["長さ(秒)_sent"]
    # 1秒あたりの字幕数 (発話速度・テンポ) を算出
    df_compare["発話速度(文/秒)"] = df_compare["字幕数"] / df_compare["長さ(秒)_sent"]
    
    # 分析に使用するカラム
    target_cols = [
        "グループ",
        "平均ヒートマップ値",
        "この動画に満足した",
        "字幕数",
        "コメント数",
        "コメント速度(件/秒)",
        "発話速度(文/秒)",
        "日本語BERT_コメント_Positive(%)",
        "日本語BERT_コメント_Neutral(%)",
        "日本語BERT_コメント_Negative(%)",
        "多言語XLM-R_コメント_Positive(%)",
        "多言語XLM-R_コメント_Neutral(%)",
        "多言語XLM-R_コメント_Negative(%)",
        "日本語BERT_字幕_Positive(%)",
        "日本語BERT_字幕_Neutral(%)",
        "日本語BERT_字幕_Negative(%)"
    ]
    
    # グループごとに平均値を計算
    df_grouped = df_compare[target_cols].groupby("グループ").mean().reset_index()
    
    # グループごとの件数（セグメント数）を追加
    group_counts = df_compare["グループ"].value_counts().to_dict()
    df_grouped.insert(1, "セグメント数", df_grouped["グループ"].map(group_counts))
    
    # 小数点以下の表示を丸める
    df_grouped = df_grouped.round(4)
    
    # 結果の保存
    df_grouped.to_csv(OUTPUT_COMP_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ 定量比較表を保存しました: {OUTPUT_COMP_PATH}")
    print(df_grouped.to_string(index=False))
    
    # ── 可視化（積み上げ棒グラフ） ───────────────────────
    # 日本語BERTによるコメント感情比率のグループ比較
    groups = df_grouped["グループ"].tolist()
    pos = df_grouped["日本語BERT_コメント_Positive(%)"]
    neu = df_grouped["日本語BERT_コメント_Neutral(%)"]
    neg = df_grouped["日本語BERT_コメント_Negative(%)"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 積み上げ棒グラフ
    bar_width = 0.5
    r = np.arange(len(groups))
    
    ax.bar(r, pos, color='#2ecc71', edgecolor='white', width=bar_width, label='Positive')
    ax.bar(r, neu, bottom=pos, color='#f1c40f', edgecolor='white', width=bar_width, label='Neutral')
    ax.bar(r, neg, bottom=[i+j for i,j in zip(pos, neu)], color='#e74c3c', edgecolor='white', width=bar_width, label='Negative')
    
    ax.set_title("グループ別コメント感情比率の比較 (日本語BERT)", fontsize=14, pad=15)
    ax.set_xticks(r)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylabel("比率 (%)", fontsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(0, 100)
    
    # 各領域に数値をテキスト表示
    for i in range(len(groups)):
        # Positive
        ax.text(i, pos[i]/2, f"{pos[i]:.1f}%", ha='center', va='center', color='white', fontweight='bold')
        # Neutral
        ax.text(i, pos[i] + neu[i]/2, f"{neu[i]:.1f}%", ha='center', va='center', color='black', fontweight='bold')
        # Negative
        ax.text(i, pos[i] + neu[i] + neg[i]/2, f"{neg[i]:.1f}%", ha='center', va='center', color='white', fontweight='bold')
        
    plt.tight_layout()
    fig.savefig(OUTPUT_IMG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ 感情比率のグループ比較グラフを保存しました: {OUTPUT_IMG_PATH}")

if __name__ == "__main__":
    main()
