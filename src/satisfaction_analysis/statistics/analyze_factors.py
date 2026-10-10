#!/usr/bin/env python3
"""
アンケート質問項目の因子分析 (Factor Analysis) スクリプト

目的:
  アンケートの評価項目（8項目）の背後にある共通の因子構造を抽出し、
  「娯楽・没頭（因子A）」と「情報価値・共有（因子B）」への分類を統計的に検証・可視化する。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from factor_analyzer import FactorAnalyzer
import japanize_matplotlib

from src.utils.config import (
    init_directories,
    PROJECT_ROOT,
    DATA_PROCESSED_DIR,
    OUT_CORRELATION_DIR
)

# ディレクトリ初期化
init_directories()

MERGED_CSV = DATA_PROCESSED_DIR / "merged_survey_heatmap.csv"
OUT_DIR = OUT_CORRELATION_DIR

def run_factor_analysis():
    print("🔮 因子分析 (Exploratory Factor Analysis) を開始します...")
    
    # データ読み込み
    df = pd.read_csv(MERGED_CSV)
    
    # 総合満足度(目的変数)を除く、8つの評価項目を分析対象にする
    items = [
        "面白かった",
        "話に引き込まれた",
        "テンポが良かった",
        "新しい情報を得られた",
        "有益な内容だった",
        "誰かに共有したいと思った",
        "気軽に視聴できた",
        "気分転換になった"
    ]
    
    X = df[items].dropna()
    print(f"分析対象セグメント数: {len(X)}")
    
    # ── 1. 因子数の決定（固有値の確認） ──
    fa_temp = FactorAnalyzer(rotation=None)
    fa_temp.fit(X)
    ev, v = fa_temp.get_eigenvalues()
    print("\n[ 固有値 (Eigenvalues) ]")
    for i, val in enumerate(ev, 1):
        print(f"  因子 {i}: {val:.4f}")
    
    # ── 1b. 平行分析 (Parallel Analysis) ──
    # ランダムデータの固有値と比較し、実データの固有値が偶然を上回る因子数を特定する
    n_iterations = 1000
    n_obs, n_vars = X.shape
    random_eigenvalues = np.zeros((n_iterations, n_vars))
    
    for iteration in range(n_iterations):
        random_data = np.random.normal(size=(n_obs, n_vars))
        fa_random = FactorAnalyzer(rotation=None)
        fa_random.fit(random_data)
        random_ev, _ = fa_random.get_eigenvalues()
        random_eigenvalues[iteration, :] = random_ev[:n_vars]
    
    # ランダムデータの95パーセンタイル固有値
    random_ev_95 = np.percentile(random_eigenvalues, 95, axis=0)
    
    print("\n[ 平行分析 (Parallel Analysis) ]")
    print(f"  （{n_iterations}回のランダムデータ生成、95パーセンタイル基準）")
    pa_n_factors = 0
    for i in range(n_vars):
        marker = "✅" if ev[i] > random_ev_95[i] else "❌"
        print(f"  因子 {i+1}: 実データ={ev[i]:.4f}  ランダム95%tile={random_ev_95[i]:.4f}  {marker}")
        if ev[i] > random_ev_95[i]:
            pa_n_factors = i + 1
    
    print(f"\n-> 平行分析の結果: 推奨因子数 = {pa_n_factors}")
    
    # ── 1c. スクリープロット + 平行分析の可視化 ──
    fig, ax = plt.subplots(figsize=(8, 5))
    factor_nums = np.arange(1, n_vars + 1)
    
    ax.plot(factor_nums, ev[:n_vars], marker="o", linewidth=2.5, color="#2C3E50",
            markersize=9, label="実データの固有値", zorder=5)
    ax.plot(factor_nums, random_ev_95, marker="s", linewidth=2, color="#E74C3C",
            linestyle="--", markersize=7, label="ランダムデータ 95%tile", zorder=4)
    ax.axhline(1.0, color="#95A5A6", linestyle=":", linewidth=1.5, label="カイザー基準 (固有値=1)")
    
    # 交差点（平行分析で因子数が決まるポイント）をハイライト
    ax.axvline(pa_n_factors + 0.5, color="#2ECC71", linestyle="-.", linewidth=1.5, alpha=0.7,
               label=f"平行分析による推奨因子数 = {pa_n_factors}")
    
    ax.set_xlabel("因子番号", fontsize=12)
    ax.set_ylabel("固有値", fontsize=12)
    ax.set_title("スクリープロット + 平行分析\n(実データ vs ランダムデータの固有値比較)", fontsize=13, pad=15)
    ax.set_xticks(factor_nums)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    scree_path = OUT_DIR / "scree_plot_parallel_analysis.png"
    fig.savefig(scree_path, dpi=150)
    plt.close(fig)
    print(f"✅ スクリープロット（平行分析付き）を保存しました: {scree_path}")
    
    # 因子数の最終決定
    # カイザー基準 (固有値>1) と平行分析の結果が一致しているか確認
    kaiser_n = int(sum(ev > 1))
    if kaiser_n == pa_n_factors:
        n_factors = kaiser_n
        print(f"\n✅ カイザー基準 ({kaiser_n}) と平行分析 ({pa_n_factors}) が一致 → 因子数 = {n_factors}")
    else:
        # 平行分析を優先（小サンプル時はカイザー基準より信頼度が高い）
        n_factors = pa_n_factors
        print(f"\n⚠️ カイザー基準 ({kaiser_n}) と平行分析 ({pa_n_factors}) が不一致 → 平行分析を優先し、因子数 = {n_factors}")
    
    # ── 2. 因子分析の実行 ──
    # 平行分析の推奨因子数とカイザー基準の両方で分析を行い、比較する
    models_to_run = sorted(set([pa_n_factors, kaiser_n]))
    
    for n_f in models_to_run:
        print(f"\n{'='*60}")
        print(f"  因子数 = {n_f} での因子分析")
        print(f"{'='*60}")
        
        # 回転の設定（1因子の場合は回転なし、2因子以上はプロマックス）
        rotation = "promax" if n_f >= 2 else None
        rotation_label = "プロマックス回転" if n_f >= 2 else "回転なし"
        
        fa = FactorAnalyzer(n_factors=n_f, method="minres", rotation=rotation)
        fa.fit(X)
        
        # 因子負荷量の取得
        col_names = [f"因子{i+1}" for i in range(n_f)]
        loadings = pd.DataFrame(fa.loadings_, index=items, columns=col_names)
        
        print(f"\n[ 因子負荷量 ({rotation_label}) ]")
        print(loadings.round(4))
        
        # 各因子の分散説明率
        variance_df = pd.DataFrame(
            fa.get_factor_variance(),
            index=["SS Loadings (累積寄与平方和)", "Proportion Var (分散説明率)", "Cumulative Var (累積説明率)"],
            columns=col_names
        )
        print(f"\n[ 因子の分散説明率 ]")
        print(variance_df.round(4))
        
        # CSVへの保存
        out_loadings_path = OUT_DIR / f"factor_loadings_{n_f}factors.csv"
        loadings.to_csv(out_loadings_path, encoding="utf-8-sig")
        print(f"\n✅ 因子負荷量を保存しました: {out_loadings_path}")
        
        # 因子負荷量のヒートマップ
        fig, ax = plt.subplots(figsize=(max(4, 2 + n_f * 2), 8))
        sns.heatmap(
            loadings,
            annot=True,
            fmt=".3f",
            cmap="RdYlBu_r",
            center=0,
            vmin=-1.0,
            vmax=1.0,
            cbar_kws={"label": "因子負荷量"},
            linewidths=0.5,
            ax=ax
        )
        ax.set_title(f"因子負荷量 ({n_f}因子, {rotation_label})", fontsize=13, pad=15)
        plt.tight_layout()
        heatmap_path = OUT_DIR / f"factor_loadings_heatmap_{n_f}factors.png"
        fig.savefig(heatmap_path, dpi=150)
        plt.close(fig)
        print(f"✅ 因子負荷量ヒートマップを保存しました: {heatmap_path}")
        
        # 2因子の場合のみ: 2次元因子空間プロット
        if n_f == 2:
            fig, ax = plt.subplots(figsize=(8, 8))
            f1 = loadings.iloc[:, 0].values
            f2 = loadings.iloc[:, 1].values
            
            ax.scatter(f1, f2, color="#3498db", s=150, edgecolors="black", zorder=5)
            
            for i, item in enumerate(items):
                ax.annotate(
                    item,
                    (f1[i], f2[i]),
                    textcoords="offset points",
                    xytext=(10, 5),
                    fontsize=11,
                    fontweight="bold"
                )
            
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlim(min(f1) - 0.2, max(f1) + 0.4)
            ax.set_ylim(min(f2) - 0.2, max(f2) + 0.4)
            ax.set_xlabel("第1因子への負荷量", fontsize=12)
            ax.set_ylabel("第2因子への負荷量", fontsize=12)
            ax.set_title("2次元因子空間における質問項目のマッピング\n(参考: カイザー基準による2因子モデル)", fontsize=14, pad=15)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            scatter_path = OUT_DIR / "factor_space_plot.png"
            fig.savefig(scatter_path, dpi=150)
            plt.close(fig)
            print(f"✅ 2次元因子空間プロットを保存しました: {scatter_path}")
    
    # ── 3. 結論のサマリー出力 ──
    print(f"\n{'='*60}")
    print("  因子数決定の総合判断")
    print(f"{'='*60}")
    print(f"  カイザー基準 (固有値>1):     {kaiser_n} 因子")
    print(f"  平行分析 (95%tile基準):      {pa_n_factors} 因子")
    print(f"  累積分散説明率 (1因子):      {ev[0]/sum(ev)*100:.1f}%")
    if len(ev) >= 2:
        print(f"  累積分散説明率 (2因子):      {sum(ev[:2])/sum(ev)*100:.1f}%")
    print()
    
    if pa_n_factors < kaiser_n:
        print("  ⚠️ 平行分析はカイザー基準より保守的な結果を示しています。")
        print(f"     N={len(X)} は変数数({len(items)})に対して小さく（推奨: 40以上）、")
        print(f"     第2因子の固有値 ({ev[1]:.4f}) がランダムデータの95%tile ({random_ev_95[1]:.4f}) を")
        print(f"     下回っています。サンプルサイズを増やすことで第2因子が有意になる可能性があります。")
        print()
        print("  → 現時点では「1因子（総合的な動画体験品質）」が統計的に支持されますが、")
        print("    相関行列のパターンや理論的背景から2因子構造を仮説として論文に記述し、")
        print("    今後のデータ拡充によって検証する、というアプローチが推奨されます。")

if __name__ == "__main__":
    run_factor_analysis()

