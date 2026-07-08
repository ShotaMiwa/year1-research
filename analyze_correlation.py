#!/usr/bin/env python3
"""
ライブ配信動画アンケート × ヒートマップ値 相関分析スクリプト

機能:
  1. yt-dlp でヒートマップ生データを取得し data/raw_heatmap.csv に保存（キャッシュ）
  2. アンケートCSV のリッカート尺度を数値化し、セグメント別平均スコアを算出
  3. セグメント別ヒートマップ統計値とアンケート回答を結合
  4. 3種の相関分析 + 散布図 + 4グループ分類
"""

import os
import sys
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# ── 日本語フォント設定 ─────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import japanize_matplotlib

plt.rcParams["axes.unicode_minus"] = False

import seaborn as sns

# ═══════════════════════════════════════════════════════════
# 設定
# ═══════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "analysis_questionnaire"
SURVEY_DIR = PROJECT_ROOT / "ライブ配信動画に関する予備調査アンケート"

VIDEO_URL = "https://www.youtube.com/watch?v=pP2KLW-_7hQ"
RAW_HEATMAP_PATH = DATA_DIR / "raw_heatmap.csv"
HEATMAP_SEGMENTS_PATH = DATA_DIR / "heatmap_segments.csv"
MERGED_PATH = DATA_DIR / "merged_survey_heatmap.csv"
SURVEY_CSV = SURVEY_DIR / "ライブ配信動画に関する予備調査アンケート.csv"

# アンケートの「セグメント1〜23」に対応する時間帯
# topicsegment.txt の64個 → 30秒以上フィルタ → 46個 のうち最初の23個
# (notebook output から確認済み)
SURVEY_SEGMENTS = [
    (1,  "0:01:49", "0:03:44"),
    (2,  "0:03:46", "0:06:00"),
    (3,  "0:06:56", "0:08:47"),
    (4,  "0:08:47", "0:09:51"),
    (5,  "0:10:07", "0:11:49"),
    (6,  "0:11:50", "0:14:22"),
    (7,  "0:14:24", "0:15:30"),
    (8,  "0:16:10", "0:17:36"),
    (9,  "0:17:36", "0:18:38"),
    (10, "0:19:04", "0:20:00"),
    (11, "0:20:01", "0:20:40"),
    (12, "0:20:53", "0:21:57"),
    (13, "0:21:59", "0:22:46"),
    (14, "0:22:47", "0:23:21"),
    (15, "0:23:47", "0:26:08"),
    (16, "0:26:10", "0:27:07"),
    (17, "0:27:19", "0:27:58"),
    (18, "0:28:34", "0:32:21"),
    (19, "0:32:41", "0:33:48"),
    (20, "0:33:51", "0:34:27"),
    (21, "0:34:30", "0:36:42"),
    (22, "0:37:03", "0:37:40"),
    (23, "0:37:59", "0:38:40"),
]

# リッカート尺度のマッピング（テキスト→数値）
LIKERT_MAP = {
    "非常にそう思う":     5,
    "ややそう思う":       4,
    "どちらでもない":     3,
    "あまりそう思わない": 2,
    "全くそう思わない":   1,
}

# 評価項目（9項目）の短縮名
ITEM_NAMES = [
    "面白かった",
    "話に引き込まれた",
    "テンポが良かった",
    "新しい情報を得られた",
    "有益な内容だった",
    "誰かに共有したいと思った",
    "気軽に視聴できた",
    "気分転換になった",
    "この動画に満足した",
]

# 項目のカテゴリ
ITEM_CATEGORIES = {
    "面白かった":             "娯楽性",
    "話に引き込まれた":       "娯楽性",
    "テンポが良かった":       "娯楽性",
    "新しい情報を得られた":   "情報性",
    "有益な内容だった":       "情報性",
    "誰かに共有したいと思った": "社会性",
    "気軽に視聴できた":       "リラックス",
    "気分転換になった":       "リラックス",
    "この動画に満足した":     "総合満足度",
}


# ═══════════════════════════════════════════════════════════
# ユーティリティ
# ═══════════════════════════════════════════════════════════
def ts_to_sec(ts: str) -> float:
    """'H:MM:SS' または 'M:SS' を秒数に変換"""
    parts = ts.strip().split(":")
    parts = [int(p) for p in parts]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return float(parts[0])


def sec_to_ts(sec: float) -> str:
    """秒数を 'H:MM:SS' に変換"""
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h}:{m:02d}:{s:02d}"


# ═══════════════════════════════════════════════════════════
# Step 1: ヒートマップデータの取得とローカル保存
# ═══════════════════════════════════════════════════════════
def fetch_and_save_heatmap():
    """yt-dlp でヒートマップ生データを取得し CSV に保存する"""
    if RAW_HEATMAP_PATH.exists():
        print(f"✅ キャッシュ済み: {RAW_HEATMAP_PATH}")
        df = pd.read_csv(RAW_HEATMAP_PATH)
        print(f"   {len(df)} 行読み込み")
        return df

    print("📥 yt-dlp でヒートマップデータを取得中...")
    import yt_dlp

    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        info = ydl.extract_info(VIDEO_URL, download=False)

    heatmap = info.get("heatmap")
    if not heatmap:
        raise RuntimeError("❌ この動画にはヒートマップデータがありません")

    title = info.get("title", "不明")
    duration = info.get("duration", 0)
    print(f"   タイトル : {title}")
    print(f"   動画時間 : {sec_to_ts(duration)}")
    print(f"   ヒートマップ区間数: {len(heatmap)}")

    rows = []
    for idx, h in enumerate(heatmap, 1):
        rows.append({
            "区間番号":   idx,
            "開始時刻":   sec_to_ts(h["start_time"]),
            "終了時刻":   sec_to_ts(h["end_time"]),
            "開始(秒)":   round(h["start_time"], 4),
            "終了(秒)":   round(h["end_time"], 4),
            "長さ(秒)":   round(h["end_time"] - h["start_time"], 4),
            "value":      h["value"],
        })

    df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_HEATMAP_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ 保存完了: {RAW_HEATMAP_PATH} ({len(df)} 行)")
    return df


def calc_segment_heatmap_stats(df_heatmap):
    """各アンケートセグメントのヒートマップ統計値を集計する"""
    results = []
    for seg_num, start_ts, end_ts in SURVEY_SEGMENTS:
        seg_start = ts_to_sec(start_ts)
        seg_end = ts_to_sec(end_ts)
        duration = seg_end - seg_start

        # ヒートマップ区間との重なりを計算
        values = []
        weights = []
        for _, row in df_heatmap.iterrows():
            h_start = row["開始(秒)"]
            h_end = row["終了(秒)"]
            overlap = max(0.0, min(seg_end, h_end) - max(seg_start, h_start))
            if overlap > 0:
                values.append(row["value"])
                weights.append(overlap)

        if values:
            values_arr = np.array(values)
            weights_arr = np.array(weights)
            mean_val = np.average(values_arr, weights=weights_arr)
            max_val = values_arr.max()
        else:
            mean_val = np.nan
            max_val = np.nan

        results.append({
            "セグメント":        seg_num,
            "開始時刻":          start_ts,
            "終了時刻":          end_ts,
            "開始(秒)":          int(seg_start),
            "終了(秒)":          int(seg_end),
            "長さ(秒)":          int(duration),
            "平均ヒートマップ値": round(mean_val, 4) if not np.isnan(mean_val) else None,
            "最大ヒートマップ値": round(max_val, 4) if not np.isnan(max_val) else None,
        })

    df_seg = pd.DataFrame(results)
    df_seg.to_csv(HEATMAP_SEGMENTS_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ セグメント別ヒートマップ集計を保存: {HEATMAP_SEGMENTS_PATH}")
    return df_seg


# ═══════════════════════════════════════════════════════════
# Step 2: アンケートデータの前処理
# ═══════════════════════════════════════════════════════════
def process_survey():
    """アンケートCSVを読み込み、リッカート尺度を数値化し、
    セグメント×項目ごとの平均スコアを返す"""
    print(f"\n📊 アンケートCSV読み込み: {SURVEY_CSV}")
    df = pd.read_csv(SURVEY_CSV, encoding="utf-8")
    print(f"   回答者数: {len(df)}")

    # ヘッダーからセグメント列を特定
    cols = df.columns.tolist()

    # セグメントごと、項目ごとの平均スコアを集計
    segment_scores = []
    for seg_num in range(1, 24):
        seg_label = f"セグメント{seg_num}"
        seg_data = {"セグメント": seg_num}

        for item_idx, item_name in enumerate(ITEM_NAMES, 1):
            # カラムの部分一致で特定
            matching_cols = [
                c for c in cols
                if seg_label in c and item_name in c
            ]
            if not matching_cols:
                print(f"  ⚠️ {seg_label} / {item_name} に一致する列が見つかりません")
                seg_data[item_name] = np.nan
                continue

            col = matching_cols[0]
            # リッカート尺度を数値に変換
            numeric_vals = df[col].map(LIKERT_MAP)
            unmapped = df[col][numeric_vals.isna()]
            if len(unmapped) > 0:
                unique_unmapped = unmapped.unique()
                if len(unique_unmapped) > 0 and not all(pd.isna(unique_unmapped)):
                    print(f"  ⚠️ マッピングできない値: {unique_unmapped} (列: {col[:50]}...)")

            seg_data[item_name] = round(numeric_vals.mean(), 4)

        segment_scores.append(seg_data)

    df_scores = pd.DataFrame(segment_scores)
    print(f"✅ セグメント×項目 平均スコア算出完了 ({len(df_scores)} セグメント × {len(ITEM_NAMES)} 項目)")
    return df_scores


# ═══════════════════════════════════════════════════════════
# Step 3: データの結合
# ═══════════════════════════════════════════════════════════
def merge_data(df_heatmap_seg, df_survey):
    """ヒートマップ統計値とアンケート回答を結合"""
    df_merged = pd.merge(df_heatmap_seg, df_survey, on="セグメント", how="inner")
    df_merged.to_csv(MERGED_PATH, index=False, encoding="utf-8-sig")
    print(f"\n✅ マージデータ保存: {MERGED_PATH} ({len(df_merged)} 行)")
    return df_merged


# ═══════════════════════════════════════════════════════════
# Step 4: 相関分析と可視化
# ═══════════════════════════════════════════════════════════
def compute_correlations(df_merged):
    """3種の相関分析を実施"""
    satisfaction_col = "この動画に満足した"
    heatmap_mean_col = "平均ヒートマップ値"
    heatmap_max_col = "最大ヒートマップ値"
    items_1_8 = ITEM_NAMES[:8]

    results = []

    # ── ①  総合満足度 × ヒートマップ値 ──
    for hm_col in [heatmap_mean_col, heatmap_max_col]:
        x = df_merged[satisfaction_col].dropna()
        y = df_merged[hm_col].dropna()
        common = x.index.intersection(y.index)
        x, y = x[common], y[common]

        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)
        results.append({
            "分析カテゴリ": "① 総合満足度 × ヒートマップ",
            "項目X": satisfaction_col,
            "項目Y": hm_col,
            "Pearson r": round(pearson_r, 4),
            "Pearson p": round(pearson_p, 4),
            "Spearman ρ": round(spearman_r, 4),
            "Spearman p": round(spearman_p, 4),
            "N": len(common),
        })

    # ── ② 項目1〜8 × 総合満足度 ──
    for item in items_1_8:
        x = df_merged[item].dropna()
        y = df_merged[satisfaction_col].dropna()
        common = x.index.intersection(y.index)
        x, y = x[common], y[common]

        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)
        results.append({
            "分析カテゴリ": "② 各項目 × 総合満足度",
            "項目X": item,
            "項目Y": satisfaction_col,
            "Pearson r": round(pearson_r, 4),
            "Pearson p": round(pearson_p, 4),
            "Spearman ρ": round(spearman_r, 4),
            "Spearman p": round(spearman_p, 4),
            "N": len(common),
        })

    # ── ③ 項目1〜8 × ヒートマップ値 ──
    for item in items_1_8:
        for hm_col in [heatmap_mean_col, heatmap_max_col]:
            x = df_merged[item].dropna()
            y = df_merged[hm_col].dropna()
            common = x.index.intersection(y.index)
            x, y = x[common], y[common]

            pearson_r, pearson_p = stats.pearsonr(x, y)
            spearman_r, spearman_p = stats.spearmanr(x, y)
            results.append({
                "分析カテゴリ": "③ 各項目 × ヒートマップ",
                "項目X": item,
                "項目Y": hm_col,
                "Pearson r": round(pearson_r, 4),
                "Pearson p": round(pearson_p, 4),
                "Spearman ρ": round(spearman_r, 4),
                "Spearman p": round(spearman_p, 4),
                "N": len(common),
            })

    df_corr = pd.DataFrame(results)
    corr_path = DATA_DIR / "correlation_results.csv"
    df_corr.to_csv(corr_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 相関分析結果を保存: {corr_path}")

    # ── 結果表示 ──
    for category in ["① 総合満足度 × ヒートマップ", "② 各項目 × 総合満足度", "③ 各項目 × ヒートマップ"]:
        print(f"\n{'='*60}")
        print(f"  {category}")
        print(f"{'='*60}")
        subset = df_corr[df_corr["分析カテゴリ"] == category]
        for _, row in subset.iterrows():
            p_sig = "***" if row["Pearson p"] < 0.001 else "**" if row["Pearson p"] < 0.01 else "*" if row["Pearson p"] < 0.05 else ""
            s_sig = "***" if row["Spearman p"] < 0.001 else "**" if row["Spearman p"] < 0.01 else "*" if row["Spearman p"] < 0.05 else ""
            print(f"  {row['項目X']:20s} × {row['項目Y']:15s} | "
                  f"Pearson r={row['Pearson r']:+.4f} (p={row['Pearson p']:.4f}){p_sig:3s} | "
                  f"Spearman ρ={row['Spearman ρ']:+.4f} (p={row['Spearman p']:.4f}){s_sig}")

    return df_corr


def plot_correlation_heatmap(df_merged):
    """相関行列ヒートマップを作成（ピアソンおよびスピアマン両方）"""
    analysis_cols = ITEM_NAMES + ["平均ヒートマップ値", "最大ヒートマップ値"]
    df_analysis = df_merged[analysis_cols].dropna()

    # 短縮ラベル
    short_labels = {
        "面白かった":             "1.面白い",
        "話に引き込まれた":       "2.引き込み",
        "テンポが良かった":       "3.テンポ",
        "新しい情報を得られた":   "4.新情報",
        "有益な内容だった":       "5.有益",
        "誰かに共有したいと思った": "6.共有",
        "気軽に視聴できた":       "7.気軽",
        "気分転換になった":       "8.気分転換",
        "この動画に満足した":     "9.満足度",
        "平均ヒートマップ値":     "HM平均",
        "最大ヒートマップ値":     "HM最大",
    }
    df_analysis = df_analysis.rename(columns=short_labels)

    for method in ["pearson", "spearman"]:
        corr_matrix = df_analysis.corr(method=method)

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            ax=ax,
            annot_kws={"size": 9},
        )
        title_method = "ピアソン" if method == "pearson" else "スピアマン"
        ax.set_title(f"アンケート評価項目 × ヒートマップ値 相関行列 ({title_method})", fontsize=14, pad=15)
        plt.tight_layout()

        out_path = DATA_DIR / f"correlation_matrix_{method}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ 相関行列ヒートマップ画像 ({title_method}) を保存: {out_path}")


def plot_scatter_and_groups(df_merged):
    """散布図（満足度 × ヒートマップ平均・最大）と4グループ分類"""
    satisfaction_col = "この動画に満足した"
    y = df_merged[satisfaction_col]
    y_med = y.median()

    # 平均ヒートマップ値と最大ヒートマップ値のループ
    for label, col_name, file_suffix in [("平均", "平均ヒートマップ値", "mean"), ("最大", "最大ヒートマップ値", "max")]:
        x = df_merged[col_name]
        x_med = x.median()

        def classify(row):
            hm = row[col_name]
            sat = row[satisfaction_col]
            if hm >= x_med and sat >= y_med:
                return "1: HM高×満足高"
            elif hm >= x_med and sat < y_med:
                return "2: HM高×満足低"
            elif hm < x_med and sat < y_med:
                return "3: HM低×満足低"
            else:
                return "4: HM低×満足高"

        df_merged[f"グループ_{label}"] = df_merged.apply(classify, axis=1)

        # 散布図描画
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = {
            "1: HM高×満足高": "#2ecc71",
            "2: HM高×満足低": "#e74c3c",
            "3: HM低×満足低": "#95a5a6",
            "4: HM低×満足高": "#3498db",
        }

        for group, color in colors.items():
            mask = df_merged[f"グループ_{label}"] == group
            ax.scatter(
                df_merged.loc[mask, col_name],
                df_merged.loc[mask, satisfaction_col],
                c=color,
                label=group,
                s=100,
                edgecolors="white",
                linewidth=1.5,
                zorder=5,
            )
            # セグメント番号をラベル付け
            for _, row in df_merged[mask].iterrows():
                ax.annotate(
                    str(int(row["セグメント"])),
                    (row[col_name], row[satisfaction_col]),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=8,
                    color=color,
                    fontweight="bold",
                )

        # 中央値の線
        ax.axvline(x_med, color="gray", linestyle="--", alpha=0.5, label=f"HM{label}中央値={x_med:.4f}")
        ax.axhline(y_med, color="gray", linestyle=":",  alpha=0.5, label=f"満足度中央値={y_med:.2f}")

        ax.set_xlabel(col_name, fontsize=12)
        ax.set_ylabel("総合満足度（平均スコア）", fontsize=12)
        ax.set_title(f"セグメント別: {col_name} × 総合満足度 散布図", fontsize=14, pad=15)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = DATA_DIR / f"scatter_heatmap_{file_suffix}_satisfaction.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ 散布図 ({label}ヒートマップ値) を保存: {out_path}")

        # グループ別統計表示
        print(f"\n{'='*60}")
        print(f"  4グループ分類結果 ({label}値ベース)")
        print(f"{'='*60}")
        print(f"  中央値: ヒートマップ={x_med:.4f}, 満足度={y_med:.2f}")
        print()
        for group in sorted(colors.keys()):
            mask = df_merged[f"グループ_{label}"] == group
            segs = df_merged.loc[mask, "セグメント"].tolist()
            count = len(segs)
            print(f"  {group} ({count}件): セグメント {segs}")
            if count > 0:
                sub = df_merged[mask]
                print(f"    平均{col_name}: {sub[col_name].mean():.4f}")
                print(f"    平均満足度:         {sub[satisfaction_col].mean():.2f}")

    # 下位互換のために元の「グループ」カラムに「グループ_平均」の値を代入しておく
    df_merged["グループ"] = df_merged["グループ_平均"]

    # グループ情報をCSVに保存
    df_merged.to_csv(MERGED_PATH, index=False, encoding="utf-8-sig")

    return df_merged


def analyze_clustering(df_merged):
    """ヒートマップ値と満足度スコアに対するクラスタリング分析（階層的 & K-Means）"""
    print("\n── クラスタリング分析の実行 ─────────────────────────")
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

    satisfaction_col = "この動画に満足した"
    colors_list = ["#2ecc71", "#3498db", "#e74c3c", "#f1c40f", "#9b59b6", "#e67e22"]

    # 分析対象のペア (平均値ベース, 最大値ベース)
    for label, col_name, file_suffix in [("平均", "平均ヒートマップ値", "mean"), ("最大", "最大ヒートマップ値", "max")]:
        print(f"\n[ {label}ヒートマップ値ベースのクラスタリング ]")
        
        # データの抽出と標準化
        features = df_merged[[col_name, satisfaction_col]].copy()
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # 1. 階層的クラスタリング（ウォード法）のデンドログラム描画
        fig, ax = plt.subplots(figsize=(10, 6))
        Z = linkage(scaled_features, method="ward")
        dendrogram(
            Z,
            labels=df_merged["セグメント"].values,
            color_threshold=None,
            ax=ax
        )
        ax.set_title(f"階層的クラスタリング デンドログラム ({label}値ベース)", fontsize=14, pad=15)
        ax.set_xlabel("セグメント番号", fontsize=12)
        ax.set_ylabel("結合距離（ウォード法）", fontsize=12)
        plt.tight_layout()
        dendrogram_path = DATA_DIR / f"dendrogram_{file_suffix}.png"
        fig.savefig(dendrogram_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  - デンドログラムを保存: {dendrogram_path}")

        # 2. K-Means のエルボー法
        sse = []
        k_list = range(1, min(11, len(df_merged)))
        for k in k_list:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_features)
            sse.append(kmeans.inertia_)
            
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(k_list, sse, marker="o", linestyle="-", color="#2c3e50")
        ax.set_title(f"K-Means エルボー曲線 ({label}値ベース)", fontsize=14, pad=15)
        ax.set_xlabel("クラスタ数 K", fontsize=12)
        ax.set_ylabel("インinertia (SSE)", fontsize=12)
        ax.set_xticks(k_list)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        elbow_path = DATA_DIR / f"elbow_curve_{file_suffix}.png"
        fig.savefig(elbow_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  - エルボー曲線を保存: {elbow_path}")

        # 3. 代表的なクラスタ数 (K=3, 4) でのクラスタリング適用とプロット
        for k_clusters in [3, 4]:
            # 階層的クラスタリングのクラスタ割り当て
            hc_labels = fcluster(Z, t=k_clusters, criterion="maxclust")
            
            # K-Meansのクラスタ割り当て
            kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
            km_labels = kmeans.fit_predict(scaled_features) + 1  # 1-indexed
            
            # グラフプロット (階層的クラスタリング結果)
            fig, ax = plt.subplots(figsize=(10, 8))
            unique_labels = sorted(list(set(hc_labels)))
            
            for i, cluster_id in enumerate(unique_labels):
                mask = hc_labels == cluster_id
                color = colors_list[i % len(colors_list)]
                ax.scatter(
                    features.loc[mask, col_name],
                    features.loc[mask, satisfaction_col],
                    label=f"クラスタ {cluster_id}",
                    c=color,
                    s=120,
                    edgecolors="white",
                    linewidth=1.5,
                    zorder=5
                )
                # セグメント番号のラベル付け
                for _, row in df_merged[mask].iterrows():
                    ax.annotate(
                        str(int(row["セグメント"])),
                        (row[col_name], row[satisfaction_col]),
                        textcoords="offset points",
                        xytext=(6, 6),
                        fontsize=9,
                        color=color,
                        fontweight="bold"
                    )
            
            ax.set_xlabel(col_name, fontsize=12)
            ax.set_ylabel("総合満足度（平均スコア）", fontsize=12)
            ax.set_title(f"セグメント別: {col_name} × 総合満足度\n(階層的クラスタリング Ward法, K={k_clusters})", fontsize=14, pad=15)
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            hc_scatter_path = DATA_DIR / f"scatter_hc_{file_suffix}_k{k_clusters}.png"
            fig.savefig(hc_scatter_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  - 階層的 (K={k_clusters}) 散布図を保存: {hc_scatter_path}")
            
            # グラフプロット (K-Means結果)
            fig, ax = plt.subplots(figsize=(10, 8))
            unique_labels_km = sorted(list(set(km_labels)))
            
            for i, cluster_id in enumerate(unique_labels_km):
                mask = km_labels == cluster_id
                color = colors_list[i % len(colors_list)]
                ax.scatter(
                    features.loc[mask, col_name],
                    features.loc[mask, satisfaction_col],
                    label=f"クラスタ {cluster_id}",
                    c=color,
                    s=120,
                    edgecolors="white",
                    linewidth=1.5,
                    zorder=5
                )
                # セグメント番号のラベル付け
                for _, row in df_merged[mask].iterrows():
                    ax.annotate(
                        str(int(row["セグメント"])),
                        (row[col_name], row[satisfaction_col]),
                        textcoords="offset points",
                        xytext=(6, 6),
                        fontsize=9,
                        color=color,
                        fontweight="bold"
                    )
            
            # K-Means セントロイドの描画
            centroids_scaled = kmeans.cluster_centers_
            centroids = scaler.inverse_transform(centroids_scaled)
            ax.scatter(
                centroids[:, 0],
                centroids[:, 1],
                marker="X",
                s=200,
                color="black",
                label="セントロイド(重心)",
                zorder=10
            )
            
            ax.set_xlabel(col_name, fontsize=12)
            ax.set_ylabel("総合満足度（平均スコア）", fontsize=12)
            ax.set_title(f"セグメント別: {col_name} × 総合満足度\n(K-Meansクラスタリング, K={k_clusters})", fontsize=14, pad=15)
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            km_scatter_path = DATA_DIR / f"scatter_kmeans_{file_suffix}_k{k_clusters}.png"
            fig.savefig(km_scatter_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  - K-Means (K={k_clusters}) 散布図を保存: {km_scatter_path}")
            
            # 結果を CSV 用に保存
            df_merged[f"クラスタ_階層_{label}_K{k_clusters}"] = hc_labels
            df_merged[f"クラスタ_KMeans_{label}_K{k_clusters}"] = km_labels

    # 結合データの更新保存
    df_merged.to_csv(MERGED_PATH, index=False, encoding="utf-8-sig")
    print(f"\n✅ クラスタリング結果を結合したCSVを保存: {MERGED_PATH}")
    return df_merged


# ═══════════════════════════════════════════════════════════
# メイン
# ═══════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  ライブ配信動画アンケート × ヒートマップ値 相関分析")
    print("=" * 70)

    # Step 1: ヒートマップデータ
    print("\n── Step 1: ヒートマップデータの取得 ─────────────────")
    df_heatmap = fetch_and_save_heatmap()
    df_heatmap_seg = calc_segment_heatmap_stats(df_heatmap)
    print("\nセグメント別ヒートマップ統計:")
    print(df_heatmap_seg.to_string(index=False))

    # Step 2: アンケートデータ
    print("\n── Step 2: アンケートデータの前処理 ─────────────────")
    df_survey = process_survey()
    print("\nセグメント別平均スコア (先頭5行):")
    print(df_survey.head().to_string(index=False))

    # Step 3: データ結合
    print("\n── Step 3: データの結合 ─────────────────────────────")
    df_merged = merge_data(df_heatmap_seg, df_survey)

    # Step 4: 相関分析と可視化
    print("\n── Step 4: 相関分析と可視化 ─────────────────────────")
    df_corr = compute_correlations(df_merged)
    plot_correlation_heatmap(df_merged)
    df_merged = plot_scatter_and_groups(df_merged)
    df_merged = analyze_clustering(df_merged)

    print("\n" + "=" * 70)
    print("  分析完了！")
    print("=" * 70)
    print(f"\n生成されたファイル:")
    for p in [RAW_HEATMAP_PATH, HEATMAP_SEGMENTS_PATH, MERGED_PATH,
              DATA_DIR / "correlation_results.csv",
              DATA_DIR / "correlation_matrix_pearson.png",
              DATA_DIR / "correlation_matrix_spearman.png",
              DATA_DIR / "scatter_heatmap_mean_satisfaction.png",
              DATA_DIR / "scatter_heatmap_max_satisfaction.png",
              DATA_DIR / "dendrogram_mean.png",
              DATA_DIR / "elbow_curve_mean.png",
              DATA_DIR / "scatter_hc_mean_k3.png",
              DATA_DIR / "scatter_hc_mean_k4.png",
              DATA_DIR / "scatter_kmeans_mean_k3.png",
              DATA_DIR / "scatter_kmeans_mean_k4.png"]:
        exists = "✅" if p.exists() else "❌"
        print(f"  {exists} {p}")


if __name__ == "__main__":
    main()
