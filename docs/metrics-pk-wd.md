# 評価指標 PK と WindowDiff（WD）の仕様

テキスト・対話セグメンテーションの評価で使っている **Pk** と **WindowDiff（WD）** の計算方法と仕様をまとめます。

---

## 1. 前提：境界とセグメントの表現

- **境界（boundary）**: 各発話（文）の直後が「セグメントの切れ目かどうか」を 0/1 で表したリスト。
  - 例: `[0, 0, 1, 0, 1]` → 2番目と4番目の発話の直後に境界がある。
- **セグメント長リスト**: 境界リストから「各セグメントの長さ」のリストに変換したもの。
  - 例: 上記なら `[3, 2]`（1つ目は長さ3、2つ目は長さ2）。

このプロジェクトでは、予測・正解とも **境界の 0/1 リスト** で持ち、PK/WD 計算の直前に **セグメント長リスト** に変換してから `segeval` に渡しています（`_boundaries_to_segments`）。

---

## 2. Pk（Pk スコア）

### 概要

- **出典**: Beeferman et al. (1999) のセグメンテーション評価指標。
- **意味**: 「スライディングウィンドウごとに、予測と正解で『ウィンドウ内の境界数』が一致しているか」を見て、不一致の割合を出したもの。
- **範囲**: 0～1。**低いほど良い**（0 が完全一致）。

### 計算の流れ

1. **ウィンドウサイズ k** を決める。  
   このプロジェクトでは **正解セグメントの平均長の 1/2** を使う（`window_size = int(avg_segment_length / 2)`）。  
   指定しない場合は `segeval` のデフォルトに依存。
2. 発話列上で、**長さ k のウィンドウ** を先頭から 1 ずつずらして考える。
3. 各ウィンドウについて:
   - 正解の境界リストから「そのウィンドウ内の境界の数」を数える。
   - 予測の境界リストから「そのウィンドウ内の境界の数」を数える。
   - この **2つの数が異なれば、そのウィンドウは誤り** とみなす。
4. **Pk = （誤りと判定したウィンドウの数） / （ウィンドウの総数）**。

境界の「位置」そのものではなく、「ウィンドウ内に何個境界があるか」が一致しているかを重視する指標です。

### このプロジェクトでの呼び出し

- **定義・計算**: `src/evaluation/metrics.py` の `MetricsCalculator.calculate_pk`
- **入力**: 予測境界リスト、正解境界リスト、オプションで `window_size`
- **内部**: 境界リスト → セグメント長リストに変換したうえで `segeval.pk(seg_pred, seg_gold, window_size=window_size)` を呼び、その戻り値を返す。

```16:39:src/evaluation/metrics.py
def calculate_pk(
    predicted_boundaries: List[int],
    gold_boundaries: List[int],
    window_size: int = None
) -> float:
    ...
    seg_pred = MetricsCalculator._boundaries_to_segments(predicted_boundaries)
    seg_gold = MetricsCalculator._boundaries_to_segments(gold_boundaries)
    pk = segeval.pk(seg_pred, seg_gold, window_size=window_size)
    return float(pk)
```

---

## 3. WindowDiff（WD）

### 概要

- **出典**: Pevzner & Hearst (2002)。
- **意味**: Pk と同様、**スライディングウィンドウ** で「予測と正解でウィンドウ内の境界数が一致しているか」を判定し、不一致の割合を出す。
- **範囲**: 0～1。**低いほど良い**（0 が完全一致）。

### 計算の流れ（Pk との違い）

- ウィンドウの取り方や「境界数が一致しなければ誤り」という考え方は Pk に近い。
- 違いは主に **ウィンドウの長さの解釈** や **境界の数え方** にあり、文献によっては「ウィンドウ幅を k+1 にする」などと説明される。  
  実装の詳細は **segeval の `window_diff`** に依存する。
- このプロジェクトでは、Pk と同様に **同じ `window_size`**（正解の平均セグメント長の 1/2）を渡して計算している。

### このプロジェクトでの呼び出し

- **定義・計算**: `src/evaluation/metrics.py` の `MetricsCalculator.calculate_window_diff`
- **入力**: 予測境界リスト、正解境界リスト、オプションで `window_size`
- **内部**: 境界リスト → セグメント長リストに変換したうえで `segeval.window_diff(seg_pred, seg_gold, window_size=window_size)` を呼び、その戻り値を返す。

```41:65:src/evaluation/metrics.py
def calculate_window_diff(
    predicted_boundaries: List[int],
    gold_boundaries: List[int],
    window_size: int = None
) -> float:
    ...
    seg_pred = MetricsCalculator._boundaries_to_segments(predicted_boundaries)
    seg_gold = MetricsCalculator._boundaries_to_segments(gold_boundaries)
    wd = segeval.window_diff(seg_pred, seg_gold, window_size=window_size)
    return float(wd)
```

---

## 4. 境界リスト → セグメント長リストの変換

PK/WD の前に、境界の 0/1 リストを **セグメント長のリスト** にしています。

- **入力**: `boundaries` — 各位置が境界なら 1、そうでなければ 0。
- **処理**: 先頭から見て、次の 1 が出るまでを 1 セグメントとして長さを数え、1 のたびにその長さをリストに追加。最後に残った区間も 1 セグメントとして追加。
- **実装**: `MetricsCalculator._boundaries_to_segments`（metrics.py 115–139 行目付近）。

例: `[0, 0, 1, 0, 1]` → 長さ 3 のセグメント、長さ 2 のセグメント → `[3, 2]`。

---

## 5. ウィンドウサイズの決め方（一括計算時）

`calculate_all_metrics` では、PK と WD の両方に **同じウィンドウサイズ** を使います。

- 正解境界からセグメント長リスト `seg_gold` を作成。
- **平均セグメント長** `avg_segment_length = np.mean(seg_gold)` を計算。
- **window_size = int(avg_segment_length / 2)** とし、これを `calculate_pk` と `calculate_window_diff` に渡す。

長いセグメントが多い対話ではウィンドウが大きくなり、短いセグメントが多いと小さくなります。

---

## 6. まとめ（他人に説明するとき）

- **Pk**: スライディングウィンドウで「予測と正解の、ウィンドウ内の境界数」を比較し、不一致だったウィンドウの割合を取った指標。0～1、低いほど良い。Beeferman et al. (1999)。
- **WindowDiff（WD）**: 同様にスライディングウィンドウで境界数の一致を見る指標。0～1、低いほど良い。Pevzner & Hearst (2002)。
- **このプロジェクト**: 予測・正解は境界の 0/1 リストで保持し、`_boundaries_to_segments` でセグメント長リストに変換したうえで、`segeval.pk` / `segeval.window_diff` を呼び出している。ウィンドウサイズは指定しない場合は自動（一括計算時は正解の平均セグメント長の 1/2）。
