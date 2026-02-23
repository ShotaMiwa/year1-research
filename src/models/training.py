"""
学習ロジックモジュール
モデルの学習処理を管理
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import bisect
import numpy as np
from typing import Dict, Optional, Tuple

from models.architecture import SegmentationModel
from utils.losses import MarginRankingLoss
from utils.depth_score import DepthScoreCalculator


class TrainingWrapper(nn.Module):
    """
    学習処理を管理するラッパークラス
    """
    
    def __init__(
        self,
        model: SegmentationModel,
        margin: int = 1,
        train_split: int = 5,
        window_size: int = 5
    ):
        """
        Args:
            model: セグメンテーションモデル
            margin: マージンランキング損失のマージン
            train_split: 学習時の分割数
            window_size: ウィンドウサイズ
        """
        super().__init__()
        
        self.model = model
        self.margin = margin
        self.train_split = train_split
        self.window_size = window_size
        
        # 損失関数
        self.topic_loss_fn = nn.CrossEntropyLoss()
        self.score_loss_fn = MarginRankingLoss(margin)
        self.depth_calculator = DepthScoreCalculator()
    
    def forward(
        self,
        input_data: Dict[str, torch.Tensor],
        window_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        学習時のフォワードパス
        
        Args:
            input_data: 入力データの辞書
            window_size: ウィンドウサイズ（指定時のみ上書き）
            
        Returns:
            (total_loss, margin_loss, topic_loss)のタプル
        """
        device = input_data['coheren_inputs'].device
        
        # Coherenceスコア計算
        coheren_pos_scores, _ = self.model.encode_coherence(
            input_data['coheren_inputs'][:, 0, :],
            attention_mask=input_data['coheren_mask'][:, 0, :],
            token_type_ids=input_data['coheren_type'][:, 0, :]
        )
        
        coheren_neg_scores, _ = self.model.encode_coherence(
            input_data['coheren_inputs'][:, 1, :],
            attention_mask=input_data['coheren_mask'][:, 1, :],
            token_type_ids=input_data['coheren_type'][:, 1, :]
        )
        
        batch_size = len(input_data['topic_context_num'])
        
        # Topicベクトル計算（学習時はコメント不使用）
        topic_context = self.model.encode_topic(
            input_ids=input_data['topic_context'],
            attention_mask=input_data['topic_context_mask']
        )
        
        topic_pos = self.model.encode_topic(
            input_ids=input_data['topic_pos'],
            attention_mask=input_data['topic_pos_mask']
        )
        
        topic_neg = self.model.encode_topic(
            input_ids=input_data['topic_neg'],
            attention_mask=input_data['topic_neg_mask']
        )
        
        # トピック学習損失計算
        topic_loss = self._compute_topic_loss(input_data, window_size or self.window_size)
        
        # 平均ベクトル計算
        topic_context_mean, topic_pos_mean, topic_neg_mean = self._compute_mean_vectors(
            topic_context, topic_pos, topic_neg,
            input_data['topic_context_num'],
            input_data['topic_pos_num'],
            input_data['topic_neg_num'],
            batch_size
        )
        
        # コサイン類似度計算
        topic_pos_scores = self.model.compute_topic_similarity(
            topic_context_mean, topic_pos_mean
        ).to(device)
        
        topic_neg_scores = self.model.compute_topic_similarity(
            topic_context_mean, topic_neg_mean
        ).to(device)
        
        # 総合スコア計算
        coheren_pos_scores = F.softmax(coheren_pos_scores, dim=1)[:, 0]
        coheren_neg_scores = F.softmax(coheren_neg_scores, dim=1)[:, 0]
        
        pos_scores = coheren_pos_scores + topic_pos_scores
        neg_scores = coheren_neg_scores + topic_neg_scores
        
        # マージン損失
        margin_loss = self.score_loss_fn(pos_scores, neg_scores)
        
        # 総合損失
        total_loss = margin_loss + topic_loss
        
        return total_loss, margin_loss, topic_loss
    
    def _compute_mean_vectors(
        self,
        topic_context: torch.Tensor,
        topic_pos: torch.Tensor,
        topic_neg: torch.Tensor,
        context_num: list,
        pos_num: list,
        neg_num: list,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        平均ベクトルを計算
        
        Returns:
            (context_mean, pos_mean, neg_mean)
        """
        topic_context_mean, topic_pos_mean, topic_neg_mean = [], [], []
        context_count, pos_count, neg_count = 0, 0, 0
        
        for i, j, z in zip(context_num, pos_num, neg_num):
            topic_context_mean.append(
                torch.mean(topic_context[context_count:context_count + i], dim=0)
            )
            topic_pos_mean.append(
                torch.mean(topic_pos[pos_count:pos_count + j], dim=0)
            )
            topic_neg_mean.append(
                torch.mean(topic_neg[neg_count:neg_count + z], dim=0)
            )
            context_count += i
            pos_count += j
            neg_count += z
        
        assert len(topic_context_mean) == len(topic_pos_mean) == len(topic_neg_mean) == batch_size
        
        context_mean = pad_sequence(topic_context_mean, batch_first=True)
        pos_mean = pad_sequence(topic_pos_mean, batch_first=True)
        neg_mean = pad_sequence(topic_neg_mean, batch_first=True)
        
        return context_mean, pos_mean, neg_mean
    
    def _compute_topic_loss(
        self,
        input_data: Dict[str, torch.Tensor],
        window_size: int
    ) -> torch.Tensor:
        """
        トピック損失を計算
        
        Args:
            input_data: 入力データ
            window_size: ウィンドウサイズ
            
        Returns:
            topic_loss
        """
        device = input_data['topic_train'].device
        topic_loss = torch.tensor(0.0, device=device)
        margin_count = 0
        
        # バッチ内の各サンプルを処理
        offset = 0  # topic_train内のオフセット
        for dial_len, current_utt, utt_count in input_data['topic_num']:
            total_utterances = dial_len
            
            # ローカルウィンドウの範囲を決定
            local_window_size = min(window_size, total_utterances - 1)
            start_idx = max(0, current_utt - local_window_size)
            end_idx = min(total_utterances, current_utt + local_window_size + 1)
            
            # 発話ベクトルを取得（オフセットを使って正しいサンプルの発話を取得）
            local_topic_train = input_data['topic_train'][offset + start_idx: offset + end_idx]
            local_mask = input_data['topic_train_mask'][offset + start_idx: offset + end_idx]
            
            local_fused = self.model.encode_topic(
                input_ids=local_topic_train.to(device),
                attention_mask=local_mask.to(device)
            )
            
            # 疑似セグメンテーション処理
            cur_loss = self._process_local_segmentation(
                local_fused,
                current_utt - start_idx,
                end_idx - start_idx,
                window_size,
                device
            )
            
            if cur_loss is not None:
                topic_loss += cur_loss
                margin_count += 1
            
            # メモリ解放
            del local_fused
            
            # 次のサンプルのオフセットを更新
            offset += utt_count
        
        return topic_loss / margin_count if margin_count > 0 else topic_loss
    
    def _process_local_segmentation(
        self,
        local_fused_embeddings: torch.Tensor,
        local_current_idx: int,
        local_dial_len: int,
        global_window_size: int,
        device: torch.device
    ) -> Optional[torch.Tensor]:
        """
        ローカルウィンドウ内での疑似セグメンテーション処理
        
        Args:
            local_fused_embeddings: ローカルウィンドウの埋め込み
            local_current_idx: 現在位置のローカルインデックス
            local_dial_len: ローカルウィンドウの長さ
            global_window_size: グローバルウィンドウサイズ
            device: デバイス
            
        Returns:
            損失値 or None
        """
        if local_dial_len <= 1:
            return None
        
        # トピックスコア計算
        top_cons, top_curs = [], []
        for i in range(1, local_dial_len):
            top_con = torch.mean(local_fused_embeddings[max(0, i-2): i], dim=0)
            top_cur = torch.mean(local_fused_embeddings[i: min(local_dial_len, i+2)], dim=0)
            top_cons.append(top_con)
            top_curs.append(top_cur)
        
        if not top_cons:
            return None
        
        top_cons = torch.stack(top_cons)
        top_curs = torch.stack(top_curs)
        topic_scores = F.cosine_similarity(top_cons, top_curs, dim=1, eps=1e-08).to(device)
        
        # 深度スコア計算
        raw_depth_scores = self.depth_calculator.calculate(torch.sigmoid(topic_scores))
        
        # Z-score正規化
        depth_scores_tensor = torch.stack([torch.tensor(s, device=device) for s in raw_depth_scores])
        local_mean = depth_scores_tensor.mean()
        local_std = depth_scores_tensor.std()
        
        if local_std > 1e-8:
            normalized_depth_scores = (depth_scores_tensor - local_mean) / local_std
        else:
            normalized_depth_scores = depth_scores_tensor - local_mean
        
        # セグメント検出
        depth_scores_np = normalized_depth_scores.cpu().detach().numpy()
        tet_seg = np.argsort(depth_scores_np)[-self.train_split:] + 1
        tet_seg = [0] + tet_seg.tolist() + [local_dial_len]
        tet_seg.sort()
        
        # 現在位置を含むセグメントを特定
        tet_mid = bisect.bisect(tet_seg, local_current_idx)
        if tet_mid >= len(tet_seg):
            tet_mid = len(tet_seg) - 1
        tet_mid_seg = (tet_seg[tet_mid-1], tet_seg[tet_mid])
        
        # マージン損失の計算
        pos_left = max(tet_mid_seg[0], local_current_idx - global_window_size)
        pos_right = min(tet_mid_seg[1], local_current_idx + global_window_size + 1)
        
        neg_left = min(tet_seg[max(0, tet_mid-1)], local_current_idx - global_window_size)
        neg_right = max(tet_seg[tet_mid], local_current_idx + global_window_size + 1)
        
        # アンカー
        anchor = local_fused_embeddings[local_current_idx].unsqueeze(0)
        
        # ポジティブサンプル
        pos_indices = list(range(pos_left, local_current_idx)) + \
                     list(range(local_current_idx + 1, pos_right))
        if not pos_indices:
            return None
        
        pos_embeddings = local_fused_embeddings[pos_indices]
        pos_scores = F.cosine_similarity(anchor, pos_embeddings, dim=1)
        
        # ネガティブサンプル
        neg_indices = list(range(neg_left)) + list(range(neg_right, local_dial_len))
        if not neg_indices:
            return None
        
        neg_embeddings = local_fused_embeddings[neg_indices]
        neg_scores = F.cosine_similarity(anchor, neg_embeddings, dim=1)
        
        # マージン損失計算
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return None
        
        margin_pos = pos_scores.unsqueeze(0).repeat(len(neg_scores), 1).T.flatten()
        margin_neg = neg_scores.repeat(len(pos_scores))
        
        cur_loss = self.score_loss_fn(margin_pos, margin_neg)
        
        return cur_loss if not torch.isnan(cur_loss) else None