import torch
import bisect
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForNextSentencePrediction, AutoModel
from transformers.models.bert.modeling_bert import *
from typing import Optional, Tuple  
from transformers.models.bert.modeling_bert import BertOnlyNSPHead
from transformers.modeling_outputs import NextSentencePredictorOutput

class MarginRankingLoss():
    def __init__(self, margin):
        self.margin = margin

    def __call__(self, p_scores, n_scores):
        scores = self.margin - (p_scores - n_scores)
        scores = scores.clamp(min=0)
        return scores.mean()

class CommentFusionLayer(nn.Module):
    def __init__(self, utterance_dim, comment_dim, output_dim):
        super(CommentFusionLayer, self).__init__()
        self.linear = nn.Linear(utterance_dim + comment_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, utterance_vec, comment_vec):
        combined = torch.cat([utterance_vec, comment_vec], dim=-1)
        fused = self.linear(combined)
        fused = self.layer_norm(fused)
        return fused

class AverageFusionLayer(nn.Module):
    """
    発話ベクトルとコメントベクトルの単純平均を取る層
    学習パラメータなしのシンプルな実装
    """
    def __init__(self):
        super(AverageFusionLayer, self).__init__()
        # 学習パラメータなし
    
    def forward(self, utterance_vec, comment_vec):
        """
        発話ベクトルとコメントベクトルの要素ごとの平均を計算
        Args:
            utterance_vec: (batch_size, 768) 発話ベクトル
            comment_vec: (batch_size, 768) コメントベクトル（開始時間～10秒間の平均）
        Returns:
            fused: (batch_size, 768) 平均化されたベクトル
        """
        # 単純な要素ごとの平均
        fused = (utterance_vec + comment_vec) / 2.0
        return fused

class SegModel(nn.Module):
    def __init__(self, model_path='', margin=1, train_split=5, window_size=5, 
                 utterance_dim=768, comment_dim=768, fused_dim=768,
                 use_pretrained_only=False,
                 coherence_model_name="cl-tohoku/bert-base-japanese",  
                 topic_model_name="pkshatech/simcse-ja-bert-base-clcmlp",  
                 use_comments_for_topic=True,
                 fusion_method='average'):  # 新しいパラメータ追加
        super(SegModel, self).__init__()
        self.margin = margin
        self.train_split = train_split
        self.window_size = window_size
        self.use_pretrained_only = use_pretrained_only
        self.use_comments_for_topic = use_comments_for_topic
        self.fusion_method = fusion_method  # 'average' または 'linear'
        
        print(f"モデル初期化: Coherence={coherence_model_name}, Topic={topic_model_name}, UseComments={use_comments_for_topic}, Fusion={fusion_method}")
        
        # モデル名をパラメータ化
        self.coherence_model_name = coherence_model_name
        self.topic_model_name = topic_model_name
        
        try:
            # Topicモデル: 指定されたモデルを使用
            self.topic_model = AutoModel.from_pretrained(topic_model_name)
            
            # Coherenceモデル: Next Sentence Prediction対応モデル
            self.coheren_model = BertForNextSentencePrediction.from_pretrained(
                coherence_model_name, 
                num_labels=2,
                output_attentions=False,
                output_hidden_states=True
            )
                
        except Exception as e:
            print(f"モデルロードエラー: {e}")
            # フォールバックとしてデフォルトモデルを使用
            print("デフォルトモデルでフォールバック")
            self.topic_model = AutoModel.from_pretrained("pkshatech/simcse-ja-bert-base-clcmlp")
            self.coheren_model = BertForNextSentencePrediction.from_pretrained(
                "cl-tohoku/bert-base-japanese", 
                num_labels=2,
                output_attentions=False,
                output_hidden_states=True
            )
        
        # 融合層の選択
        if self.use_comments_for_topic:
            if fusion_method == 'average':
                # 新しい平均融合層
                self.comment_fusion = AverageFusionLayer()
                print("✅ 平均融合層を使用（発話ベクトル + コメントベクトルの平均）")
            elif fusion_method == 'linear':
                # 既存の線形結合層
                self.comment_fusion = CommentFusionLayer(
                    utterance_dim=utterance_dim,
                    comment_dim=comment_dim,
                    output_dim=fused_dim
                )
                print("✅ 線形融合層を使用")
            else:
                raise ValueError(f"未知の融合方法: {fusion_method}")
        else:
            self.comment_fusion = None
            print("⚠️ コメント不使用モード")
        
        # 事前学習モデルのみを使用する場合は損失関数を不要に
        if not use_pretrained_only:
            self.topic_loss = nn.CrossEntropyLoss()
            self.score_loss = MarginRankingLoss(self.margin)
        else:
            self.topic_loss = None
            self.score_loss = None

    def tet(self, scores):
        output_scores = []
        for i in range(len(scores)):
            lflag, rflag = scores[i], scores[i]
            if i == 0:
                hl = scores[i]
                for r in range(i+1,len(scores)):
                    if rflag <= scores[r]:
                        rflag = scores[r]
                    else:
                        break
            elif i == len(scores)-1:
                hr = scores[i]
                for l in range(i-1, -1, -1):
                    if lflag <= scores[l]:
                        lflag = scores[l]
                    else:
                        break
            else:
                for r in range(i+1,len(scores)):
                    if rflag <= scores[r]:
                        rflag = scores[r]
                    else:
                        break
                for l in range(i-1, -1, -1):
                    if lflag <= scores[l]:
                        lflag = scores[l]
                    else:
                        break
            depth_score = 0.5*(lflag+rflag-2*scores[i])
            output_scores.append(depth_score.cpu().detach())
        return output_scores

    def forward(self, input_data, window_size=None):
        if self.use_pretrained_only:
            return self.inference_forward(input_data)
        
        device, topic_loss = input_data['coheren_inputs'].device, torch.tensor(0)
        topic_context_count, topic_pos_count, topic_neg_count = 0, 0, 0
        topic_context_mean, topic_pos_mean, topic_neg_mean = [], [], []
        
        coheren_pos_scores, coheren_pos_feature = self.coheren_model(
            input_data['coheren_inputs'][:, 0, :],
            attention_mask=input_data['coheren_mask'][:, 0, :],
            token_type_ids=input_data['coheren_type'][:, 0, :]
        )
        coheren_neg_scores, coheren_neg_feature = self.coheren_model(
            input_data['coheren_inputs'][:, 1, :],
            attention_mask=input_data['coheren_mask'][:, 1, :],
            token_type_ids=input_data['coheren_type'][:, 1, :]
        )

        batch_size = len(input_data['topic_context_num'])
        
        topic_context_utterance = self.topic_model(
            input_ids=input_data['topic_context'],
            attention_mask=input_data['topic_context_mask']
        ).last_hidden_state[:, 0, :]
        
        topic_pos_utterance = self.topic_model(
            input_ids=input_data['topic_pos'],
            attention_mask=input_data['topic_pos_mask']
        ).last_hidden_state[:, 0, :]
        
        topic_neg_utterance = self.topic_model(
            input_ids=input_data['topic_neg'],
            attention_mask=input_data['topic_neg_mask']
        ).last_hidden_state[:, 0, :]
        
        # コメント使用フラグに基づいて融合処理を分岐
        if self.use_comments_for_topic and self.comment_fusion is not None:
            topic_context = self.comment_fusion(
                topic_context_utterance, 
                input_data['topic_context_comments']
            )
            topic_pos = self.comment_fusion(
                topic_pos_utterance, 
                input_data['topic_pos_comments']
            )
            topic_neg = self.comment_fusion(
                topic_neg_utterance, 
                input_data['topic_neg_comments']
            )
        else:
            # コメントを使用しない場合は発話ベクトルのみを使用
            topic_context = topic_context_utterance
            topic_pos = topic_pos_utterance
            topic_neg = topic_neg_utterance
        
        topic_loss = self.topic_train(input_data, window_size)

        for i, j, z in zip(input_data['topic_context_num'], input_data['topic_pos_num'], input_data['topic_neg_num']):
            topic_context_mean.append(torch.mean(topic_context[topic_context_count:topic_context_count + i], dim=0))
            topic_pos_mean.append(torch.mean(topic_pos[topic_pos_count:topic_pos_count + j], dim=0))
            topic_neg_mean.append(torch.mean(topic_neg[topic_neg_count:topic_neg_count + z], dim=0))
            topic_context_count, topic_pos_count, topic_neg_count = topic_context_count + i, topic_pos_count + j, topic_neg_count + z

        assert len(topic_context_mean) == len(topic_pos_mean) == len(topic_neg_mean) == batch_size

        topic_context_mean, topic_pos_mean = pad_sequence(topic_context_mean, batch_first=True), pad_sequence(topic_pos_mean, batch_first=True)
        topic_neg_mean = pad_sequence(topic_neg_mean, batch_first=True)

        topic_pos_scores = F.cosine_similarity(topic_context_mean, topic_pos_mean, dim=1, eps=1e-08).to(device)
        topic_neg_scores = F.cosine_similarity(topic_context_mean, topic_neg_mean, dim=1, eps=1e-08).to(device)

        pos_scores = coheren_pos_scores.logits[:, 0] + topic_pos_scores
        neg_scores = coheren_neg_scores.logits[:, 0] + topic_neg_scores

        margin_loss = self.score_loss(pos_scores, neg_scores)
        loss = margin_loss.clone() + topic_loss
        
        return loss, margin_loss, topic_loss

    def infer(self, coheren_input, coheren_mask, coheren_type_id, 
            topic_input=None, topic_mask=None, topic_comments=None, topic_num=None,
            use_comments_for_topic=None, 
            fusion_method=None,  # 新しい引数（推論時に融合方法を変更可能）
            global_coherence_mean=None, global_coherence_std=None,
            global_topic_mean=None, global_topic_std=None):
        """
        推論実行
        Args:
            use_comments_for_topic: コメント使用フラグ
            fusion_method: 融合方法（'average' または 'linear'）
            global_coherence_mean, global_coherence_std: グローバルなコヒーレンス統計
            global_topic_mean, global_topic_std: グローバルなトピック統計
        """
        device = coheren_input.device
        
        # 融合方法の決定（引数＞クラス設定）
        fusion_type = fusion_method if fusion_method is not None else self.fusion_method
        
        # コメント使用フラグの決定（引数が指定されればそれを使用、否则はクラス設定を使用）
        use_comments = use_comments_for_topic if use_comments_for_topic is not None else self.use_comments_for_topic
        
        # コヒーレンススコア計算
        coheren_scores, coheren_feature = self.coheren_model(
            coheren_input, 
            attention_mask=coheren_mask, 
            token_type_ids=coheren_type_id
        )
        coherence_raw = coheren_scores.logits[:, 0]

        # トピックモデル推論
        topic_input_0 = topic_input[0].unsqueeze(0) if topic_input[0].dim() == 1 else topic_input[0]
        topic_input_1 = topic_input[1].unsqueeze(0) if topic_input[1].dim() == 1 else topic_input[1]
        topic_mask_0 = topic_mask[0].unsqueeze(0) if topic_mask[0].dim() == 1 else topic_mask[0]
        topic_mask_1 = topic_mask[1].unsqueeze(0) if topic_mask[1].dim() == 1 else topic_mask[1]
        
        topic_context_utterance = self.topic_model(
            input_ids=topic_input_0,
            attention_mask=topic_mask_0
        ).last_hidden_state[:, 0, :]
        
        topic_cur_utterance = self.topic_model(
            input_ids=topic_input_1,
            attention_mask=topic_mask_1
        ).last_hidden_state[:, 0, :]
        
        # コメント融合（コメント使用フラグに基づいて分岐）
        if use_comments and self.comment_fusion is not None:
            topic_comments_0 = topic_comments[0].unsqueeze(0) if topic_comments[0].dim() == 1 else topic_comments[0]
            topic_comments_1 = topic_comments[1].unsqueeze(0) if topic_comments[1].dim() == 1 else topic_comments[1]
            
            # 融合実行
            topic_context = self.comment_fusion(topic_context_utterance, topic_comments_0)
            topic_cur = self.comment_fusion(topic_cur_utterance, topic_comments_1)
        else:
            # コメントを使用しない場合は発話ベクトルのみを使用
            topic_context = topic_context_utterance
            topic_cur = topic_cur_utterance
        
        # トピックスコア計算
        topic_context_count = topic_cur_count = 0
        topic_context_mean, topic_cur_mean = [], []

        for i, j in zip(topic_num[0], topic_num[1]):
            topic_context_mean.append(torch.mean(topic_context[topic_context_count:topic_context_count + i], dim=0))
            topic_cur_mean.append(torch.mean(topic_cur[topic_cur_count:topic_cur_count + j], dim=0))
            topic_context_count, topic_cur_count = topic_context_count + i, topic_cur_count + j
            
        topic_context_mean, topic_cur_mean = pad_sequence(topic_context_mean, batch_first=True), pad_sequence(topic_cur_mean, batch_first=True)
        topic_scores = F.cosine_similarity(topic_context_mean, topic_cur_mean, dim=1, eps=1e-08).to(device)
        
        # グローバル統計を使用してZ-score正規化
        coherence_normalized = None
        topic_normalized = None
        
        if (global_coherence_mean is not None and global_coherence_std is not None and 
            global_topic_mean is not None and global_topic_std is not None):
            
            # コヒーレンススコアのZ-score正規化
            if global_coherence_std > 1e-8:
                coherence_normalized = (coherence_raw - global_coherence_mean) / global_coherence_std
            else:
                coherence_normalized = coherence_raw - global_coherence_mean
                
            # トピックスコアのZ-score正規化
            if global_topic_std > 1e-8:
                topic_normalized = (topic_scores - global_topic_mean) / global_topic_std
            else:
                topic_normalized = topic_scores - global_topic_mean
            
            # 正規化したスコアを統合
            final_scores = coherence_normalized + topic_normalized
            
        else:
            # グローバル統計がない場合は生スコアをそのまま加算（フォールバック）
            final_scores = coherence_raw + topic_scores
            coherence_normalized = coherence_raw  # フォールバック時は生スコアを使用
            topic_normalized = topic_scores       # フォールバック時は生スコアを使用
        
        # デバッグ情報を保存（正規化後のスコアも追加）
        self.last_inference_debug_info = {
            'coherence_raw': coherence_raw.detach().cpu().numpy().tolist(),
            'topic_raw': topic_scores.detach().cpu().numpy().tolist(),
            'final_raw': final_scores.detach().cpu().numpy().tolist(),
            'use_comments_for_topic': use_comments,
            'fusion_method': fusion_type,
            'coherence_normalized': coherence_normalized.detach().cpu().numpy().tolist() if coherence_normalized is not None else coherence_raw.detach().cpu().numpy().tolist(),
            'topic_normalized': topic_normalized.detach().cpu().numpy().tolist() if topic_normalized is not None else topic_scores.detach().cpu().numpy().tolist(),
            'normalization_stats': {
                'coherence_mean': float(global_coherence_mean) if global_coherence_mean is not None else 0.0,
                'coherence_std': float(global_coherence_std) if global_coherence_std is not None else 1.0,
                'topic_mean': float(global_topic_mean) if global_topic_mean is not None else 0.0,
                'topic_std': float(global_topic_std) if global_topic_std is not None else 1.0
            } if (global_coherence_mean is not None and global_topic_mean is not None) else None
        }

        return torch.sigmoid(final_scores).detach().cpu().numpy().tolist()

    def topic_train(self, input_data, window_size):
        device = input_data['coheren_inputs'].device
        batch_size = len(input_data['topic_context_num'])
        
        topic_loss = torch.tensor(0.0).to(device)
        margin_count = 0
        
        for b in range(batch_size):
            total_utterances, current_utt = input_data['topic_num'][b]
            
            local_window_size = 100 # ローカルウィンドウサイズの設定
            start_idx = max(0, current_utt - local_window_size)
            end_idx = min(total_utterances, current_utt + local_window_size + 1)
            
            # データの取得
            local_topic_train = input_data['topic_train'][start_idx:end_idx]
            local_mask = input_data['topic_train_mask'][start_idx:end_idx]
            local_comments = input_data['topic_train_comments'][b, start_idx:end_idx]
            
            # メモリ効率化: 明示的な勾配計算制御
            with torch.set_grad_enabled(True):
                local_utterance = self.topic_model(
                    input_ids=local_topic_train.to(device),
                    attention_mask=local_mask.to(device)
                ).last_hidden_state[:, 0, :]
            
            # コメント融合（コメント使用フラグに基づいて分岐）
            if self.use_comments_for_topic and self.comment_fusion is not None:
                if local_comments.dim() == 3:
                    local_comments = local_comments.squeeze(0)
                
                local_fused = self.comment_fusion(local_utterance, local_comments.to(device))
            else:
                # コメントを使用しない場合は発話ベクトルのみを使用
                local_fused = local_utterance
            
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
            del local_utterance, local_fused
            torch.cuda.empty_cache()
        
        return topic_loss / margin_count if margin_count > 0 else topic_loss

    def _process_local_segmentation(self, local_fused_embeddings, local_current_idx, local_dial_len, global_window_size, device):
        """ローカルウィンドウ内での疑似セグメンテーション処理（深度スコア正規化版）"""
        if local_dial_len <= 1:
            return None
        
        # 1. トピックスコア計算（従来通り）
        top_cons, top_curs = [], []
        for i in range(1, local_dial_len):
            top_con = torch.mean(local_fused_embeddings[max(0, i-2): i], dim=0)
            top_cur = torch.mean(local_fused_embeddings[i: min(local_dial_len, i+2)], dim=0)
            top_cons.append(top_con)
            top_curs.append(top_cur)
        
        if not top_cons:
            return None
                
        top_cons, top_curs = torch.stack(top_cons), torch.stack(top_curs)
        topic_scores = F.cosine_similarity(top_cons, top_curs, dim=1, eps=1e-08).to(device)
        
        # 2. 深度スコア計算（従来通り）
        raw_depth_scores = self.tet(torch.sigmoid(topic_scores))
        
        # 3. 推論時と同じZ-score正規化を適用
        depth_scores_tensor = torch.stack(raw_depth_scores).to(device)
        
        # ローカルウィンドウ内での統計量計算
        local_mean = depth_scores_tensor.mean()
        local_std = depth_scores_tensor.std()
        
        # 推論時の正規化ロジックと同様に処理
        if local_std > 1e-8:
            normalized_depth_scores = (depth_scores_tensor - local_mean) / local_std
        else:
            normalized_depth_scores = depth_scores_tensor - local_mean
        
        # 4. 正規化された深度スコアでセグメント検出
        depth_scores_np = normalized_depth_scores.cpu().detach().numpy()
        tet_seg = np.argsort(depth_scores_np)[-self.train_split:] + 1
        tet_seg = [0] + tet_seg.tolist() + [local_dial_len]
        tet_seg.sort()

        # 5. 現在位置を含むセグメントを特定
        tet_mid = bisect.bisect(tet_seg, local_current_idx)
        if tet_mid >= len(tet_seg):
            tet_mid = len(tet_seg) - 1
        tet_mid_seg = (tet_seg[tet_mid-1], tet_seg[tet_mid])
        
        # 6. マージン損失の計算（ローカルウィンドウ内）
        pos_left = max(tet_mid_seg[0], local_current_idx - global_window_size)
        pos_right = min(tet_mid_seg[1], local_current_idx + global_window_size + 1)
        
        neg_left = min(tet_seg[max(0, tet_mid-1)], local_current_idx - global_window_size)
        neg_right = max(tet_seg[tet_mid], local_current_idx + global_window_size + 1)

        # 7. ローカルなマージン損失計算
        anchor = local_fused_embeddings[local_current_idx].unsqueeze(0)
        
        # ポジティブサンプル（同じトピックセグメント内）
        pos_indices = list(range(pos_left, local_current_idx)) + list(range(local_current_idx + 1, pos_right))
        if not pos_indices:
            return None
            
        pos_embeddings = local_fused_embeddings[pos_indices]
        pos_scores = F.cosine_similarity(anchor, pos_embeddings, dim=1)
        
        # ネガティブサンプル（別トピックセグメント）
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
        
        cur_loss = self.score_loss(margin_pos, margin_neg)
        
        # 8. デバッグ情報の保存（推論時と同様の形式）
        self.last_local_segmentation_debug = {
            'raw_depth_scores': [score.item() for score in raw_depth_scores],
            'normalized_depth_scores': normalized_depth_scores.cpu().detach().numpy().tolist(),
            'local_mean': local_mean.item(),
            'local_std': local_std.item(),
            'segmentation_boundaries': tet_seg,
            'current_segment': tet_mid_seg,
            'positive_samples_count': len(pos_indices),
            'negative_samples_count': len(neg_indices),
            'window_size': local_dial_len
        }
        
        return cur_loss if not torch.isnan(cur_loss) else None

class BertForNextSentencePrediction(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_feature=False,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ), pooled_output

def tet(scores):
    output_scores = []
    for i in range(len(scores)):
        lflag, rflag = scores[i], scores[i]
        if i == 0:
            hl = scores[i]
            for r in range(i+1,len(scores)):
                if rflag <= scores[r]:
                    rflag = scores[r]
                else:
                    break
        elif i == len(scores)-1:
            hr = scores[i]
            for l in range(i-1, -1, -1):
                if lflag <= scores[l]:
                    lflag = scores[l]
                else:
                    break
        else:
            for r in range(i+1,len(scores)):
                if rflag <= scores[r]:
                    rflag = scores[r]
                else:
                    break
            for l in range(i-1, -1, -1):
                if lflag <= scores[l]:
                    lflag = scores[l]
                else:
                    break
        depth_score = 0.5*(lflag+rflag-2*scores[i])
        output_scores.append(depth_score.cpu().detach())
    return output_scores