"""
=========================================================================
 MULTIMODAL DEEP LEARNING ARCHITECTURE (PyTorch)
=========================================================================
Kiến trúc Dung hợp Đa phương thức (LSTM + Time Attention + NLP Sentiment Fusion)
sử dụng hàm mất mát Huber Loss và Cross-Entropy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TimeAttention(nn.Module):
    """Cơ chế Chú ý theo thời gian (Time-step Attention)."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn_weights = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [Batch, Seq_len, Hidden_dim]
        scores = torch.tanh(self.attn_weights(x))  # [Batch, Seq_len, 1]
        alpha = F.softmax(scores, dim=1)           # [Batch, Seq_len, 1]
        context = torch.sum(x * alpha, dim=1)      # [Batch, Hidden_dim]
        return context, alpha.squeeze(-1)


class MultimodalStockModel(nn.Module):
    """
    Mô hình Đa phương thức:
    - Nhánh 1 (Giá): LSTM 2 tầng + Time Attention
    - Nhánh 2 (Tin tức): Dense + BatchNorm
    - Khối Dung hợp: Concatenation + Dense + Dropout
    - Dual Output: Giá dự báo (Hồi quy Huber) + Xác suất xu hướng (3 lớp)
    """
    def __init__(
        self,
        n_features: int = 19,
        sequence_length: int = 30,
        nlp_dim: int = 3,
        hidden_lstm: int = 48,
        hidden_dense: int = 32,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.nlp_dim = nlp_dim

        # Nhánh 1: LSTM chuỗi thời gian
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_lstm,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0.0,
        )
        self.attention = TimeAttention(hidden_lstm)

        # Nhánh 2: Xử lý NLP Sentiment
        self.nlp_fc = nn.Sequential(
            nn.Linear(nlp_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )

        # Khối Dung hợp (Fusion Block)
        fusion_in_dim = hidden_lstm + 16
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_in_dim, hidden_dense),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Đầu ra 1: Hồi quy giá phiên tiếp theo (Continuous Price)
        self.price_head = nn.Linear(hidden_dense, 1)

        # Đầu ra 2: Phân loại xu hướng 3 lớp (-1=Giảm, 0=Đi ngang, 1=Tăng)
        self.trend_head = nn.Linear(hidden_dense, 3)

    def forward(
        self, x_ts: torch.Tensor, x_nlp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x_ts: [Batch, Seq_len, n_features]
        x_nlp: [Batch, nlp_dim]
        Returns: (pred_price, trend_logits, attention_weights)
        """
        # Nhánh chuỗi thời gian
        lstm_out, _ = self.lstm(x_ts)                     # [Batch, Seq, hidden_lstm]
        v_ts, attn_weights = self.attention(lstm_out)     # [Batch, hidden_lstm]

        # Nhánh cảm xúc văn bản
        v_nlp = self.nlp_fc(x_nlp)                        # [Batch, 16]

        # Dung hợp
        v_fused = torch.cat([v_ts, v_nlp], dim=1)         # [Batch, hidden_lstm + 16]
        features = self.fusion_fc(v_fused)                # [Batch, hidden_dense]

        pred_price = self.price_head(features).squeeze(-1) # [Batch]
        trend_logits = self.trend_head(features)           # [Batch, 3]

        return pred_price, trend_logits, attn_weights
