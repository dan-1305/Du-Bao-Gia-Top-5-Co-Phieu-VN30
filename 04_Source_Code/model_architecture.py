"""
=========================================================================
 MULTIMODAL DEEP LEARNING ARCHITECTURE (PyTorch)
=========================================================================
Kiến trúc Dung hợp Đa phương thức chuẩn xác theo Mục 3.1 & 3.4.2.1 đồ án:
- Nhánh định lượng: LSTM 2 tầng (hidden=32) + Time-Step Attention
- Nhánh ngôn ngữ: Dense Layer đưa vector cảm xúc lên 16 chiều
- Khối Dung hợp: Concatenation (48 chiều) -> Dense (24 chiều) -> Dropout (0.25)
- Đầu ra kép: Hồi quy giá phiên T+1 (Huber Loss) & Phân loại xu hướng 3 lớp
"""
import os
import logging
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv

load_dotenv()
TARGET_DEVICE = os.getenv("TORCH_DEVICE", "cpu")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TimeAttention(nn.Module):
    """Cơ chế Chú ý theo bước thời gian (Time-step Self-Attention)."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn_weights: nn.Linear = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Đầu vào: x [Batch, Seq_len, hidden_dim]
        Đầu ra: context_vector [Batch, hidden_dim], attention_weights [Batch, Seq_len]
        """
        scores = torch.tanh(self.attn_weights(x))
        alpha = F.softmax(scores, dim=1)
        context = torch.sum(x * alpha, dim=1)
        return context, alpha.squeeze(-1)


class MultimodalStockModel(nn.Module):
    """Mô hình Dung hợp Đa phương thức kết hợp chuỗi thời gian và cảm xúc tin tức."""

    def __init__(
        self,
        n_features: int = 19,
        sequence_length: int = 30,
        nlp_dim: int = 3,
        hidden_lstm: int = 32,
        nlp_hidden: int = 16,
        hidden_dense: int = 24,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.sequence_length: int = sequence_length
        self.n_features: int = n_features
        self.nlp_dim: int = nlp_dim

        # Nhánh 1: Chuỗi thời gian (LSTM 2 tầng + Attention)
        self.lstm: nn.LSTM = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_lstm,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0.0 else 0.0,
        )
        self.attention: TimeAttention = TimeAttention(hidden_lstm)

        # Nhánh 2: Biến đổi vector cảm xúc tin tức lên 16 chiều
        self.nlp_fc: nn.Sequential = nn.Sequential(
            nn.Linear(nlp_dim, nlp_hidden),
            nn.BatchNorm1d(nlp_hidden),
            nn.ReLU(),
        )

        # Khối Dung hợp (Fusion Block: 32 + 16 = 48 chiều -> 24 chiều)
        fusion_input_dim = hidden_lstm + nlp_hidden
        self.fusion_fc: nn.Sequential = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dense),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Đầu ra 1: Hồi quy giá liên tục (Price Head)
        self.price_head: nn.Linear = nn.Linear(hidden_dense, 1)

        # Đầu ra 2: Phân loại xác suất xu hướng 3 lớp (Trend Head)
        self.trend_head: nn.Linear = nn.Linear(hidden_dense, 3)

    def forward(
        self, x_ts: torch.Tensor, x_nlp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Thực thi lan truyền tiến với cơ chế điều hướng thiết bị tự động.
        x_ts: [Batch, Seq_len, n_features]
        x_nlp: [Batch, nlp_dim]
        """
        device = next(self.parameters()).device
        if x_ts.device != device:
            x_ts = x_ts.to(device)
        if x_nlp.device != device:
            x_nlp = x_nlp.to(device)

        # Trích xuất đặc trưng chuỗi thời gian -> v_ts (32 chiều)
        lstm_out, _ = self.lstm(x_ts)
        v_ts, attn_weights = self.attention(lstm_out)

        # Trích xuất đặc trưng cảm xúc -> v_nlp (16 chiều)
        v_nlp = self.nlp_fc(x_nlp)

        # Nối tầng thành vector đa phương thức -> v_fused (48 chiều)
        v_fused = torch.cat([v_ts, v_nlp], dim=1)
        features = self.fusion_fc(v_fused)

        # Đầu ra dự báo
        pred_price = self.price_head(features).squeeze(-1)
        trend_logits = self.trend_head(features)

        return pred_price, trend_logits, attn_weights


# =========================================================================
# UNIT TEST BLUEPRINTS
# =========================================================================
def test_attention_dimensions() -> None:
    """Kiểm thử chiều không gian vector của module TimeAttention."""
    attn = TimeAttention(hidden_dim=32)
    dummy_input = torch.randn(16, 30, 32)
    context, weights = attn(dummy_input)

    assert context.shape == (16, 32), "Lỗi: Chiều véc-tơ ngữ cảnh không bằng 32"
    assert weights.shape == (16, 30), "Lỗi: Trọng số phân bổ thời gian không khớp seq_len"
    assert torch.allclose(torch.sum(weights, dim=1), torch.ones(16), atol=1e-5), "Lỗi: Tổng trọng số Attention != 1"
    logger.info("Unit Test Passed: TimeAttention vận hành chính xác.")


def test_multimodal_model_forward() -> None:
    """Kiểm thử tính tương thích kích thước đa tầng của MultimodalStockModel."""
    model = MultimodalStockModel(
        n_features=19,
        sequence_length=30,
        nlp_dim=3,
        hidden_lstm=32,
        nlp_hidden=16,
        hidden_dense=24,
        dropout=0.25,
    )
    model.eval()

    dummy_ts = torch.randn(8, 30, 19)
    dummy_nlp = torch.randn(8, 3)

    with torch.no_grad():
        price, trend, attn = model(dummy_ts, dummy_nlp)

    assert price.shape == (8,), f"Lỗi Price Head: Kỳ vọng shape (8,), thực tế {price.shape}"
    assert trend.shape == (8, 3), f"Lỗi Trend Head: Kỳ vọng shape (8, 3), thực tế {trend.shape}"
    assert attn.shape == (8, 30), f"Lỗi Attention: Kỳ vọng shape (8, 30), thực tế {attn.shape}"
    logger.info("Unit Test Passed: Kích thước mạng nơ-ron khớp hoàn toàn với Báo cáo Đồ án.")


if __name__ == "__main__":
    test_attention_dimensions()
    test_multimodal_model_forward()