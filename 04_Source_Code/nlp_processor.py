"""
=========================================================================
 NLP PROCESSOR MODULE - FINANCIAL SENTIMENT ANALYSIS
=========================================================================
Phân tích cảm xúc tin tức tài chính tiếng Việt (PhoBERT / FinBERT / Lexicon fallback).
Xuất ra vector phân phối xác suất [P_Positive, P_Negative, P_Neutral]
và điểm cảm xúc liên tục Sentiment Score [-1.0, +1.0].
"""
import numpy as np
from typing import List, Dict, Union


# Từ điển trọng số cảm xúc tài chính tiếng Việt (Lexicon Rule-based cho tốc độ tức thì)
VIETNAMESE_FINANCIAL_LEXICON = {
    # Tích cực (Bullish)
    "tăng trưởng": 0.8, "lãi kỷ lục": 0.9, "vượt kế hoạch": 0.85, "mua ròng": 0.7,
    "bứt phá": 0.75, "trúng thầu": 0.8, "hợp đồng nghìn tỷ": 0.9, "cổ tức cao": 0.75,
    "mở rộng thị phần": 0.7, "khởi sắc": 0.65, "đột biến": 0.6, "doanh thu tăng": 0.75,
    "lợi nhuận tăng": 0.8, "triển vọng tích cực": 0.7, "khuyến nghị mua": 0.8,
    "nâng hạng": 0.75, "hút dòng tiền": 0.65, "nới room": 0.8, "sóng tăng": 0.7,
    
    # Tiêu cực (Bearish)
    "sụt giảm": -0.8, "lỗ nặng": -0.9, "bị khởi tố": -0.95, "bán tháo": -0.85,
    "bán ròng": -0.7, "lao dốc": -0.8, "thua lỗ": -0.85, "nợ xấu": -0.8,
    "vi phạm": -0.75, "hủy niêm yết": -0.95, "cắt margin": -0.7, "điều chỉnh sâu": -0.65,
    "áp lực bán": -0.6, "thắt chặt": -0.5, "cảnh báo": -0.6, "khủng hoảng": -0.85,
    "lạm phát": -0.5, "suy thoái": -0.75, "đình trệ": -0.6,
}


class NLPProcessor:
    """Xử lý phân tích cảm xúc tin tức tài chính."""
    def __init__(self, model_name: str = "ProsusAI/finbert", use_hf: bool = False):
        self.use_hf = use_hf
        self.model = None
        self.tokenizer = None
        
        if use_hf:
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
            except Exception as e:
                print(f"[NLPProcessor] Không load được HuggingFace model ({e}), chuyển sang Lexicon fallback.")
                self.use_hf = False

    def score_text_lexicon(self, text: str) -> Dict[str, float]:
        """Tính điểm cảm xúc bằng từ điển tài chính tiếng Việt."""
        text_lower = text.lower()
        score = 0.0
        match_count = 0
        
        for phrase, weight in VIETNAMESE_FINANCIAL_LEXICON.items():
            if phrase in text_lower:
                score += weight
                match_count += 1
                
        if match_count > 0:
            score = max(-1.0, min(1.0, score / match_count))
        else:
            score = 0.0
            
        # Chuyển score sang xác suất softmax 3 lớp: [Positive, Negative, Neutral]
        if score > 0.1:
            p_pos = min(0.9, 0.4 + score * 0.5)
            p_neg = max(0.05, 0.2 - score * 0.15)
            p_neu = 1.0 - p_pos - p_neg
        elif score < -0.1:
            p_neg = min(0.9, 0.4 + abs(score) * 0.5)
            p_pos = max(0.05, 0.2 - abs(score) * 0.15)
            p_neu = 1.0 - p_pos - p_neg
        else:
            p_pos, p_neg, p_neu = 0.2, 0.2, 0.6

        return {
            "score": round(float(score), 4),
            "probabilities": [round(float(p_pos), 4), round(float(p_neg), 4), round(float(p_neu), 4)],
            "label": "Tích cực" if score > 0.15 else ("Tiêu cực" if score < -0.15 else "Trung lập")
        }

    def get_sentiment(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Trả về mảng xác suất cảm xúc shape [N, 3] cho danh sách bài báo:
        [P_Positive, P_Negative, P_Neutral]
        """
        if self.use_hf and self.model is not None and self.tokenizer is not None:
            import torch
            self.model.eval()
            all_probs = []
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    inputs = self.tokenizer(
                        batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
                    ).to(self.device)
                    outputs = self.model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    all_probs.append(probs.cpu().numpy())
            return np.concatenate(all_probs, axis=0)
        else:
            results = [self.score_text_lexicon(t)["probabilities"] for t in texts]
            return np.array(results)

    def analyze_single(self, text: str) -> Dict[str, Union[float, str, List[float]]]:
        """Phân tích một tiêu đề hoặc đoạn tin tức."""
        return self.score_text_lexicon(text)

