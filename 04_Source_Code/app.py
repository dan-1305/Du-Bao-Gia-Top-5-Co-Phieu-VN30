import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
import os
import sys
import json
import joblib
from pathlib import Path

# 1. CẤU HÌNH TRANG
st.set_page_config(
    page_title="Hệ Thống Dự Báo Cổ Phiếu VN30 (Multimodal AI)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR.parent / "03_Models"
sys.path.insert(0, str(APP_DIR))

try:
    from model_architecture import MultimodalStockModel
    from backtest_engine import BacktestEngine
    from nlp_processor import NLPProcessor
except ImportError:
    pass


@st.cache_resource
def load_all_models():
    """Load model PyTorch, Random Forest, SVM và Scaler."""
    models = {}
    if not MODELS_DIR.exists():
        return models
    try:
        if (MODELS_DIR / "scaler.pkl").exists():
            models["scaler"] = joblib.load(MODELS_DIR / "scaler.pkl")
        if (MODELS_DIR / "feature_columns.json").exists():
            with open(MODELS_DIR / "feature_columns.json", encoding="utf-8") as f:
                models["feature_cols"] = json.load(f)
        if (MODELS_DIR / "rf_model.pkl").exists():
            models["rf"] = joblib.load(MODELS_DIR / "rf_model.pkl")
            models["rf_classes"] = joblib.load(MODELS_DIR / "rf_classes.pkl")
        if (MODELS_DIR / "svm_model.pkl").exists():
            models["svm"] = joblib.load(MODELS_DIR / "svm_model.pkl")
        if (MODELS_DIR / "metrics.json").exists():
            with open(MODELS_DIR / "metrics.json", encoding="utf-8") as f:
                models["metrics"] = json.load(f)
        if (MODELS_DIR / "multimodal_model.pt").exists():
            import torch
            with open(MODELS_DIR / "multimodal_config.json", encoding="utf-8") as f:
                cfg = json.load(f)
            pt_model = MultimodalStockModel(
                n_features=cfg["n_features"], sequence_length=cfg["sequence_length"],
                nlp_dim=cfg["nlp_dim"], hidden_lstm=cfg.get("hidden_lstm", 32),
                hidden_dense=cfg.get("hidden_dense", 24)
            )
            pt_model.load_state_dict(torch.load(MODELS_DIR / "multimodal_model.pt", map_location="cpu"))
            pt_model.eval()
            models["multimodal"] = pt_model
            models["lstm_scaler"] = joblib.load(MODELS_DIR / "lstm_scaler.pkl")
            models["lstm_y_scaler"] = joblib.load(MODELS_DIR / "lstm_y_scaler.pkl")
            models["multimodal_cfg"] = cfg
    except Exception as e:
        st.sidebar.error(f"Lỗi load model: {e}")
    return models


def build_latest_features(latest, feature_cols):
    """Xây dựng véc-tơ đặc trưng cho một dòng dữ liệu."""
    vals = []
    for col in feature_cols:
        vals.append(latest.get(col, 0.0))
    return np.array([vals], dtype=np.float32)


def predict_trend_rf(df):
    """Dự báo xu hướng tăng/giảm/đi ngang bằng Random Forest."""
    if "rf" not in MODELS or "scaler" not in MODELS or "feature_cols" not in MODELS:
        return None
    feature_cols = MODELS["feature_cols"]
    latest = df.iloc[-1]
    X = build_latest_features(latest, feature_cols)
    X_s = MODELS["scaler"].transform(X)
    proba = MODELS["rf"].predict_proba(X_s)[0]
    idx = int(np.argmax(proba))
    label_num = int(MODELS["rf_classes"][idx])
    label_map = {1: "Tăng", 0: "Đi Ngang", -1: "Giảm"}
    trend = label_map.get(label_num, f"Lớp {label_num}")
    return trend, float(proba[idx])


def predict_price_lstm(df, n_days=5):
    """Dự báo đệ quy giá n ngày tiếp theo bằng mô hình PyTorch Multimodal."""
    if "multimodal" not in MODELS or "lstm_scaler" not in MODELS or "lstm_y_scaler" not in MODELS:
        return None
    import torch
    model = MODELS["multimodal"]
    scaler = MODELS["lstm_scaler"]
    y_scaler = MODELS["lstm_y_scaler"]
    cfg = MODELS["multimodal_cfg"]
    feature_cols = MODELS["feature_cols"]
    
    if len(df) < cfg["sequence_length"]:
        return None
        
    predictions = []
    current_df = df.copy()
    
    for step in range(n_days):
        # Lấy window cuối cùng
        seq_df = current_df.tail(cfg["sequence_length"])
        X_seq = seq_df[feature_cols].values
        
        # Chuẩn hóa đầu vào
        X_seq_s = scaler.transform(X_seq) # [30, F]
        X_tensor = torch.tensor([X_seq_s], dtype=torch.float32) # [1, 30, F]
        
        # Lấy sentiment
        latest_sentiment = float(current_df["sentiment_score"].iloc[-1])
        if latest_sentiment > 0.1:
            nlp_feat = [0.7, 0.1, 0.2]
        elif latest_sentiment < -0.1:
            nlp_feat = [0.1, 0.7, 0.2]
        else:
            nlp_feat = [0.2, 0.2, 0.6]
        nlp_tensor = torch.tensor([nlp_feat], dtype=torch.float32) # [1, 3]
        
        # Chạy dự báo
        with torch.no_grad():
            pred_p_s, _, _ = model(X_tensor, nlp_tensor)
            pred_p = float(y_scaler.inverse_transform(pred_p_s.numpy().reshape(-1, 1))[0][0])
            
        predictions.append(pred_p)
        
        # Append dòng mới giả lập để chạy đệ quy
        new_row = current_df.iloc[-1].copy()
        new_index = current_df.index[-1] + pd.Timedelta(days=1)
        new_row["close"] = pred_p
        new_row["sentiment_score"] = 0.0 # Tương lai trung lập
        current_df.loc[new_index] = new_row
        
    return predictions


MODELS = load_all_models()


@st.cache_data
def load_stock_data(ticker: str):
    data_path = APP_DIR.parent / "01_Data" / "processed_data.csv"
    if not data_path.exists():
        return None
    df = pd.read_csv(data_path, parse_dates=["time"])
    if "ticker" in df.columns:
        df = df[df["ticker"] == ticker].copy()
    df = df.sort_values("time").set_index("time")
    return df



# 2. SIDEBAR CẤU HÌNH MỚI (Multimodal AI)
st.sidebar.title("💎 VN30 AI ROBO-ADVISOR")
st.sidebar.markdown("**Đồ án:** Dự báo giá Top 5 cổ phiếu VN30 bằng Multimodal AI (LSTM-Attention + NLP Sentiment).")
st.sidebar.markdown("---")

ticker = st.sidebar.selectbox("🎯 Chọn mã cổ phiếu", ['FPT', 'HPG', 'MBB', 'MWG', 'VNM'])
model_type = st.sidebar.selectbox(
    "🧠 Chọn Mô Hình Trí Tuệ Nhân Tạo",
    [
        "Học sâu: LSTM Multimodal (Dự báo giá liên tục)",
        "Học máy: Random Forest (Phân loại xu hướng)",
        "Học máy: SVM (Phân loại xu hướng)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Quản Trị Rủi Ro Mặc Định")
st.sidebar.info(
    "• **Hard Stop-Loss:** -7.0%\n"
    "• **Take-Profit:** +14.0%\n"
    "• **Vốn giả lập:** 100,000,000 VNĐ\n"
    "• **Tỷ trọng/lệnh:** 20% NAV"
)

# 3. KHU VỰC CHÍNH (MAIN DASHBOARD)
st.title(f"📈 DASHBOARD DỰ BÁO CỔ PHIẾU {ticker} (VN30)")
st.markdown("Hệ thống hỗ trợ ra quyết định đầu tư tích hợp **Học sâu Đa phương thức** (Chuỗi thời gian + Tâm lý Tin tức).")

# Tải dữ liệu thật
df = load_stock_data(ticker)
if df is None or len(df) < 2:
    st.warning(f"⚠️ Chưa có dữ liệu cho mã {ticker} trong processed_data.csv.")
    st.stop()

# Bổ sung các cột đặc trưng kỹ thuật còn thiếu (khớp feature_columns.json lúc train)
if 'log_ret' not in df.columns:
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
if 'price_change' not in df.columns:
    df['price_change'] = df['close'].pct_change()
if 'sma_10' not in df.columns:
    df['sma_10'] = df['close'].rolling(10).mean()
if 'sma_20' not in df.columns:
    df['sma_20'] = df['close'].rolling(20).mean()
if 'sentiment_score' not in df.columns:
    df['sentiment_score'] = 0.0
# Cột target phục vụ sliding window cho backtest (nhãn 3 lớp: -1/0/1 theo ngưỡng ±1%)
_next_ret = (df['close'].shift(-1) - df['close']) / df['close']
df['target_3class'] = 0
df.loc[_next_ret > 0.01, 'target_3class'] = 1
df.loc[_next_ret < -0.01, 'target_3class'] = -1
df['target_price_next'] = df['close'].shift(-1).fillna(df['close'].iloc[-1])
df = df.dropna(subset=['log_ret', 'price_change', 'sma_10', 'sma_20'])

latest_close = float(df['close'].iloc[-1])
prev_close = float(df['close'].iloc[-2])
price_change = latest_close - prev_close
pct_change = (price_change / prev_close) * 100
latest_vol = float(df['volume'].iloc[-1])
rsi_val = float(df["RSI_14"].iloc[-1]) if "RSI_14" in df.columns else 50.0
sent_val = float(df["sentiment_score"].iloc[-1]) if "sentiment_score" in df.columns else 0.0

# 4. HÀNG CHỈ SỐ KPI (4 thẻ)
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric(label="Giá Đóng Cửa Hiện Tại (VND)",
              value=f"{latest_close:,.0f}",
              delta=f"{price_change:,.0f} ({pct_change:.2f}%)")

with k2:
    if "Học sâu" in model_type:
        lstm_preds = predict_price_lstm(df, n_days=5)
        if lstm_preds is not None and len(lstm_preds) > 0:
            predicted_price = float(lstm_preds[0])
            pred_change = predicted_price - latest_close
            pred_pct = (pred_change / latest_close) * 100
            st.metric(label="Dự báo T+1 (Multimodal LSTM)",
                      value=f"{predicted_price:,.0f}",
                      delta=f"{pred_change:,.0f} ({pred_pct:.2f}%)")
        else:
            st.metric(label="Dự báo T+1 (Multimodal LSTM)",
                      value="Chưa có model",
                      delta="Chạy train_models.py trước", delta_color="off")
    elif "Random Forest" in model_type:
        result = predict_trend_rf(df)
        if result:
            trend, conf = result
            color = "normal" if trend == "Tăng" else ("inverse" if trend == "Giảm" else "off")
            st.metric(label="Dự báo Xu hướng (Random Forest)",
                      value=f"{trend} ({conf*100:.0f}%)",
                      delta="Model thật | chỉ báo kỹ thuật + sentiment", delta_color=color)
        else:
            st.metric(label="Dự báo Xu hướng (RF)", value="Lỗi", delta="", delta_color="off")
    else:
        if "svm" in MODELS and "scaler" in MODELS:
            feature_cols = MODELS.get("feature_cols", [])
            X = build_latest_features(df.iloc[-1], feature_cols)
            X_s = MODELS["scaler"].transform(X)
            proba = MODELS["svm"].predict_proba(X_s)[0]
            idx = int(np.argmax(proba))
            label_num = int(MODELS["svm"].classes_[idx])
            label_map = {1: "Tăng", 0: "Đi Ngang", -1: "Giảm"}
            trend = label_map.get(label_num, f"Lớp {label_num}")
            color = "normal" if trend == "Tăng" else ("inverse" if trend == "Giảm" else "off")
            st.metric(label="Dự báo Xu hướng (SVM)",
                      value=f"{trend} ({proba[idx]*100:.0f}%)",
                      delta="Model thật | chỉ báo kỹ thuật + sentiment", delta_color=color)
        else:
            st.metric(label="Dự báo Xu hướng (SVM)", value="Chưa có model",
                      delta="Chạy train_models.py trước", delta_color="off")

with k3:
    rsi_state = "Quá Mua ⚠️" if rsi_val > 70 else ("Quá Bán 🔻" if rsi_val < 30 else "Bình Thường")
    st.metric(label="Chỉ Số RSI (14)", value=f"{rsi_val:.1f}", delta=rsi_state, delta_color="off")

with k4:
    sent_desc = "Tích Cực 🟢" if sent_val > 0.1 else ("Tiêu Cực 🔴" if sent_val < -0.1 else "Trung Lập ⚪")
    st.metric(label="Tâm Lý Tin Tức (NLP)", value=f"{sent_val:+.2f}", delta=sent_desc, delta_color="off")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Dự Báo & Biểu Đồ", "📰 Phân Tích Cảm Xúc", "💰 Giả Lập Giao Dịch (Backtest)"])

with tab1:
    st.subheader("📈 Biểu đồ Biến động Giá & Đường Dự Báo")
    df_plot = df.tail(60).copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['close'], mode='lines+markers', name='Giá Thực Tế', line=dict(color='#2980b9', width=2)))

    if "Học sâu" in model_type:
        last_date = df_plot.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=5)
        lstm_preds = predict_price_lstm(df, n_days=5)
        if lstm_preds is not None and len(lstm_preds) == 5:
            pred_dates = [last_date] + list(future_dates)
            pred_prices = [latest_close] + [float(p) for p in lstm_preds]
            fig.add_trace(go.Scatter(x=pred_dates, y=pred_prices, mode='lines+markers', name='Dự Báo (LSTM-Attention)', line=dict(color='#c0392b', width=2, dash='dash')))
        else:
            st.info("ℹ️ Đường dự báo LSTM sẽ hiển thị sau khi train model (multimodal_model.pt trong 03_Models/).")

    fig.update_layout(height=500, margin=dict(l=20, r=20, t=30, b=20), xaxis_title='Thời gian', yaxis_title='Giá (VND)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("👁️‍🗨️ Xem chi tiết dữ liệu gốc (Bảng)"):
        st.dataframe(df.sort_index(ascending=False).head(20), use_container_width=True)

with tab2:
    st.subheader("📰 Phân Tích Cảm Xúc Tin Tức Tài Chính")
    st.markdown("Hệ thống tự động phân loại các tiêu đề tin tức thành 3 nhóm cảm xúc bằng mô hình **PhoBERT/FinBERT** hoặc **Từ điển Cảm xúc tiếng Việt chuyên biệt**.")
    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        st.markdown("### 🔍 Phân tích Tin tức Độc lập")
        user_news = st.text_input("Nhập tiêu đề hoặc bài viết tài chính cần phân tích:", "FPT báo lãi ròng quý 2 tăng trưởng kỷ lục 35% nhờ mảng xuất khẩu phần mềm bùng nổ")
        if st.button("Phân tích Sắc thái"):
            nlp = NLPProcessor()
            res = nlp.analyze_single(user_news)
            st.success(f"**Kết quả sắc thái:** {res['label']}")
            st.write(f"**Điểm cảm xúc liên tục (Sentiment Score):** `{res['score']}`")
            probs = res['probabilities']
            prob_df = pd.DataFrame({"Cảm xúc": ["Tích cực (Bullish)", "Tiêu cực (Bearish)", "Trung lập (Neutral)"], "Xác suất (%)": [probs[0]*100, probs[1]*100, probs[2]*100]})
            st.bar_chart(prob_df.set_index("Cảm xúc"))
    with col_s2:
        st.markdown("### 📌 Đồng hồ Cảm xúc Hiện tại")
        latest_sent = float(df["sentiment_score"].iloc[-1]) if "sentiment_score" in df.columns else 0.0
        st.metric(label="Điểm Cảm xúc Toàn thị trường (Phiên cuối)", value=f"{latest_sent:+.2f}")
        if latest_sent > 0.15: st.success("🔥 THỊ TRƯỜNG ĐANG HƯNG PHẤN (BULLISH)")
        elif latest_sent < -0.15: st.error("📉 THỊ TRƯỜNG ĐANG BI QUAN (BEARISH)")
        else: st.warning("⚖️ THỊ TRƯỜNG ĐANG LƯỠNG LỰ (NEUTRAL)")


with tab3:
    st.subheader("💰 Giả Lập Giao Dịch & Quản Trị Rủi Ro (Backtest)")
    st.markdown("Cấu hình kịch bản mô phỏng giao dịch thực tế trên rổ dữ liệu lịch sử bằng động cơ **BacktestEngine** để kiểm định hiệu năng sinh lời thực tế.")

    col_b1, col_b2 = st.columns([1, 2])
    with col_b1:
        st.markdown("#### ⚙️ Tham số Giao dịch")
        init_cap = st.number_input("Vốn khởi tạo (VND)", min_value=10_000_000.0, max_value=1_000_000_000.0, value=100_000_000.0, step=10_000_000.0)
        comm = st.slider("Phí giao dịch (%)", 0.0, 1.0, 0.15, step=0.05) / 100.0
        tax = st.slider("Thuế chứng khoán (%)", 0.0, 1.0, 0.10, step=0.05) / 100.0
        sl = st.slider("Cắt lỗ cứng (Stop-loss %)", -20.0, -1.0, -7.0, step=1.0) / 100.0
        tp = st.slider("Chốt lời mục tiêu (Take-profit %)", 5.0, 50.0, 14.0, step=1.0) / 100.0
        pos_size = st.slider("Phân bổ vị thế (Position Size %)", 5.0, 100.0, 20.0, step=5.0) / 100.0

    with col_b2:
        st.markdown("#### 🏆 Kết quả Kiểm thử (Backtest Summary)")
        
        feature_cols = MODELS.get("feature_cols", [])
        signals = np.zeros(len(df))
        
        if "scaler" in MODELS and feature_cols:
            if "Random Forest" in model_type and "rf" in MODELS:
                X = df[feature_cols].values
                X_s = MODELS["scaler"].transform(X)
                signals = MODELS["rf"].predict(X_s)
            elif "SVM" in model_type and "svm" in MODELS:
                X = df[feature_cols].values
                X_s = MODELS["scaler"].transform(X)
                signals = MODELS["svm"].predict(X_s)
            elif "Học sâu" in model_type and "multimodal" in MODELS:
                from feature_engineering import FeatureEngineer
                cfg = MODELS["multimodal_cfg"]
                fe_temp = FeatureEngineer(window_size=cfg["sequence_length"])
                X_ts_win, _, _ = fe_temp.create_sliding_windows(df, feature_cols)
                if len(X_ts_win) > 0:
                    import torch
                    model = MODELS["multimodal"]
                    scaler = MODELS["lstm_scaler"]
                    X_ts_s = scaler.transform(X_ts_win.reshape(-1, X_ts_win.shape[-1])).reshape(X_ts_win.shape)
                    
                    nlp_feats = []
                    for s in df["sentiment_score"].values[cfg["sequence_length"]-1: -1]:
                        if s > 0.1: nlp_feats.append([0.7, 0.1, 0.2])
                        elif s < -0.1: nlp_feats.append([0.1, 0.7, 0.2])
                        else: nlp_feats.append([0.2, 0.2, 0.6])
                    if len(nlp_feats) < len(X_ts_win):
                        nlp_feats = np.pad(nlp_feats, ((0, len(X_ts_win) - len(nlp_feats)), (0, 0)), mode='edge')
                        
                    t_ts = torch.tensor(X_ts_s, dtype=torch.float32)
                    t_nlp = torch.tensor(nlp_feats, dtype=torch.float32)
                    with torch.no_grad():
                        _, trend_logits, _ = model(t_ts, t_nlp)
                        pred_cls = torch.argmax(trend_logits, dim=1).numpy()
                    inv_map = {0: -1, 1: 0, 2: 1}
                    pred_cls_orig = np.array([inv_map[v] for v in pred_cls])
                    signals[cfg["sequence_length"]:] = pred_cls_orig

        be = BacktestEngine(
            initial_capital=init_cap,
            commission=comm,
            tax=tax,
            stop_loss_pct=sl,
            take_profit_pct=tp,
            position_size_pct=pos_size
        )
        res = be.run(df, signals)
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Lợi nhuận AI (ROI %)", f"{res['cumulative_return_pct']}%")
        col_m2.metric("Lợi nhuận Buy & Hold", f"{res['buy_and_hold_return_pct']}%")
        col_m3.metric("Tỷ lệ Sharpe", f"{res['sharpe_ratio']}")
        col_m4.metric("Sụt giảm tối đa (MaxDD %)", f"{res['max_drawdown_pct']}%")

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=res['equity_curve'].index, y=res['equity_curve'], mode='lines', name='Vốn AI (Multimodal AI Portfolio)', line=dict(color='#2ecc71', width=2.5)))
        fig_eq.add_trace(go.Scatter(x=res['buy_and_hold_curve'].index, y=res['buy_and_hold_curve'], mode='lines', name='Vốn Buy & Hold (Mua & Giữ)', line=dict(color='#7f8c8d', width=1.5, dash='dot')))
        fig_eq.update_layout(
            title="So sánh Tăng trưởng Vốn (Equity Curve)",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Thời gian",
            yaxis_title="Giá trị tài sản (VND)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_eq, use_container_width=True)