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


def predict_price_lstm(df, n_days=5, mdl=None):
    """Dự báo đệ quy giá n ngày tiếp theo bằng mô hình PyTorch Multimodal (theo từng mã)."""
    src = mdl if (mdl and "multimodal" in mdl) else MODELS
    if "multimodal" not in src or "lstm_scaler" not in src or "lstm_y_scaler" not in src:
        return None
    import torch
    model = src["multimodal"]
    scaler = src["lstm_scaler"]
    y_scaler = src["lstm_y_scaler"]
    cfg = src["multimodal_cfg"]
    feature_cols = src.get("feature_cols", MODELS.get("feature_cols", []))
    
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
    """Load du lieu theo tung ma: uu tien processed_{TICKER}.csv, fallback processed_data.csv."""
    per_ticker = APP_DIR.parent / "01_Data" / f"processed_{ticker}.csv"
    data_path = per_ticker if per_ticker.exists() else APP_DIR.parent / "01_Data" / "processed_data.csv"
    if not data_path.exists():
        return None
    df = pd.read_csv(data_path, parse_dates=["time"])
    if "ticker" in df.columns:
        df = df[df["ticker"] == ticker].copy()
    if "RSI_14" not in df.columns:
        try:
            from feature_engineering import FeatureEngineer
            df = FeatureEngineer(window_size=30).add_indicators(df)
        except Exception:
            pass
    df = df.sort_values("time").set_index("time")
    df = df[~df.index.duplicated(keep="last")]
    return df


@st.cache_resource
def load_ticker_deep(ticker: str):
    """Load model Multimodal theo tung ma (03_Models/top5/), FPT dung model goc."""
    try:
        if ticker == "FPT":
            if "multimodal" in MODELS:
                return {k: MODELS.get(k) for k in ("multimodal", "lstm_scaler", "lstm_y_scaler", "multimodal_cfg", "feature_cols")}
            return None
        T5 = MODELS_DIR / "top5"
        need = [T5 / f"{ticker}_model.pt", T5 / f"{ticker}_ts_scaler.pkl", T5 / f"{ticker}_y_scaler.pkl"]
        if not all(p.exists() for p in need):
            return None
        import torch
        model = MultimodalStockModel(n_features=19, sequence_length=30, nlp_dim=3, hidden_lstm=32, hidden_dense=24)
        model.load_state_dict(torch.load(need[0], map_location="cpu"))
        model.eval()
        return {"multimodal": model, "lstm_scaler": joblib.load(need[1]), "lstm_y_scaler": joblib.load(need[2]),
                "multimodal_cfg": {"sequence_length": 30}, "feature_cols": MODELS.get("feature_cols", [])}
    except Exception:
        return None



# 2. SIDEBAR CẤU HÌNH MỚI (Multimodal AI)
st.sidebar.title("💎 VN30 AI ROBO-ADVISOR")
st.sidebar.markdown("**Đồ án:** Dự báo giá Top 5 cổ phiếu VN30 bằng Multimodal AI (LSTM-Attention + NLP Sentiment).")
st.sidebar.markdown("---")

ticker = st.sidebar.selectbox("🎯 Chọn mã cổ phiếu", ['FPT', 'HPG', 'MBB', 'MWG', 'VNM'])
model_type = st.sidebar.selectbox(
    "🧠 Mô hình Dự báo (KPI & biểu đồ Tab 1)",
    [
        "Học sâu: LSTM Multimodal (Dự báo giá liên tục)",
        "Học máy: Random Forest (Phân loại xu hướng)",
        "Học máy: SVM (Phân loại xu hướng)"
    ],
    help="Chọn mô hình dùng cho thẻ KPI và đường dự báo ở Tab 1. KHÔNG ảnh hưởng Tab Backtest — "
         "nơi có 'Nguồn tín hiệu vào lệnh' riêng để kiểm định chiến lược giao dịch."
)
if ticker != "FPT" and "Học sâu" not in model_type:
    st.sidebar.caption("⚠️ RF/SVM chỉ train trên FPT — với mã khác hãy chọn Học sâu. (Tab Backtest dùng nguồn tín hiệu riêng, độc lập lựa chọn này.)")

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
        lstm_preds = predict_price_lstm(df, n_days=5, mdl=load_ticker_deep(ticker))
        if lstm_preds is not None and len(lstm_preds) > 0:
            predicted_price = float(lstm_preds[0])
            pred_change = predicted_price - latest_close
            pred_pct = (pred_change / latest_close) * 100
            st.metric(label="Dự báo T+1 (Multimodal LSTM)",
                      value=f"{predicted_price:,.0f}",
                      delta=f"{pred_change:,.0f} ({pred_pct:.2f}%)",
                      help="Giá đóng cửa dự báo cho phiên kế tiếp bằng model Multimodal LSTM-Attention; "
                           "từ đó hệ thống đệ quy dự báo tiếp 5 phiên (đường đỏ nét đứt trên biểu đồ Tab 1). "
                           "Chỉ mang tính tham khảo, không phải khuyến nghị đầu tư.")
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
    st.metric(label="Chỉ Số RSI (14)", value=f"{rsi_val:.1f}", delta=rsi_state, delta_color="off",
              help="Relative Strength Index (14 phiên) — chỉ báo động lượng giá: >70 vùng quá mua "
                   "(cảnh báo điều chỉnh), <30 vùng quá bán (cảnh báo hồi phục), giữa 30–70 là trung tính.")

with k4:
    sent_desc = "Tích Cực 🟢" if sent_val > 0.1 else ("Tiêu Cực 🔴" if sent_val < -0.1 else "Trung Lập ⚪")
    st.metric(label="Tâm Lý Tin Tức (NLP)", value=f"{sent_val:+.2f}", delta=sent_desc, delta_color="off",
              help="Điểm cảm xúc tin tức trong khoảng [-1, +1] do module NLP tính từ tiêu đề tin: "
                   ">0 tích cực, <0 tiêu cực. Đây là đầu vào thứ 2 (bên cạnh chuỗi giá) của model Multimodal.")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Dự Báo & Biểu Đồ", "📰 Phân Tích Cảm Xúc", "💰 Giả Lập Giao Dịch (Backtest)"])

with tab1:
    st.subheader("📈 Biểu đồ Biến động Giá & Đường Dự Báo")
    src_note = "model gốc (03_Models/multimodal_model.pt)" if ticker == "FPT" else f"model riêng theo mã (03_Models/top5/{ticker}_model.pt)"
    st.caption(f"📅 Dữ liệu thật: {df.index[0]:%d/%m/%Y} → {df.index[-1]:%d/%m/%Y} · {len(df):,} phiên giao dịch · Nguồn model: {src_note}")
    df_plot = df.tail(60).copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['close'], mode='lines+markers', name='Giá Thực Tế', line=dict(color='#2980b9', width=2)))

    if "Học sâu" in model_type:
        last_date = df_plot.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=5)
        lstm_preds = predict_price_lstm(df, n_days=5, mdl=load_ticker_deep(ticker))
        if lstm_preds is not None and len(lstm_preds) == 5:
            pred_dates = [last_date] + list(future_dates)
            pred_prices = [latest_close] + [float(p) for p in lstm_preds]
            fig.add_trace(go.Scatter(x=pred_dates, y=pred_prices, mode='lines+markers', name='Dự Báo (LSTM-Attention)', line=dict(color='#c0392b', width=2, dash='dash')))
        else:
            st.info("ℹ️ Đường dự báo LSTM sẽ hiển thị sau khi train model (multimodal_model.pt trong 03_Models/).")

    fig.update_layout(height=500, margin=dict(l=20, r=20, t=30, b=20), xaxis_title='Thời gian', yaxis_title='Giá (VND)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🏆 Chất lượng Mô hình — Độ chính xác Dự báo (DA / RMSE / MAPE)"):
        st.markdown("**FPT — model gốc** (kiến trúc Multimodal, test 20% cuối):")
        met_path = MODELS_DIR / "metrics.json"
        if met_path.exists():
            with open(met_path, encoding="utf-8") as f:
                met = json.load(f)
            mdl = met.get("multimodal_deep_learning", {})
            rf_acc = met.get("classification", {}).get("random_forest", {}).get("accuracy")
            svm_acc = met.get("classification", {}).get("svm", {}).get("accuracy")
            row_fpt = pd.DataFrame([{
                "DA (%)": round(mdl.get("direction_accuracy", 0) * 100, 2),
                "RMSE": round(mdl.get("rmse", 0)),
                "MAE": round(mdl.get("mae", 0)),
                "MAPE (%)": mdl.get("mape"),
                "RF baseline acc (%)": round(rf_acc * 100, 2) if rf_acc else None,
                "SVM baseline acc (%)": round(svm_acc * 100, 2) if svm_acc else None,
            }])
            st.dataframe(row_fpt, use_container_width=True, hide_index=True)
        else:
            st.info("Chưa có metrics.json.")
        st.markdown("**4 mã còn lại — model top5 (thí nghiệm tin thật vs neutral, cùng kiến trúc + seed 42):**")
        t5_path = MODELS_DIR / "top5_results_v3.json"
        if t5_path.exists():
            with open(t5_path, encoding="utf-8") as f:
                t5 = json.load(f)
            rows_t5 = pd.DataFrame([{
                "Mã": r["ticker"],
                "DA tin thật (%)": round(r["da_news"] * 100, 2),
                "DA neutral (%)": round(r["da_neutral"] * 100, 2),
                "Δ DA (điểm %)": r["delta_pp"],
                "RMSE": r.get("rmse_news"),
                "Số phiên test": r.get("n_test"),
                "Ngày có tin phủ": r.get("sent_days_cov"),
            } for r in t5])
            st.dataframe(rows_t5, use_container_width=True, hide_index=True)
            st.caption("DA = Direction Accuracy (tỷ lệ phiên dự báo đúng chiều). Δ DA trong biên nhiễu do tin RSS chỉ phủ 3–16 ngày/mã "
                       "→ khai báo trung thực: cần news archive sâu hơn (hướng phát triển của đồ án).")
        else:
            st.info("Chưa có top5_results_v3.json.")

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

    # --- Bảng tin tức đã thu thập thật (Google News RSS + Lexicon) ---
    st.markdown("---")
    st.markdown("### 🗞️ Tin tức Đã Thu Thập (Google News RSS + Lexicon 38 cụm)")
    news_path = APP_DIR.parent / "01_Data" / f"news_{ticker}.csv"
    if news_path.exists():
        news_raw = pd.read_csv(news_path)
        news_top = news_raw.sort_values("date", ascending=False).head(30)
        show_news = pd.DataFrame({
            "Ngày": pd.to_datetime(news_top["date"]).dt.strftime("%d/%m/%Y"),
            "Điểm": news_top["score"].map(lambda v: "🟢 +1" if int(v) > 0 else ("🔴 -1" if int(v) < 0 else "⚪ 0")),
            "Tiêu đề": news_top["title"],
        })
        st.dataframe(show_news, use_container_width=True, hide_index=True, height=420)
        st.caption(f"📊 Tổng cộng {len(news_raw):,} tin đã crawl cho {ticker} — điểm cảm xúc chấm tự động bằng "
                   f"Lexicon tiếng Việt 38 cụm (§2.3.1). Tin này là nguồn sentiment đầu vào thứ 2 của model Multimodal.")
    else:
        st.info(f"ℹ️ Chưa có file tin tức cho {ticker} (01_Data/news_{ticker}.csv) — hiện có sẵn cho HPG/MBB/MWG/VNM.")


with tab3:
    st.subheader("💰 Giả Lập Giao Dịch & Quản Trị Rủi Ro (Backtest)")
    st.markdown("Chọn **nguồn tín hiệu** → bấm **▶ Chạy Backtest**. Chiến lược Multimodal chạy trên **tập kiểm thử** (20% cuối) bằng tín hiệu dự báo thật đúng ngưỡng §3.5.1 — đồ thị giá đánh dấu từng lệnh MUA▲/BÁN▼ (kèm SL/TP) để soi backtest có ổn không. *Lưu ý: nguồn tín hiệu backtest ở Tab 3 độc lập với mô hình hiển thị dự báo ở Tab 1 — mỗi tab một mục đích.*")

    col_b1, col_b2 = st.columns([1, 2])
    with col_b1:
        st.markdown("#### ⚙️ Tham số Giao dịch")
        strat_options = ["🤖 Multimodal LSTM-Attention — mô hình chính (§3.5.1)"]
        if ticker == "FPT":
            strat_options.append("🌲 Random Forest — baseline đối chiếu (§4.3.1)")
        strat = st.selectbox("Nguồn tín hiệu vào lệnh", strat_options,
                             help="AI nào phát tín hiệu MUA/BÁN trong giả lập. Lý do học thuật của 2 lựa chọn: "
                                  "mở expander '🎯 Vì sao có lựa chọn nguồn tín hiệu?' ngay bên dưới.")
        is_rf_bt = "Random Forest" in strat
        use_sent_bt = st.toggle(
            "🔒 Đồng thuận sentiment khi MUA (§3.5.1)", value=True,
            help="Bật: chỉ vào lệnh khi điểm sentiment > 0 (đúng thiết kế §3.5.1). Tắt: chỉ dùng ngưỡng dự báo ±1%/ngày.",
        )
        with st.expander("🎯 Vì sao có lựa chọn nguồn tín hiệu? (đáp phản biện)"):
            st.markdown(
                "Đồ án so sánh **2 họ mô hình** trên cùng bài toán dự báo, nên kiểm định chiến lược cần 2 nguồn tín hiệu tương ứng:\n\n"
                "- **🤖 Multimodal LSTM-Attention (đề xuất chính — §3.5.1):** tín hiệu MUA/BÁN từ dự báo lợi nhuận vượt ngưỡng ±1%/ngày, kèm đồng thuận sentiment.\n"
                "- **🌲 Random Forest (baseline ML truyền thống — §4.3.1):** tín hiệu từ nhãn Tăng/Giảm/Đi ngang — chỉ train trên FPT nên chỉ mở với mã FPT.\n\n"
                "Nguồn tín hiệu này **độc lập với ô 'Mô hình Dự báo' ở sidebar** — mỗi vai trò một mục đích: sidebar điều khiển hiển thị dự báo (KPI + biểu đồ Tab 1), còn nguồn tín hiệu ở đây quyết định AI phát lệnh khi **kiểm định chiến lược giao dịch** (Tab 3). Đây là phép đối chiếu baseline vs đề xuất theo phương pháp luận đồ án."
            )
        init_cap = st.number_input("Vốn khởi tạo (VND)", min_value=10_000_000, max_value=1_000_000_000, value=100_000_000, step=10_000_000, format="%d",
                                   help="Số vốn giả lập ban đầu (VND). Mọi chỉ số ROI, MaxDD, Equity curve đều tính trên vốn này — không phải tiền thật.")
        comm = st.slider("Phí giao dịch (%)", 0.0, 1.0, 0.15, step=0.05, help="Phí môi giới chứng khoán tính trên mỗi lệnh MUA và BÁN (HOSE phổ biến 0,15%). Giao dịch càng nhiều, tổng phí càng ăn mòn lợi nhuận.") / 100.0
        tax = st.slider("Thuế chứng khoán (%)", 0.0, 1.0, 0.10, step=0.05, help="Thuế chuyển nhượng chứng khoán 0,1% theo quy định Việt Nam — chỉ tính khi BÁN cổ phiếu.") / 100.0
        sl = st.slider("Cắt lỗ cứng (Stop-loss %)", -20.0, -1.0, -7.0, step=1.0, help="Nếu lệnh đang giữ lỗ chạm ngưỡng này (ví dụ -7%), hệ thống BÁN bắt buộc để giới hạn thua lỗ — nguyên tắc quản trị rủi ro số 1 của đồ án (§4.4.1).") / 100.0
        tp = st.slider("Chốt lời mục tiêu (Take-profit %)", 5.0, 50.0, 14.0, step=1.0, help="Nếu lệnh lãi đạt ngưỡng này (ví dụ +14%), hệ thống BÁN để hiện thực hóa lợi nhuận trước khi giá quay đầu.") / 100.0
        pos_size = st.slider("Phân bổ vị thế (Position Size %)", 5.0, 100.0, 20.0, step=5.0, help="Tỷ trọng vốn dành cho MỖI lệnh so với tổng tài sản. 20% nghĩa là tối đa chia vốn 5 phần — tránh 'bỏ hết trứng vào một giỏ'.") / 100.0
        run_btn = st.button("▶ Chạy Backtest", type="primary", use_container_width=True)

    with col_b2:
        st.markdown("#### 🏆 Kết quả Kiểm thử (Backtest Summary)")
        if run_btn:
            if is_rf_bt:
                if ticker != "FPT":
                    st.warning("⚠️ Random Forest chỉ được train trên FPT — hãy chọn chiến lược Multimodal.")
                elif "scaler" in MODELS and "rf" in MODELS and MODELS.get("feature_cols"):
                    X_bt = df[MODELS["feature_cols"]].values
                    sig = MODELS["rf"].predict(MODELS["scaler"].transform(X_bt))
                    be = BacktestEngine(initial_capital=init_cap, commission=comm, tax=tax,
                                        stop_loss_pct=sl, take_profit_pct=tp, position_size_pct=pos_size)
                    res = be.run(df, sig)
                    st.session_state["bt"] = {"kind": "rf", "ticker": ticker,
                        "res": {"roi": res["cumulative_return_pct"], "bh": res["buy_and_hold_return_pct"],
                                "sharpe": res["sharpe_ratio"], "maxdd": res["max_drawdown_pct"],
                                "win": res["win_rate_pct"], "n": res["total_trades"]},
                        "eq": res["equity_curve"], "bhq": res["buy_and_hold_curve"], "trades": []}
                else:
                    st.warning("Chưa có model RF trong 03_Models/.")
            else:
                preds_path = MODELS_DIR / "top5" / f"{ticker}_predictions.csv"
                if preds_path.exists():
                    pdf = pd.read_csv(preds_path)
                    from backtest_threshold import run_threshold_backtest
                    r = run_threshold_backtest(pdf, init_capital=init_cap, commission=comm, tax=tax,
                                               stop_loss=sl, take_profit=tp, pos_size=pos_size,
                                               use_sentiment=use_sent_bt)
                    st.session_state["bt"] = {"kind": "mm", "ticker": ticker,
                        "res": {"roi": r["roi_pct"], "bh": r["bh_pct"], "sharpe": r["sharpe"],
                                "maxdd": r["maxdd_pct"], "win": r["win_rate"], "n": r["n_trades"]},
                        "eq": r["equity"], "bhq": r["bh"], "dd": r["dd"],
                        "close": pd.Series(r["close"], index=r["times"]), "trades": r["trades"]}
                else:
                    st.info(f"ℹ️ Chưa có file dự báo cho {ticker} (03_Models/top5/{ticker}_predictions.csv).")

        bt = st.session_state.get("bt")
        if not bt:
            st.info("👆 Cấu hình tham số bên trái rồi bấm **▶ Chạy Backtest**.")
        elif bt.get("ticker") != ticker:
            st.info(f"Đang hiển thị kết quả của mã {bt.get('ticker')} — bấm ▶ để chạy lại cho {ticker}.")
        else:
            m = bt["res"]
            ma, mb, mc = st.columns(3)
            md, me, mf = st.columns(3)
            ma.metric("ROI AI", f"{float(m['roi']):+,.2f}%",
                      help="Return on Investment — tổng lợi nhuận/lỗ tích lũy của tài khoản AI trên vốn ban đầu, "
                           "đã trừ phí môi giới + thuế + trượt giá. So sánh trực tiếp với Buy & Hold để đánh giá "
                           "chiến lược AI có đáng dùng hay không.")
            mb.metric("Buy & Hold", f"{float(m['bh']):+,.2f}%",
                      help="Chiến lược đối chiếu thụ động: mua ngay phiên đầu kỳ và giữ nguyên đến cuối, không giao dịch. "
                           "Đây là 'chuẩn mực' — chiến lược AI chỉ có giá trị khi ROI AI vượt qua Buy & Hold.")
            mc.metric("Sharpe", f"{float(m['sharpe']):,.2f}",
                      help="Sharpe Ratio — lợi nhuận vượt lãi suất phi rủi ro trên mỗi đơn vị biến động lợi nhuận "
                           "(quy đổi theo năm, √252). Thang tham khảo: <1 kém · 1–2 tốt · >2 rất tốt.")
            md.metric("Max Drawdown", f"{float(m['maxdd']):,.2f}%",
                      help="Mức sụt vốn tối đa từ đỉnh xuống đáy trong kỳ kiểm thử (peak-to-trough). "
                           "Đo rủi ro tột cùng mà nhà đầu tư phải chịu — càng gần 0 càng an toàn.")
            me.metric("Win Rate", f"{float(m['win']):,.2f}%",
                      help="Tỷ lệ số lệnh có lãi (PnL > 0) trên tổng số lệnh đã đóng. Win rate cao chưa chắc tổng tài sản "
                           "tăng nếu lỗ nặng lãi nhẹ — cần xem kèm ROI và Max Drawdown.")
            mf.metric("Số lệnh", f"{int(m['n'])}",
                      help="Tổng số vòng giao dịch MUA→BÁN hoàn chỉnh trong kỳ. Giao dịch càng nhiều, "
                           "phí + thuế càng ăn mòn lợi nhuận (xem tham số Phí giao dịch).")
            if bt["kind"] == "mm":
                fig_bt = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.3, 0.2],
                                       vertical_spacing=0.04,
                                       subplot_titles=("Giá & điểm vào/ra lệnh (tập kiểm thử)", "Vốn AI vs Buy & Hold", "Drawdown (%)"))
                fig_bt.add_trace(go.Scatter(x=bt["close"].index, y=bt["close"], mode="lines", name="Giá đóng cửa", line=dict(color="#2980b9", width=1.8)), row=1, col=1)
                tr_df = pd.DataFrame(bt["trades"]) if bt["trades"] else pd.DataFrame()
                if len(tr_df):
                    fig_bt.add_trace(go.Scatter(x=tr_df["ngay_mua"], y=tr_df["gia_mua"], mode="markers", name="MUA ▲",
                                                marker=dict(symbol="triangle-up", size=12, color="#2ecc71"),
                                                hovertext=[f"MUA {d} @ {p:,.0f}" for d, p in zip(tr_df["ngay_mua"], tr_df["gia_mua"])]), row=1, col=1)
                    fig_bt.add_trace(go.Scatter(x=tr_df["ngay_ban"], y=tr_df["gia_ban"], mode="markers", name="BÁN ▼",
                                                marker=dict(symbol="triangle-down", size=12, color="#e74c3c"),
                                                hovertext=[f"{rr} | {d} @ {p:,.0f} ({g:+.2f}%)" for rr, d, p, g in zip(tr_df["ly_do"], tr_df["ngay_ban"], tr_df["gia_ban"], tr_df["pnl_pct"])]), row=1, col=1)
                fig_bt.add_trace(go.Scatter(x=bt["eq"].index, y=bt["eq"], mode="lines", name="Vốn AI", line=dict(color="#2ecc71", width=2.4)), row=2, col=1)
                fig_bt.add_trace(go.Scatter(x=bt["bhq"].index, y=bt["bhq"], mode="lines", name="Buy & Hold", line=dict(color="#7f8c8d", width=1.5, dash="dot")), row=2, col=1)
                fig_bt.add_trace(go.Scatter(x=bt["dd"].index, y=bt["dd"], mode="lines", name="Drawdown", line=dict(color="#e67e22", width=1.5), fill="tozeroy"), row=3, col=1)
                fig_bt.update_layout(height=780, margin=dict(l=20, r=20, t=40, b=20),
                                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                fig_bt.update_yaxes(tickformat=",.0f", row=1, col=1)
                fig_bt.update_yaxes(tickformat=",.0f", row=2, col=1)
                fig_bt.update_yaxes(tickformat=",.1f", ticksuffix="%", row=3, col=1)
                st.plotly_chart(fig_bt, use_container_width=True)
                if len(tr_df):
                    st.markdown("##### 📒 Trade Log — từng lệnh (ngày/giá/lý do/PnL)")
                    show = pd.DataFrame({
                        "Ngày mua": pd.to_datetime(tr_df["ngay_mua"]).dt.strftime("%d/%m/%Y"),
                        "Giá mua": tr_df["gia_mua"].map(lambda v: f"{v:,.0f}"),
                        "Ngày bán": pd.to_datetime(tr_df["ngay_ban"]).dt.strftime("%d/%m/%Y"),
                        "Giá bán": tr_df["gia_ban"].map(lambda v: f"{v:,.0f}"),
                        "Lý do thoát": tr_df["ly_do"],
                        "PnL (%)": tr_df["pnl_pct"].map(lambda v: f"{v:+.2f}%"),
                    })
                    st.dataframe(show, use_container_width=True, hide_index=True)
            else:
                st.caption("ℹ️ Baseline RF (§4.3.1) chạy toàn kỳ bằng BacktestEngine — không có trade log & markers từng lệnh như chiến lược Multimodal (§3.5.1).")
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(x=bt["eq"].index, y=bt["eq"], mode="lines", name="Vốn AI (RF)", line=dict(color="#2ecc71", width=2.5)))
                fig_eq.add_trace(go.Scatter(x=bt["bhq"].index, y=bt["bhq"], mode="lines", name="Buy & Hold", line=dict(color="#7f8c8d", width=1.5, dash="dot")))
                fig_eq.update_layout(height=460, title="So sánh Tăng trưởng Vốn (Equity Curve)", margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Thời gian", yaxis_title="Giá trị tài sản (VND)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                fig_eq.update_yaxes(tickformat=",.0f")
                st.plotly_chart(fig_eq, use_container_width=True)

st.markdown("---")
st.caption("🎓 Đồ án Tổng hợp — Dự báo giá Top 5 cổ phiếu VN30 bằng Multimodal AI (LSTM-Attention + NLP Sentiment). "
           "Toàn bộ số liệu giao dịch là giả lập phục vụ mục đích học thuật — không phải khuyến nghị đầu tư.")