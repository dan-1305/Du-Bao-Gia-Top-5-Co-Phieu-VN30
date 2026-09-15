# 📈 Stock Forecasting Project — Dự báo giá Top 5 cổ phiếu VN30 bằng Multimodal AI

Đồ án Tổng hợp: dự báo giá cổ phiếu **FPT · HPG · MBB · MWG · VNM** bằng kiến trúc
**Multimodal LSTM-Attention** (chuỗi thời gian + tâm lý tin tức NLP), kèm hệ thống
**giả lập giao dịch (backtest)** chuẩn HOSE và dashboard Streamlit trực quan.

---

## 🗂️ Cấu trúc thư mục

```
Stock_Forecasting_Project/
├── 01_Data/               # Dữ liệu giá đã xử lý (processed_*.csv) + tin tức đã crawl (news_*.csv)
├── 02_Notebooks/          # (đã gộp vào 04_Source_Code — giữ chỗ theo cấu trúc nộp bài)
├── 03_Models/             # Model + scaler + artifact kết quả (metrics.json, top5_results_v3.json, ...)
├── 04_Source_Code/        # Toàn bộ mã nguồn: modules .py + notebooks .ipynb
│   ├── app.py                  # Dashboard Streamlit (DEMO)
│   ├── train_models.py         # Pipeline train RF/SVM/Multimodal
│   ├── model_architecture.py   # Class MultimodalStockModel (LSTM-Attention + NLP fusion)
│   ├── backtest_engine.py      # Engine giả lập giao dịch chuẩn HOSE
│   ├── backtest_threshold.py   # Backtest ngưỡng §3.5.1 (MUA > +1% + đồng thuận sentiment)
│   ├── nlp_processor.py        # Lexicon tiếng Việt 38 cụm (+ PhoBERT/FinBERT tùy chọn)
│   ├── feature_engineering.py  # RSI/MACD/BB/SMA thuần pandas
│   ├── data_ingestion.py       # Tải dữ liệu OHLCV (yfinance)
│   └── 1→4_*.ipynb             # Notebooks theo từng chương báo cáo
├── 05_Docs_Reports/       # Ảnh demo + biểu đồ phục vụ báo cáo
└── requirements.txt
```

## 🚀 Chạy demo

### Cách 1 — Local (VS Code)
```bash
pip install -r requirements.txt
streamlit run 04_Source_Code/app.py --server.port 8599
```
Mở trình duyệt → `http://localhost:8599`.

### Cách 2 — Google Colab (backup)
1. Tải thư mục project lên Drive (giữ nguyên cấu trúc)
2. Mở `04_Source_Code/0_RUN_DEMO.ipynb` trong Colab → **Run All**
3. Notebook tự: gắn Drive → cài thư viện → chạy Streamlit → in **URL public (localtunnel)** + **IP mật khẩu**
4. Mở URL → dán IP vào ô *Tunnel password* → dashboard chạy trực tiếp trên trình duyệt

### 🎬 Kịch bản demo 5 phút
1. **Tab 1** (1.5'): biểu đồ + đường dự báo 5 phiên → mở expander **🏆 Chất lượng Mô hình** (DA/MAPE + bảng tin-vs-neutral trung thực)
2. **Tab 2** (1'): phân tích cảm xúc nhập tay + bảng tin RSS đã crawl (HPG/MBB/MWG/VNM)
3. **Tab 3** (2'): giữ Multimodal §3.5.1 → ▶ Chạy Backtest → đồ thị MUA▲/BÁN▼ + trade log → đổi nguồn tín hiệu sang RF baseline (FPT) đối chiếu
4. **30s cuối**: hover icon **?** bất kỳ để chứng minh dashboard tự giải thích thuật ngữ; trả lời phản biện bằng expander "🎯 Vì sao có lựa chọn nguồn tín hiệu?"

Dashboard có 3 tab:
1. **📊 Dự Báo & Biểu Đồ** — đường dự báo 5 phiên (LSTM-Attention), expander chất lượng mô hình
2. **📰 Phân Tích Cảm Xúc** — phân tích tay + bảng 100 tin RSS đã crawl kèm điểm Lexicon
3. **💰 Giả Lập Giao Dịch (Backtest)** — chạy backtest theo ngưỡng §3.5.1 hoặc baseline RF §4.3.1

> Mỗi thuật ngữ chuyên ngành (ROI AI, Sharpe, MaxDD...) đều có icon **?** — hover để xem giải thích.

## 📓 Notebooks

| Notebook | Nội dung | Chương | Môi trường |
|---|---|---|---|
| `0_RUN_DEMO.ipynb` | **Launcher demo 1-click** (Streamlit + localtunnel) | — | Colab + Local |
| `1_Data_Pipeline.ipynb` | Thu thập & tiền xử lý dữ liệu | Chương 2 | Colab + Local |
| `2_ML_Baselines.ipynb` | Baseline RF/SVM phân loại xu hướng | §4.3.1 | Colab + Local |
| `3_DL_LSTM_Models.ipynb` | Multimodal LSTM-Attention + training | Chương 3 | Colab + Local |
| `4_Backtesting_Eval.ipynb` | Backtest & đánh giá (đọc artifact thật) | §4.4 | Colab + Local |

## 🏆 Kết quả chính

| Chỉ số | Giá trị |
|---|---|
| Direction Accuracy (FPT) | **54.26%** · MAPE 5.39% |
| Baseline RF/SVM accuracy | ~34% |
| Backtest AI (FPT, 259 phiên test) | **−4.15%** vs Buy & Hold **−27.55%** |
| Max Drawdown AI | −8.45% (giảm ~85% tổn thất trong giai đoạn xuống) |

*(Số liệu chi tiết: `03_Models/metrics.json`, `03_Models/top5_results_v3.json`, `03_Models/multimodal_backtest_real.json`)*

## ⚠️ Miễn trừ

Toàn bộ giao dịch trong demo là **giả lập phục vụ học thuật** — không phải khuyến nghị đầu tư.
