"""
=========================================================================
 TRAIN MODELS - Stock Forecasting Project
=========================================================================
Train Random Forest + SVM (phân loại Tăng/Giảm) và LSTM (hồi quy giá)
trên dữ liệu processed_data.csv. Lưu toàn bộ model vào 03_Models/.

 -------------------------------------------------------------------------
 CÁCH CHẠY TRÊN GOOGLE COLAB (KHUYẾN NGHỊ - có GPU/RAM dồi dào):
 -------------------------------------------------------------------------
    1. Upload cả folder Stock_Forecasting_Project lên Google Drive.
    2. Trong Colab cell đầu tiên:
         from google.colab import drive
         drive.mount('/content/drive')
       Rồi sửa PROJECT_DIR ở phần CONFIG PATH bên dưới cho trỏ tới
       folder trong Drive, ví dụ:
         PROJECT_DIR = Path('/content/drive/MyDrive/DeTaiHocSauKetHopDoAnTongHop/Stock_Forecasting_Project')
    3. Chạy file này (Run all). Sẽ sinh ra 03_Models/*.pkl, *.keras, *.json.
    4. Tải cả folder 03_Models/ về máy local để app.py Streamlit load.

 -------------------------------------------------------------------------
 CÁCH CHẠY LOCAL (chỉ RF/SVM, vì TF tốn RAM):
 -------------------------------------------------------------------------
    cd Stock_Forecasting_Project/04_Source_Code
    python train_models.py --no-lstm
-------------------------------------------------------------------------
Author: CEO Sovereign - LangGraph Agent System
=========================================================================
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
import joblib

# =====================================================================
# CONFIG PATH
# =====================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
# Mặc định local: cha của 04_Source_Code chính là thư mục project
PROJECT_DIR = SCRIPT_DIR.parent

# >>> COLAB: BỎ COMMENT DÒNG DƯỚI & SỬA ĐƯỜNG DẪN CHO ĐÚNG VỚI DRIVE <<<
# PROJECT_DIR = Path('/content/drive/MyDrive/DeTaiHocSauKetHopDoAnTongHop/Stock_Forecasting_Project')

DATA_PATH = PROJECT_DIR / "01_Data" / "processed_data.csv"
MODELS_DIR = PROJECT_DIR / "03_Models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# 1. LOAD & CHUẨN BỊ DỮ LIỆU
# =====================================================================
def load_data() -> pd.DataFrame:
    """Load processed_data.csv, ép kiểu ngày, sort theo thời gian."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {DATA_PATH}. Kiểm tra lại đường dẫn PROJECT_DIR."
        )
    df = pd.read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    print(f"[Data] Loaded {len(df)} rows | {df['time'].min().date()} -> {df['time'].max().date()}")
    print(f"[Data] Ticker: {df['ticker'].unique().tolist()}")
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Xây dựng feature set cho bài toán phân loại xu hướng.
    Trả về (df_featured, feature_columns).
    """
    df = df.copy()

    # Feature kỹ thuật bổ sung (an toàn với time-series)
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["price_change"] = df["close"].pct_change()

    # Chọn feature columns (loại time, ticker, target ra)
    drop_cols = {"time", "ticker", "target"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Bỏ các dòng NaN do tạo feature (log_ret, sma)
    df_clean = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
    print(f"[Features] {len(feature_cols)} features: {feature_cols}")
    return df_clean, feature_cols


def time_series_split(df: pd.DataFrame, train_ratio: float = 0.8):
    """
    Time-series split: 80% đầu = train, 20% cuối = test.
    KHÔNG shuffle để tránh data leakage (quan trọng cho đồ án tốt nghiệp).
    """
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    print(f"[Split] Train: {len(train)} rows | Test: {len(test)} rows")
    print(f"        Train: {train['time'].min().date()} -> {train['time'].max().date()}")
    print(f"        Test : {test['time'].min().date()} -> {test['time'].max().date()}")
    return train, test


# =====================================================================
# 2. TRAIN CLASSIFIERS (RF + SVM)
# =====================================================================
def train_classifiers(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Train Random Forest + SVM, trả về dict metrics + scaler."""
    X_train = train[feature_cols].values
    y_train = train["target"].values
    X_test = test[feature_cols].values
    y_test = test["target"].values

    # StandardScaler (bắt buộc cho SVM)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Lưu scaler + feature columns (app.py sẽ dùng để transform input mới)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    with open(MODELS_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    print(f"[Save] scaler.pkl + feature_columns.json")

    metrics = {}

    # --- Random Forest ---
    print("\n[Train] Random Forest ...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    y_pred_rf = rf.predict(X_test_s)
    joblib.dump(rf, MODELS_DIR / "rf_model.pkl")
    joblib.dump(rf.classes_.tolist(), MODELS_DIR / "rf_classes.pkl")
    metrics["random_forest"] = _eval("Random Forest", y_test, y_pred_rf)
    # Lưu feature importance (đẹp cho báo cáo đồ án)
    fi = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    fi.to_csv(MODELS_DIR / "rf_feature_importance.csv")
    print(f"[Save] rf_model.pkl | Top 5 features:\n{fi.head().to_string()}")

    # --- SVM ---
    print("\n[Train] SVM (RBF kernel) ...")
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
    svm.fit(X_train_s, y_train)
    y_pred_svm = svm.predict(X_test_s)
    joblib.dump(svm, MODELS_DIR / "svm_model.pkl")
    metrics["svm"] = _eval("SVM", y_test, y_pred_svm)
    print(f"[Save] svm_model.pkl")

    return metrics


def _eval(name: str, y_true, y_pred) -> dict:
    """Tính metrics + in báo cáo."""
    n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0, average="macro")
    prec = precision_score(y_true, y_pred, zero_division=0, average="macro")
    rec = recall_score(y_true, y_pred, zero_division=0, average="macro")
    cm = confusion_matrix(y_true, y_pred).tolist()
    print(f"  [{name}] Accuracy={acc:.4f} | Macro F1={f1:.4f} | Precision={prec:.4f} | Recall={rec:.4f}")
    return {
        "accuracy": round(acc, 4), "f1": round(f1, 4),
        "precision": round(prec, 4), "recall": round(rec, 4),
        "n_classes": int(n_classes),
        "confusion_matrix": cm,
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }


def train_pytorch_multimodal(df_feat: pd.DataFrame, feature_cols: list[str], window: int = 30) -> dict:
    """Huấn luyện Multimodal PyTorch model (LSTM + Attention + Sentiment Fusion)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from feature_engineering import FeatureEngineer
    from model_architecture import MultimodalStockModel
    from backtest_engine import BacktestEngine
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    print(f"\n[PyTorch] Huấn luyện Multimodal Model (LSTM-Attention + Sentiment) | Window={window}")
    fe = FeatureEngineer(window_size=window)

    X_ts, y_reg_raw, y_cls_raw = fe.create_sliding_windows(df_feat, feature_cols)
    
    sent_scores = df_feat["sentiment_score"].values[window - 1: -1]
    if len(sent_scores) < len(X_ts):
        sent_scores = np.pad(sent_scores, (0, len(X_ts) - len(sent_scores)), mode='edge')
    
    nlp_feats = []
    for s in sent_scores:
        if s > 0.1:
            nlp_feats.append([0.7, 0.1, 0.2])
        elif s < -0.1:
            nlp_feats.append([0.1, 0.7, 0.2])
        else:
            nlp_feats.append([0.2, 0.2, 0.6])
    X_nlp = np.array(nlp_feats, dtype=np.float32)

    cls_map = {-1: 0, 0: 1, 1: 2}
    y_cls = np.array([cls_map.get(v, 1) for v in y_cls_raw], dtype=np.int64)

    split_idx = int(len(X_ts) * 0.8)
    X_tr_ts, X_te_ts = X_ts[:split_idx], X_ts[split_idx:]
    X_tr_nlp, X_te_nlp = X_nlp[:split_idx], X_nlp[split_idx:]
    y_tr_reg, y_te_reg = y_reg_raw[:split_idx], y_reg_raw[split_idx:]
    y_tr_cls, y_te_cls = y_cls[:split_idx], y_cls[split_idx:]

    ts_scaler = StandardScaler()
    N_tr, T, F = X_tr_ts.shape
    X_tr_ts_s = ts_scaler.fit_transform(X_tr_ts.reshape(-1, F)).reshape(N_tr, T, F)
    X_te_ts_s = ts_scaler.transform(X_te_ts.reshape(-1, F)).reshape(-1, T, F)

    y_scaler = MinMaxScaler(feature_range=(0, 1))
    y_tr_reg_s = y_scaler.fit_transform(y_tr_reg.reshape(-1, 1)).ravel()

    joblib.dump(ts_scaler, MODELS_DIR / "lstm_scaler.pkl")
    joblib.dump(y_scaler, MODELS_DIR / "lstm_y_scaler.pkl")

    train_ds = TensorDataset(
        torch.tensor(X_tr_ts_s, dtype=torch.float32),
        torch.tensor(X_tr_nlp, dtype=torch.float32),
        torch.tensor(y_tr_reg_s, dtype=torch.float32),
        torch.tensor(y_tr_cls, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    model = MultimodalStockModel(n_features=F, sequence_length=T, nlp_dim=3, hidden_lstm=32, hidden_dense=24)
    criterion_huber = nn.SmoothL1Loss()
    criterion_ce = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)

    model.train()
    for epoch in range(1, 21):
        total_loss = 0.0
        for b_ts, b_nlp, b_yreg, b_ycls in train_loader:
            optimizer.zero_grad()
            pred_p, logits_trend, _ = model(b_ts, b_nlp)
            loss_reg = criterion_huber(pred_p, b_yreg)
            loss_cls = criterion_ce(logits_trend, b_ycls)
            loss = loss_reg + 0.5 * loss_cls
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 5 == 0:
            print(f"  [Epoch {epoch:02d}/20] Loss: {total_loss/len(train_loader):.4f}")


    model.eval()
    with torch.no_grad():
        t_te_ts = torch.tensor(X_te_ts_s, dtype=torch.float32)
        t_te_nlp = torch.tensor(X_te_nlp, dtype=torch.float32)
        pred_p_s, logits_trend, _ = model(t_te_ts, t_te_nlp)
        
        y_pred_reg = y_scaler.inverse_transform(pred_p_s.numpy().reshape(-1, 1)).ravel()
        y_pred_cls = torch.argmax(logits_trend, dim=1).numpy()

    rmse = float(np.sqrt(mean_squared_error(y_te_reg, y_pred_reg)))
    mae = float(mean_absolute_error(y_te_reg, y_pred_reg))
    mape = float(np.mean(np.abs((y_te_reg - y_pred_reg) / y_te_reg)) * 100.0)
    dir_acc = float(np.mean(np.sign(y_te_reg[1:] - y_te_reg[:-1]) == np.sign(y_pred_reg[1:] - y_pred_reg[:-1])))
    
    inv_map = {0: -1, 1: 0, 2: 1}
    y_te_cls_orig = np.array([inv_map[v] for v in y_te_cls])
    y_pred_cls_orig = np.array([inv_map[v] for v in y_pred_cls])
    acc_cls = float(accuracy_score(y_te_cls_orig, y_pred_cls_orig))
    f1_cls = float(f1_score(y_te_cls_orig, y_pred_cls_orig, average="macro", zero_division=0))

    print(f"\n[Multimodal Eval] RMSE={rmse:,.0f} đ | MAE={mae:,.0f} đ | MAPE={mape:.2f}% | DirectionAcc={dir_acc*100:.1f}% | Macro F1={f1_cls:.4f}")

    be = BacktestEngine(initial_capital=100_000_000.0)
    test_df_slice = df_feat.iloc[split_idx + window:].reset_index(drop=True)
    backtest_res = be.run(test_df_slice, y_pred_cls_orig[:len(test_df_slice)])
    print(f"[Backtest] ROI={backtest_res['cumulative_return_pct']}% | Sharpe={backtest_res['sharpe_ratio']} | MaxDD={backtest_res['max_drawdown_pct']}%")

    torch.save(model.state_dict(), MODELS_DIR / "multimodal_model.pt")
    with open(MODELS_DIR / "multimodal_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_features": F, "sequence_length": T, "nlp_dim": 3,
            "hidden_lstm": 32, "hidden_dense": 24
        }, f, indent=2)

    return {
        "rmse": round(rmse, 2), "mae": round(mae, 2), "mape": round(mape, 2),
        "direction_accuracy": round(dir_acc, 4), "accuracy_cls": round(acc_cls, 4),
        "macro_f1_cls": round(f1_cls, 4),
        "backtest": {
            "cumulative_return_pct": backtest_res["cumulative_return_pct"],
            "buy_and_hold_return_pct": backtest_res["buy_and_hold_return_pct"],
            "sharpe_ratio": backtest_res["sharpe_ratio"],
            "max_drawdown_pct": backtest_res["max_drawdown_pct"],
            "win_rate_pct": backtest_res["win_rate_pct"],
            "total_trades": backtest_res["total_trades"],
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Train Stock Forecasting models")
    parser.add_argument("--no-multimodal", action="store_true", help="Bỏ qua PyTorch Multimodal")
    args = parser.parse_args()

    print("=" * 60)
    print("  STOCK FORECASTING - TRAIN MODELS PIPELINE")
    print(f"  Project dir: {PROJECT_DIR}")
    print(f"  Data:        {DATA_PATH}")
    print(f"  Output:      {MODELS_DIR}")
    print("=" * 60)

    df = load_data()
    from feature_engineering import FeatureEngineer
    fe = FeatureEngineer(window_size=30)
    df_feat = fe.add_indicators(df)
    drop_cols = {"time", "ticker", "target", "target_3class", "target_price_next"}
    feature_cols = [c for c in df_feat.columns if c not in drop_cols]
    
    split_idx = int(len(df_feat) * 0.8)
    train_df, test_df = df_feat.iloc[:split_idx].copy(), df_feat.iloc[split_idx:].copy()

    all_metrics = {}
    all_metrics["classification"] = train_classifiers(train_df, test_df, feature_cols)

    if not args.no_multimodal:
        all_metrics["multimodal_deep_learning"] = train_pytorch_multimodal(df_feat, feature_cols, window=30)

    metrics_path = MODELS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"\n[Save] {metrics_path.name}")

    print("\n" + "=" * 60)
    print("  ✅ TRAINING HOÀN TẤT")
    for p in sorted(MODELS_DIR.glob("*")):
        print(f"    - {p.name} ({p.stat().st_size:,} bytes)")
    print("=" * 60)


if __name__ == "__main__":
    main()



