"""
=========================================================================
 FEATURE ENGINEERING MODULE - STOCK FORECASTING
=========================================================================
Tính toán các chỉ báo kỹ thuật tài chính (RSI, MACD, Bollinger Bands, Log Ret),
chuẩn hóa dữ liệu và trượt cửa sổ thời gian (Sliding Window Tensor).
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Tính Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Tính MACD, MACD Signal và MACD Histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def compute_bollinger_bands(series: pd.Series, period: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Tính Bollinger Bands: Lower, Middle, Upper, Bandwidth, %B."""
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + (k * std)
    lower = middle - (k * std)
    bandwidth = (upper - lower) / middle.replace(0, np.nan) * 100
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return lower, middle, upper, bandwidth, pct_b


class FeatureEngineer:
    def __init__(self, window_size: int = 30, forecast_horizon: int = 3):
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.scaler = StandardScaler()

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Thêm đầy đủ chỉ báo kỹ thuật vào DataFrame giá thô."""
        df = df.copy()
        close_col = "close" if "close" in df.columns else ("Close" if "Close" in df.columns else df.columns[0])
        
        # Chỉ báo cơ bản
        df['RSI_14'] = compute_rsi(df[close_col], 14)
        macd, macd_s, macd_h = compute_macd(df[close_col], 12, 26, 9)
        df['MACD_12_26_9'] = macd
        df['MACDs_12_26_9'] = macd_s
        df['MACDh_12_26_9'] = macd_h
        
        bbl, bbm, bbu, bbb, bbp = compute_bollinger_bands(df[close_col], 20, 2.0)
        df['BBL_5_2.0_2.0'] = bbl
        df['BBM_5_2.0_2.0'] = bbm
        df['BBU_5_2.0_2.0'] = bbu
        df['BBB_5_2.0_2.0'] = bbb
        df['BBP_5_2.0_2.0'] = bbp
        
        df['log_ret'] = np.log(df[close_col] / df[close_col].shift(1)).fillna(0)
        df['price_change'] = df[close_col].pct_change().fillna(0)
        df['sma_10'] = df[close_col].rolling(10).mean().fillna(df[close_col])
        df['sma_20'] = df[close_col].rolling(20).mean().fillna(df[close_col])

        if 'sentiment_score' not in df.columns:
            df['sentiment_score'] = 0.0

        # Phân loại 3 lớp: 1=Tăng (>1%), -1=Giảm (<-1%), 0=Đi ngang
        next_ret = (df[close_col].shift(-1) - df[close_col]) / df[close_col]
        df['target_3class'] = 0
        df.loc[next_ret > 0.01, 'target_3class'] = 1
        df.loc[next_ret < -0.01, 'target_3class'] = -1
        
        # Target hồi quy giá phiên tiếp theo
        df['target_price_next'] = df[close_col].shift(-1)

        return df.dropna().reset_index(drop=True)

    def create_sliding_windows(self, df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Tạo ma trận 3D Tensor [N, window_size, n_features] cho LSTM/Attention
        và nhãn hồi quy (y_reg) + nhãn phân loại (y_cls).
        """
        data = df[feature_cols].values
        y_reg_raw = df['target_price_next'].values
        y_cls_raw = df['target_3class'].values

        X, y_reg, y_cls = [], [], []
        for i in range(len(data) - self.window_size):
            X.append(data[i:i + self.window_size])
            y_reg.append(y_reg_raw[i + self.window_size - 1])
            y_cls.append(y_cls_raw[i + self.window_size - 1])

        return np.array(X), np.array(y_reg), np.array(y_cls)

    def time_series_split(self, X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8) -> Tuple:
        """Chia tập train/test theo thứ tự thời gian, chuẩn hóa theo train scaler."""
        split_idx = int(len(X) * train_ratio)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        N_tr, T, F = X_train.shape
        X_tr_flat = self.scaler.fit_transform(X_train.reshape(-1, F))
        X_te_flat = self.scaler.transform(X_test.reshape(-1, F))

        return X_tr_flat.reshape(N_tr, T, F), X_te_flat.reshape(-1, T, F), y_train, y_test

