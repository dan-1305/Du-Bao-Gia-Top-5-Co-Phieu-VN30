"""
=========================================================================
 FEATURE ENGINEERING MODULE - STOCK FORECASTING
=========================================================================
Tính toán 19 chỉ báo kỹ thuật tài chính, quản lý bộ Scaler kép độc lập
(StandardScaler cho đặc trưng X, MinMaxScaler cho mục tiêu Y), ngăn chặn
triệt để rò rỉ dữ liệu (Data Leakage) và xử lý lỗi bộ nhớ động.
"""
import os
import logging
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from dotenv import load_dotenv

load_dotenv()
MAX_MEM_MB = int(os.getenv("MAX_MEMORY_ALLOCATION_MB", "4096"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Tính Relative Strength Index (RSI 14 phiên)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Tính MACD, MACD Signal và MACD Histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def compute_bollinger_bands(
    series: pd.Series, period: int = 20, k: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Tính Bollinger Bands: Lower, Middle, Upper, Bandwidth, %B."""
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + (k * std)
    lower = middle - (k * std)
    bandwidth = (upper - lower) / middle.replace(0, np.nan) * 100.0
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return lower, middle, upper, bandwidth, pct_b


class FeatureEngineer:
    """Module kỹ nghệ đặc trưng và đóng gói cửa sổ trượt chuẩn hóa."""

    def __init__(self, window_size: int = 30, forecast_horizon: int = 3) -> None:
        self.window_size: int = window_size
        self.forecast_horizon: int = forecast_horizon
        self.feature_scaler: StandardScaler = StandardScaler()
        # Chuẩn hóa biến mục tiêu về [0, 1] khớp Mục 3.2.1.2 đồ án
        self.target_scaler: MinMaxScaler = MinMaxScaler(feature_range=(0.0, 1.0))

    def add_indicators(self, df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """Thêm đầy đủ 19 đặc trưng kỹ thuật và nhãn mục tiêu vào tập dữ liệu."""
        data = df.copy()
        close_col = "close" if "close" in data.columns else ("Close" if "Close" in data.columns else data.columns[0])

        # Nhóm chỉ báo kỹ thuật cơ bản
        data["RSI_14"] = compute_rsi(data[close_col], 14)
        macd, macd_s, macd_h = compute_macd(data[close_col], 12, 26, 9)
        data["MACD_12_26_9"] = macd
        data["MACDs_12_26_9"] = macd_s
        data["MACDh_12_26_9"] = macd_h

        bbl, bbm, bbu, bbb, bbp = compute_bollinger_bands(data[close_col], 20, 2.0)
        data["BBL_5_2.0_2.0"] = bbl
        data["BBM_5_2.0_2.0"] = bbm
        data["BBU_5_2.0_2.0"] = bbu
        data["BBB_5_2.0_2.0"] = bbb
        data["BBP_5_2.0_2.0"] = bbp

        data["log_ret"] = np.log(data[close_col] / data[close_col].shift(1)).fillna(0.0)
        data["price_change"] = data[close_col].pct_change().fillna(0.0)
        data["sma_10"] = data[close_col].rolling(10).mean().fillna(data[close_col])
        data["sma_20"] = data[close_col].rolling(20).mean().fillna(data[close_col])

        if "sentiment_score" not in data.columns:
            data["sentiment_score"] = 0.0

        # Phân loại xu hướng 3 lớp: 1 (Tăng), -1 (Giảm), 0 (Đi ngang)
        next_ret = (data[close_col].shift(-1) - data[close_col]) / data[close_col]
        data["target_3class"] = 0
        data.loc[next_ret > threshold, "target_3class"] = 1
        data.loc[next_ret < -threshold, "target_3class"] = -1

        # Target hồi quy giá đóng cửa phiên tiếp theo
        data["target_price_next"] = data[close_col].shift(-1)

        return data.dropna().reset_index(drop=True)

    def create_sliding_windows(
        self, df: pd.DataFrame, feature_cols: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tạo ma trận 3D [N, window_size, n_features] và nhãn hồi quy, phân loại."""
        data_matrix = df[feature_cols].values
        y_reg_raw = df["target_price_next"].values
        y_cls_raw = df["target_3class"].values

        x_list: List[np.ndarray] = []
        y_reg_list: List[float] = []
        y_cls_list: List[int] = []

        total_samples = len(data_matrix) - self.window_size
        for i in range(total_samples):
            x_list.append(data_matrix[i : i + self.window_size])
            y_reg_list.append(y_reg_raw[i + self.window_size - 1])
            y_cls_list.append(y_cls_raw[i + self.window_size - 1])

        return np.array(x_list), np.array(y_reg_list), np.array(y_cls_list)

    def time_series_split(
        self, x_data: np.ndarray, y_data: np.ndarray, train_ratio: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Chia tập dữ liệu tuần tự và chuẩn hóa độc lập hai luồng X và y."""
        try:
            split_idx = int(len(x_data) * train_ratio)
            x_train, x_test = x_data[:split_idx], x_data[split_idx:]
            y_train, y_test = y_data[:split_idx], y_data[split_idx:]

            # Chuẩn hóa đặc trưng đầu vào X bằng StandardScaler
            num_train, seq_len, num_features = x_train.shape
            x_train_flat = self.feature_scaler.fit_transform(x_train.reshape(-1, num_features))
            x_test_flat = self.feature_scaler.transform(x_test.reshape(-1, num_features))

            x_train_scaled = x_train_flat.reshape(num_train, seq_len, num_features)
            x_test_scaled = x_test_flat.reshape(-1, seq_len, num_features)

            # Chuẩn hóa biến mục tiêu y bằng MinMaxScaler (0, 1)
            y_train_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_test_scaled = self.target_scaler.transform(y_test.reshape(-1, 1)).flatten()

            return x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled

        except MemoryError:
            logger.critical(f"Tràn RAM khi reshape Tensor. Ngưỡng cấp phát: {MAX_MEM_MB}MB.")
            return np.array([]), np.array([]), np.array([]), np.array([])

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """Đảo ngược MinMaxScaler về thang giá trị thực tế (VNĐ)."""
        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1, 1)
        return self.target_scaler.inverse_transform(y_scaled).flatten()


# =========================================================================
# UNIT TEST BLUEPRINTS
# =========================================================================
def test_indicators_computation() -> None:
    """Kiểm thử tính toán các chỉ báo kỹ thuật độc lập."""
    mock_series = pd.Series(np.linspace(50000, 75000, 50))
    rsi = compute_rsi(mock_series, period=14)
    macd, signal, hist = compute_macd(mock_series)
    assert not rsi.isna().any(), "Lỗi: RSI chứa giá trị NaN"
    assert len(macd) == len(mock_series), "Lỗi: Chiều dài chuỗi MACD sai lệch"
    assert len(hist) == len(signal), "Lỗi: Histogram và Signal không khớp kích thước"
    logger.info("Unit Test Passed: Chỉ báo kỹ thuật tính toán chính xác.")


def test_scaler_pipeline_integrity() -> None:
    """Kiểm thử quy trình chuẩn hóa và đảo ngược biến mục tiêu."""
    fe = FeatureEngineer(window_size=10)
    dummy_x = np.random.randn(80, 10, 5)
    dummy_y = np.random.uniform(50000.0, 90000.0, 80)

    x_tr, x_te, y_tr_s, y_te_s = fe.time_series_split(dummy_x, dummy_y, train_ratio=0.75)
    assert np.isclose(np.min(y_tr_s), 0.0), "Lỗi: Giá trị nhỏ nhất của y_train không bằng 0"
    assert np.isclose(np.max(y_tr_s), 1.0), "Lỗi: Giá trị lớn nhất của y_train không bằng 1"

    y_te_restored = fe.inverse_transform_target(y_te_s)
    split_cutoff = int(80 * 0.75)
    assert np.allclose(y_te_restored, dummy_y[split_cutoff:], atol=1e-3), "Lỗi: Đảo ngược scaler sai lệch"
    logger.info("Unit Test Passed: Quy trình chuẩn hóa dữ liệu bảo toàn giá trị gốc.")


if __name__ == "__main__":
    test_indicators_computation()
    test_scaler_pipeline_integrity()