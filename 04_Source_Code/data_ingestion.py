import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional

class DataIngestor:
    """Module chịu trách nhiệm thu thập và làm sạch dữ liệu chứng khoán."""
    def __init__(self, ticker: str, start_date: str, end_date: str):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def fetch_stock_data(self) -> pd.DataFrame:
        """Lấy dữ liệu OHLCV và xử lý dữ liệu trống."""
        print(f"Fetching data for {self.ticker}...")
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date, auto_adjust=True)

        # Xử lý các ngày nghỉ lễ/cuối tuần bị thiếu bằng Forward Fill
        df = df.reindex(pd.date_range(start=self.start_date, end=self.end_date, freq='B'))
        df = df.ffill()

        # Xử lý MultiIndex nếu yfinance trả về
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index.name = 'time'
        return df.dropna()
