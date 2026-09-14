"""
=========================================================================
 BACKTEST ENGINE - FINANCIAL SIMULATION & RISK MANAGEMENT
=========================================================================
Giả lập giao dịch TTCK Việt Nam: Vốn 100tr, phí 0.15%, thuế 0.10%, trượt giá 0.05%,
Hard Stop-Loss (-7%), Take-Profit (+14%), Position Sizing (20%).
"""
import pandas as pd
import numpy as np
from typing import Dict, Any


class BacktestEngine:
    def __init__(
        self,
        initial_capital: float = 100_000_000.0,
        commission: float = 0.0015,
        tax: float = 0.0010,
        slippage: float = 0.0005,
        stop_loss_pct: float = -0.07,
        take_profit_pct: float = 0.14,
        position_size_pct: float = 0.20,
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.tax = tax
        self.slippage = slippage
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.position_size_pct = position_size_pct

    def run(self, df: pd.DataFrame, signals: np.ndarray) -> Dict[str, Any]:
        """df chứa cột 'close', signals chứa nhãn 1 (MUA), -1/0 (BÁN/GIỮ)."""
        close_col = "close" if "close" in df.columns else ("Close" if "Close" in df.columns else df.columns[0])
        prices = df[close_col].values
        dates = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.get("time", df.index))

        capital = float(self.initial_capital)
        shares = 0.0
        buy_price = 0.0
        equity_history = []
        winning_trades = 0
        total_closed_trades = 0

        for i in range(len(prices)):
            p = float(prices[i])
            sig = int(signals[i]) if i < len(signals) else 0

            if shares > 0:
                ret_unrealized = (p - buy_price) / buy_price
                # 1. Stop-loss
                if ret_unrealized <= self.stop_loss_pct:
                    sell_p = p * (1.0 - self.slippage)
                    gross = shares * sell_p
                    cost = gross * (self.commission + self.tax)
                    capital += gross - cost
                    shares = 0.0
                    buy_price = 0.0
                    total_closed_trades += 1
                # 2. Take-profit
                elif ret_unrealized >= self.take_profit_pct:
                    sell_p = p * (1.0 - self.slippage)
                    gross = shares * sell_p
                    cost = gross * (self.commission + self.tax)
                    capital += gross - cost
                    shares = 0.0
                    buy_price = 0.0
                    winning_trades += 1
                    total_closed_trades += 1
                # 3. AI Sell
                elif sig in [-1, 0] and sig != 1:
                    sell_p = p * (1.0 - self.slippage)
                    gross = shares * sell_p
                    cost = gross * (self.commission + self.tax)
                    capital += gross - cost
                    if ret_unrealized > 0:
                        winning_trades += 1
                    shares = 0.0
                    buy_price = 0.0
                    total_closed_trades += 1
            elif sig == 1 and shares == 0:
                allocated = capital * self.position_size_pct
                buy_p = p * (1.0 + self.slippage)
                cost = allocated * self.commission
                net_alloc = allocated - cost
                if net_alloc > 0 and buy_p > 0:
                    shares = net_alloc / buy_p
                    capital -= allocated
                    buy_price = buy_p

            current_nav = capital + (shares * p)
            equity_history.append(current_nav)

        equity_series = pd.Series(equity_history, index=dates)
        bh_shares = (self.initial_capital * (1.0 - self.commission)) / (prices[0] * (1.0 + self.slippage))
        bh_equity = pd.Series(bh_shares * prices, index=dates)

        final_equity = equity_history[-1]
        cumulative_return = (final_equity - self.initial_capital) / self.initial_capital * 100.0
        bh_return = (bh_equity.iloc[-1] - self.initial_capital) / self.initial_capital * 100.0

        daily_ret = equity_series.pct_change().dropna()
        sharpe = float((daily_ret.mean() / daily_ret.std()) * np.sqrt(252)) if daily_ret.std() > 0 else 0.0
        peak = equity_series.cummax()
        dd = (equity_series - peak) / peak
        max_dd = float(dd.min()) * 100.0
        win_rate = (winning_trades / total_closed_trades * 100.0) if total_closed_trades > 0 else 0.0

        return {
            "initial_capital": self.initial_capital,
            "final_equity": round(final_equity, 2),
            "cumulative_return_pct": round(cumulative_return, 2),
            "buy_and_hold_return_pct": round(bh_return, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "win_rate_pct": round(win_rate, 2),
            "total_trades": total_closed_trades,
            "equity_curve": equity_series,
            "buy_and_hold_curve": bh_equity,
        }

