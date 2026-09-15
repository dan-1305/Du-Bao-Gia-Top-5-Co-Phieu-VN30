"""
=========================================================================
  THRESHOLD BACKTEST MODULE (Demo Rework - Goi A)
=========================================================================
Chien luoc nguong theo dung do an 3.5.1:
  MUA khi r_pred > +1.0% (dong thuan sentiment > 0)
  BAN khi r_pred < -1.0% hoac sentiment < -0.3
  SL/TP/phi/thue/slippage trung khop BacktestEngine.
Input: pred_df co cot [time, close, r_pred, sent] (tap kiem thu).
Output: dict equity/bh/drawdown + metrics + trade log tung lenh.
"""
from typing import Dict, Any
import numpy as np
import pandas as pd


def run_threshold_backtest(
    pred_df: pd.DataFrame,
    init_capital: float = 100_000_000.0,
    commission: float = 0.0015,
    tax: float = 0.0010,
    slippage: float = 0.0005,
    stop_loss: float = -0.07,
    take_profit: float = 0.14,
    pos_size: float = 0.20,
    buy_thr: float = 0.01,
    sell_thr: float = -0.01,
    sent_sell: float = -0.3,
    use_sentiment: bool = True,
) -> Dict[str, Any]:
    """Chay backtest nguong tren DataFrame du bao; tra metrics + trade log."""
    times = pd.to_datetime(pred_df["time"]).reset_index(drop=True)
    cur = pred_df["close"].values.astype(float)
    rp = pred_df["r_pred"].values.astype(float)
    if "sent" in pred_df.columns:
        st = pred_df["sent"].values.astype(float)
    else:
        st = np.zeros(len(cur))
    n = len(cur)
    cap = float(init_capital)
    shares = 0.0
    buy_p = 0.0
    buy_i = -1
    eq: list = []
    trades: list = []
    wins = 0
    for i in range(n):
        p = cur[i]
        if shares > 0:
            unreal = (p - buy_p) / buy_p
            reason = None
            if unreal <= stop_loss:
                reason = "STOP-LOSS"
            elif unreal >= take_profit:
                reason = "TAKE-PROFIT"
            elif (rp[i] < sell_thr) or (use_sentiment and st[i] < sent_sell):
                reason = "AI-SELL"
            if reason:
                sell_p = p * (1 - slippage)
                gross = shares * sell_p
                cap += gross - gross * (commission + tax)
                pnl = (sell_p - buy_p) / buy_p * 100.0
                if unreal > 0:
                    wins += 1
                trades.append({
                    "ngay_mua": times[buy_i], "gia_mua": round(buy_p, 1),
                    "ngay_ban": times[i], "gia_ban": round(p, 1),
                    "ly_do": reason, "pnl_pct": round(pnl, 2),
                })
                shares = 0.0
                buy_p = 0.0
        else:
            if (rp[i] > buy_thr) and ((st[i] > 0) if use_sentiment else True):
                alloc = cap * pos_size
                bp = p * (1 + slippage)
                net = alloc - alloc * commission
                if net > 0 and bp > 0:
                    shares = net / bp
                    cap -= alloc
                    buy_p = bp
                    buy_i = i
        eq.append(cap + shares * p)
    eqs = pd.Series(eq, index=times)
    bh_sh = init_capital * (1 - commission) / (cur[0] * (1 + slippage))
    bh = pd.Series(bh_sh * cur, index=times)
    dr = eqs.pct_change().dropna()
    sharpe = float((dr.mean() / dr.std()) * np.sqrt(252)) if len(dr) > 2 and dr.std() > 0 else 0.0
    peak = eqs.cummax()
    mdd = float(((eqs - peak) / peak).min()) * 100.0
    ret = (eqs.iloc[-1] - init_capital) / init_capital * 100.0
    bhr = (bh.iloc[-1] - init_capital) / init_capital * 100.0
    dd_series = (eqs - peak) / peak * 100.0
    return {
        "equity": eqs, "bh": bh, "dd": dd_series, "times": times, "close": cur,
        "roi_pct": round(ret, 2), "bh_pct": round(bhr, 2), "sharpe": round(sharpe, 2),
        "maxdd_pct": round(mdd, 2),
        "win_rate": round(wins / len(trades) * 100, 2) if trades else 0.0,
        "n_trades": len(trades), "trades": trades,
    }
