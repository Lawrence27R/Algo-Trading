import json
import os
import datetime
from collections import defaultdict

TRADE_LOG_PATH = "data/logs/trades.json"
PORTFOLIO_PATH = "data/logs/portfolio.json"


def load_trades():
    if os.path.exists(TRADE_LOG_PATH):
        with open(TRADE_LOG_PATH) as f:
            return json.load(f)
    return []


def load_portfolio():
    if os.path.exists(PORTFOLIO_PATH):
        with open(PORTFOLIO_PATH) as f:
            return json.load(f)
    return {"cash": 100000, "positions": {}, "pnl_history": []}


def calc_brokerage(transaction_type, quantity, price, mode="paper"):
    """
    Calculates Indian stock market charges for intraday equity trades.
    Returns a dict with all charge components.
    """
    turnover = quantity * price

    # Zerodha-style flat brokerage (Rs 20 or 0.03% whichever is lower per leg)
    brokerage = min(20, turnover * 0.0003)

    # STT - 0.025% on sell side for intraday
    stt = turnover * 0.00025 if transaction_type == "SELL" else 0

    # Exchange Transaction Charge (NSE) - 0.00325%
    exchange_charge = turnover * 0.0000325

    # SEBI Charges - Rs 10 per crore = 0.000001
    sebi_charge = turnover * 0.000001

    # GST - 18% on (brokerage + exchange charge + SEBI charge)
    gst = (brokerage + exchange_charge + sebi_charge) * 0.18

    # Stamp Duty - 0.003% on buy side
    stamp_duty = turnover * 0.00003 if transaction_type == "BUY" else 0

    total_charges = brokerage + stt + exchange_charge + sebi_charge + gst + stamp_duty

    return {
        "turnover": round(turnover, 2),
        "brokerage": round(brokerage, 2),
        "stt": round(stt, 2),
        "exchange_charge": round(exchange_charge, 4),
        "sebi_charge": round(sebi_charge, 4),
        "gst": round(gst, 4),
        "stamp_duty": round(stamp_duty, 4),
        "total_charges": round(total_charges, 2)
    }


def enrich_trades_with_charges(trades):
    """Add brokerage/charges to each trade record."""
    enriched = []
    for t in trades:
        charges = calc_brokerage(
            t.get("transaction_type", "BUY"),
            t.get("quantity", 0),
            t.get("price", 0),
            t.get("mode", "paper")
        )
        enriched.append({**t, "charges": charges})
    return enriched


def get_per_trade_report(trades=None):
    """Full per-trade report with charges and net P&L."""
    if trades is None:
        trades = load_trades()
    enriched = enrich_trades_with_charges(trades)

    # Match BUY and SELL for P&L per round-trip
    buy_stack = defaultdict(list)
    report = []

    for t in sorted(enriched, key=lambda x: x.get("timestamp", "")):
        symbol = t["symbol"]
        charges = t["charges"]

        if t["transaction_type"] == "BUY":
            buy_stack[symbol].append(t)
        elif t["transaction_type"] == "SELL" and buy_stack[symbol]:
            buy_trade = buy_stack[symbol].pop(0)
            buy_charges = buy_trade["charges"]
            gross_pnl = (t["price"] - buy_trade["price"]) * t["quantity"]
            total_charges = charges["total_charges"] + buy_charges["total_charges"]
            net_pnl = gross_pnl - total_charges

            report.append({
                "symbol": symbol,
                "buy_time": buy_trade["timestamp"],
                "sell_time": t["timestamp"],
                "quantity": t["quantity"],
                "buy_price": buy_trade["price"],
                "sell_price": t["price"],
                "gross_pnl": round(gross_pnl, 2),
                "total_charges": round(total_charges, 2),
                "net_pnl": round(net_pnl, 2),
                "mode": t["mode"],
                "buy_charges": buy_charges,
                "sell_charges": charges,
            })

    return report


def get_daily_report(trades=None):
    """Aggregate P&L, charges, trade count per day."""
    per_trade = get_per_trade_report(trades)
    daily = defaultdict(lambda: {
        "date": "",
        "trades": 0,
        "gross_pnl": 0,
        "total_charges": 0,
        "net_pnl": 0,
        "wins": 0,
        "losses": 0,
        "symbols": set()
    })

    for t in per_trade:
        date = t["sell_time"][:10]
        d = daily[date]
        d["date"] = date
        d["trades"] += 1
        d["gross_pnl"] += t["gross_pnl"]
        d["total_charges"] += t["total_charges"]
        d["net_pnl"] += t["net_pnl"]
        d["wins"] += 1 if t["net_pnl"] > 0 else 0
        d["losses"] += 1 if t["net_pnl"] <= 0 else 0
        d["symbols"].add(t["symbol"])

    result = []
    for date, d in sorted(daily.items()):
        d["gross_pnl"] = round(d["gross_pnl"], 2)
        d["total_charges"] = round(d["total_charges"], 2)
        d["net_pnl"] = round(d["net_pnl"], 2)
        d["symbols"] = list(d["symbols"])
        result.append(d)
    return result


def get_weekly_report(trades=None):
    """Aggregate P&L per ISO week."""
    per_trade = get_per_trade_report(trades)
    weekly = defaultdict(lambda: {
        "week": "", "trades": 0, "gross_pnl": 0,
        "total_charges": 0, "net_pnl": 0, "wins": 0, "losses": 0
    })

    for t in per_trade:
        dt = datetime.datetime.fromisoformat(t["sell_time"])
        week_key = f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
        d = weekly[week_key]
        d["week"] = week_key
        d["trades"] += 1
        d["gross_pnl"] += t["gross_pnl"]
        d["total_charges"] += t["total_charges"]
        d["net_pnl"] += t["net_pnl"]
        d["wins"] += 1 if t["net_pnl"] > 0 else 0
        d["losses"] += 1 if t["net_pnl"] <= 0 else 0

    result = []
    for key, d in sorted(weekly.items()):
        d["gross_pnl"] = round(d["gross_pnl"], 2)
        d["total_charges"] = round(d["total_charges"], 2)
        d["net_pnl"] = round(d["net_pnl"], 2)
        result.append(d)
    return result


def get_monthly_report(trades=None):
    """Aggregate P&L per calendar month."""
    per_trade = get_per_trade_report(trades)
    monthly = defaultdict(lambda: {
        "month": "", "trades": 0, "gross_pnl": 0,
        "total_charges": 0, "net_pnl": 0, "wins": 0, "losses": 0
    })

    for t in per_trade:
        month_key = t["sell_time"][:7]
        d = monthly[month_key]
        d["month"] = month_key
        d["trades"] += 1
        d["gross_pnl"] += t["gross_pnl"]
        d["total_charges"] += t["total_charges"]
        d["net_pnl"] += t["net_pnl"]
        d["wins"] += 1 if t["net_pnl"] > 0 else 0
        d["losses"] += 1 if t["net_pnl"] <= 0 else 0

    result = []
    for key, d in sorted(monthly.items()):
        d["gross_pnl"] = round(d["gross_pnl"], 2)
        d["total_charges"] = round(d["total_charges"], 2)
        d["net_pnl"] = round(d["net_pnl"], 2)
        result.append(d)
    return result


def get_summary_stats(trades=None):
    """Overall performance statistics."""
    per_trade = get_per_trade_report(trades)
    if not per_trade:
        return {}

    pnls = [t["net_pnl"] for t in per_trade]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total_charges = sum(t["total_charges"] for t in per_trade)
    gross_pnl = sum(t["gross_pnl"] for t in per_trade)
    net_pnl = sum(pnls)

    # Max drawdown
    cumulative = []
    running = 0
    for p in pnls:
        running += p
        cumulative.append(running)
    peak = cumulative[0]
    max_drawdown = 0
    for val in cumulative:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_drawdown:
            max_drawdown = dd

    # Sharpe Ratio (simplified, annualized assuming 252 trading days)
    import statistics
    avg = statistics.mean(pnls) if pnls else 0
    std = statistics.stdev(pnls) if len(pnls) > 1 else 1
    sharpe = (avg / std) * (252 ** 0.5) if std != 0 else 0

    return {
        "total_trades": len(per_trade),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": round(len(wins) / len(per_trade) * 100, 1) if per_trade else 0,
        "gross_pnl": round(gross_pnl, 2),
        "total_charges": round(total_charges, 2),
        "net_pnl": round(net_pnl, 2),
        "avg_profit_per_trade": round(sum(wins) / len(wins), 2) if wins else 0,
        "avg_loss_per_trade": round(sum(losses) / len(losses), 2) if losses else 0,
        "best_trade": round(max(pnls), 2) if pnls else 0,
        "worst_trade": round(min(pnls), 2) if pnls else 0,
        "max_drawdown": round(max_drawdown, 2),
        "sharpe_ratio": round(sharpe, 2),
        "profit_factor": round(sum(wins) / abs(sum(losses)), 2) if losses and sum(losses) != 0 else (float('inf') if wins else 0),
        "avg_charges_per_trade": round(total_charges / len(per_trade), 2) if per_trade else 0,
    }


def get_symbol_report(trades=None):
    """P&L breakdown per symbol."""
    per_trade = get_per_trade_report(trades)
    symbol_map = defaultdict(lambda: {
        "symbol": "", "trades": 0, "gross_pnl": 0,
        "total_charges": 0, "net_pnl": 0, "wins": 0, "losses": 0
    })
    for t in per_trade:
        s = symbol_map[t["symbol"]]
        s["symbol"] = t["symbol"]
        s["trades"] += 1
        s["gross_pnl"] += t["gross_pnl"]
        s["total_charges"] += t["total_charges"]
        s["net_pnl"] += t["net_pnl"]
        s["wins"] += 1 if t["net_pnl"] > 0 else 0
        s["losses"] += 1 if t["net_pnl"] <= 0 else 0

    result = []
    for sym, d in symbol_map.items():
        d["gross_pnl"] = round(d["gross_pnl"], 2)
        d["total_charges"] = round(d["total_charges"], 2)
        d["net_pnl"] = round(d["net_pnl"], 2)
        result.append(d)
    return sorted(result, key=lambda x: x["net_pnl"], reverse=True)
