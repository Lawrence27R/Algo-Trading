import json
import os
import datetime
from config.settings import MAX_DAILY_LOSS, MAX_POSITION_SIZE
from loguru import logger

TRADE_LOG_PATH = "data/logs/trades.json"


class RiskManager:
    def __init__(self):
        self.max_daily_loss = MAX_DAILY_LOSS
        self.max_position_size = MAX_POSITION_SIZE

    def _get_todays_trades(self):
        if not os.path.exists(TRADE_LOG_PATH):
            return []
        with open(TRADE_LOG_PATH) as f:
            trades = json.load(f)
        today = datetime.date.today().isoformat()
        return [t for t in trades if t.get("timestamp", "").startswith(today)]

    def get_daily_pnl(self):
        from reports.report_engine import get_per_trade_report
        today = datetime.date.today().isoformat()
        all_trades = get_per_trade_report()
        daily_trades = [t for t in all_trades if (t.get("sell_time") or "").startswith(today)]
        return sum(t["net_pnl"] for t in daily_trades)

    def can_trade(self, symbol, quantity, price):
        """Check if a new trade is within risk limits."""
        position_value = quantity * price
        daily_pnl = self.get_daily_pnl()

        if position_value > self.max_position_size:
            logger.warning(f"RISK BLOCK: Position size {position_value} > max {self.max_position_size}")
            return False, f"Position size ₹{position_value} exceeds limit ₹{self.max_position_size}"

        if daily_pnl < -self.max_daily_loss:
            logger.warning(f"RISK BLOCK: Daily loss {daily_pnl} > max {self.max_daily_loss}")
            return False, f"Daily loss ₹{abs(daily_pnl):.0f} exceeds max loss limit ₹{self.max_daily_loss}"

        return True, "Trade allowed"

    def get_risk_status(self):
        daily_pnl = self.get_daily_pnl()
        return {
            "daily_pnl": round(daily_pnl, 2),
            "max_daily_loss": self.max_daily_loss,
            "daily_loss_used_pct": round(abs(min(daily_pnl, 0)) / self.max_daily_loss * 100, 1),
            "max_position_size": self.max_position_size,
            "can_trade": daily_pnl > -self.max_daily_loss,
            "status": "ACTIVE" if daily_pnl > -self.max_daily_loss else "HALTED"
        }
