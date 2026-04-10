import json
import os

PORTFOLIO_PATH = "data/logs/portfolio.json"


class Portfolio:
    def __init__(self, initial_cash=100000):
        self.cash = initial_cash
        self.positions = {}
        self.pnl_history = []
        self._load()

    def _load(self):
        if os.path.exists(PORTFOLIO_PATH):
            with open(PORTFOLIO_PATH, "r") as f:
                data = json.load(f)
                self.cash = data.get("cash", self.cash)
                self.positions = data.get("positions", {})
                self.pnl_history = data.get("pnl_history", [])

    def save(self):
        os.makedirs(os.path.dirname(PORTFOLIO_PATH), exist_ok=True)
        with open(PORTFOLIO_PATH, "w") as f:
            json.dump({"cash": self.cash, "positions": self.positions, "pnl_history": self.pnl_history}, f, indent=2)

    def buy(self, symbol, quantity, price):
        cost = quantity * price
        if cost > self.cash:
            return False, "Insufficient funds"
        self.cash -= cost
        self.positions[symbol] = self.positions.get(symbol, {"qty": 0, "avg_price": 0})
        pos = self.positions[symbol]
        total_qty = pos["qty"] + quantity
        pos["avg_price"] = (pos["qty"] * pos["avg_price"] + cost) / total_qty
        pos["qty"] = total_qty
        self.save()
        return True, "BUY executed"

    def sell(self, symbol, quantity, price):
        if symbol not in self.positions or self.positions[symbol]["qty"] < quantity:
            return False, "Insufficient position"
        pos = self.positions[symbol]
        pnl = (price - pos["avg_price"]) * quantity
        self.cash += quantity * price
        pos["qty"] -= quantity
        self.pnl_history.append({"symbol": symbol, "pnl": pnl, "price": price})
        if pos["qty"] == 0:
            del self.positions[symbol]
        self.save()
        return True, f"SELL executed, PnL: {pnl:.2f}"

    def get_total_pnl(self):
        return sum(p["pnl"] for p in self.pnl_history)