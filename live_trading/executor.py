from dhan_client.order_manager import OrderManager
from config.settings import IS_PAPER_TRADING
from loguru import logger

class LiveExecutor:
    def __init__(self):
        if IS_PAPER_TRADING:
            raise RuntimeError("Switch TRADING_MODE=live in .env to use LiveExecutor")
        self.order_manager = OrderManager()

    def execute_signal(self, symbol, signal, quantity, price=0):
        if signal in ('BUY', 'SELL'):
            result = self.order_manager.place_order(symbol, quantity, "MARKET", signal, price)
            logger.info(f"[LIVE] {signal} {symbol}: {result}")
            return result
        return "HOLD"