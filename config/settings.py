import os
from dotenv import load_dotenv

load_dotenv()

# Trading mode
TRADING_MODE = os.getenv("TRADING_MODE", "paper")
IS_PAPER_TRADING = TRADING_MODE.lower() != "live"

# Dhan API credentials
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID", "")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "")

# Risk management
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "5000"))
MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", "50000"))

# Alert thresholds
DAILY_PROFIT_TARGET = float(os.getenv("DAILY_PROFIT_TARGET", "10000"))
