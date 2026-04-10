from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import json
import os
import datetime

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

TRADE_LOG_PATH = "data/logs/trades.json"
PORTFOLIO_PATH = "data/logs/portfolio.json"

def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/trades")
def get_trades():
    trades = load_json(TRADE_LOG_PATH)
    return jsonify(trades)

@app.route("/api/portfolio")
def get_portfolio():
    portfolio = load_json(PORTFOLIO_PATH)
    return jsonify(portfolio)

@app.route("/api/stats")
def get_stats():
    trades = load_json(TRADE_LOG_PATH)
    portfolio = load_json(PORTFOLIO_PATH)
    total_trades = len(trades)
    buys = sum(1 for t in trades if t.get("transaction_type") == "BUY")
    sells = sum(1 for t in trades if t.get("transaction_type") == "SELL")
    pnl_history = portfolio.get("pnl_history", []) if isinstance(portfolio, dict) else []
    total_pnl = sum(p["pnl"] for p in pnl_history)
    cash = portfolio.get("cash", 100000) if isinstance(portfolio, dict) else 100000
    return jsonify({
        "total_trades": total_trades,
        "buys": buys,
        "sells": sells,
        "total_pnl": round(total_pnl, 2),
        "cash": round(cash, 2),
        "pnl_history": pnl_history
    })

@app.route("/api/verify_trade/<trade_id>")
def verify_trade(trade_id):
    trades = load_json(TRADE_LOG_PATH)
    trade = next((t for t in trades if t.get("id") == trade_id), None)
    if trade:
        return jsonify({"verified": True, "trade": trade})
    return jsonify({"verified": False, "message": "Trade not found"}), 404

@socketio.on("connect")
def handle_connect():
    emit("message", {"data": "Connected to Algo Trading Dashboard"})

if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000)
