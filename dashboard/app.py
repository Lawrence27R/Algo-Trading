from flask import Flask, jsonify, render_template, request, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import json
import os
import datetime
import sys

# Allow imports from project root when running from dashboard directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


# ── Report API endpoints ──────────────────────────────────────────────────────

@app.route("/api/reports/per_trade")
def api_per_trade_report():
    from reports.report_engine import get_per_trade_report
    return jsonify(get_per_trade_report())


@app.route("/api/reports/daily")
def api_daily_report():
    from reports.report_engine import get_daily_report
    return jsonify(get_daily_report())


@app.route("/api/reports/weekly")
def api_weekly_report():
    from reports.report_engine import get_weekly_report
    return jsonify(get_weekly_report())


@app.route("/api/reports/monthly")
def api_monthly_report():
    from reports.report_engine import get_monthly_report
    return jsonify(get_monthly_report())


@app.route("/api/reports/summary")
def api_summary_stats():
    from reports.report_engine import get_summary_stats
    return jsonify(get_summary_stats())


@app.route("/api/reports/symbol")
def api_symbol_report():
    from reports.report_engine import get_symbol_report
    return jsonify(get_symbol_report())


# ── CSV export endpoints ──────────────────────────────────────────────────────

@app.route("/api/export/trades_csv")
def export_trades_csv():
    from reports.csv_exporter import export_per_trade_csv
    csv_data = export_per_trade_csv()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=trades.csv"}
    )


@app.route("/api/export/daily_csv")
def export_daily_csv():
    from reports.csv_exporter import export_daily_csv as _export
    csv_data = _export()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=daily_report.csv"}
    )


@app.route("/api/export/monthly_csv")
def export_monthly_csv():
    from reports.csv_exporter import export_monthly_csv as _export
    csv_data = _export()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=monthly_report.csv"}
    )


# ── Risk management endpoints ─────────────────────────────────────────────────

@app.route("/api/risk/status")
def api_risk_status():
    from risk.risk_manager import RiskManager
    rm = RiskManager()
    return jsonify(rm.get_risk_status())


@app.route("/api/risk/check")
def api_risk_check():
    from risk.risk_manager import RiskManager
    symbol = request.args.get("symbol", "")
    quantity = int(request.args.get("quantity", 0))
    price = float(request.args.get("price", 0))
    rm = RiskManager()
    allowed, message = rm.can_trade(symbol, quantity, price)
    return jsonify({"allowed": allowed, "message": message})


# ── WebSocket ─────────────────────────────────────────────────────────────────

@socketio.on("connect")
def handle_connect():
    emit("message", {"data": "Connected to Algo Trading Dashboard"})


if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000)
