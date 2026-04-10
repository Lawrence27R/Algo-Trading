import csv
import io
from reports.report_engine import (
    get_per_trade_report, get_daily_report,
    get_weekly_report, get_monthly_report,
    get_summary_stats, get_symbol_report
)


def export_per_trade_csv():
    data = get_per_trade_report()
    output = io.StringIO()
    if not data:
        return output.getvalue()
    fieldnames = ["symbol", "buy_time", "sell_time", "quantity",
                  "buy_price", "sell_price", "gross_pnl", "total_charges", "net_pnl", "mode"]
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(data)
    return output.getvalue()


def export_daily_csv():
    data = get_daily_report()
    output = io.StringIO()
    if not data:
        return output.getvalue()
    fieldnames = ["date", "trades", "gross_pnl", "total_charges", "net_pnl", "wins", "losses"]
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(data)
    return output.getvalue()


def export_monthly_csv():
    data = get_monthly_report()
    output = io.StringIO()
    if not data:
        return output.getvalue()
    fieldnames = ["month", "trades", "gross_pnl", "total_charges", "net_pnl", "wins", "losses"]
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(data)
    return output.getvalue()
