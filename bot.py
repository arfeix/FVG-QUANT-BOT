#!/usr/bin/env python3
"""
FVG Quant Bot (Multi-pair) with:
 - FVG detection
 - probability estimate (Z-score) per-pair using historical FVG/ATR ratios
 - SL/TP (manual or ATR-based)
 - position sizing enforced by 1% max risk (can be overridden per-run)
 - journal per pair (trade results)
 - performance metrics (winrate, profit factor, expectancy, max drawdown)
 - equity curve plotting
"""
import math
import csv
import os
import statistics
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from datetime import datetime

DEFAULT_RISK_PER_TRADE = 0.01  # default max 1% per trade
MIN_HISTORY_FOR_STATS = 5

# ---------------------------
# Helpers & core computations
# ---------------------------
def norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def fvg_probability(size, atr_value, mu, sigma, base=0.70, scale=0.30, conf_bonus=0.15, confirmation=False):
    ratio = (size / atr_value) if atr_value > 0 else 0.0
    eff_sigma = sigma if sigma and sigma > 1e-9 else 0.3
    Z = (ratio - mu) / eff_sigma
    p = norm_cdf(Z)
    adj = (p - 0.5) * scale
    p_no_conf = clamp01(base + adj)
    p_with_conf = clamp01(p_no_conf + (conf_bonus if confirmation else 0.0))
    return size, ratio, Z, p_no_conf, p_with_conf

def expected_value(prob, rr, risk=DEFAULT_RISK_PER_TRADE):
    return prob * (rr * risk) - (1 - prob) * risk

def kelly_criterion(prob, rr):
    b = rr
    if b <= 0:
        return 0.0
    k = (prob * (b + 1) - 1) / b
    return max(0.0, k)

def calc_position_size(equity, used_fraction, entry_price, stop_loss):
    if entry_price is None or stop_loss is None:
        return None, None
    risk_per_unit = abs(entry_price - stop_loss)
    if risk_per_unit <= 0:
        return None, None
    risk_amount = equity * used_fraction
    units = risk_amount / risk_per_unit
    return units, risk_per_unit

# ---------------------------
# File helpers (per pair)
# ---------------------------
def data_filename(pair): return f"fvg_data_{pair}.csv"
def results_filename(pair): return f"results_{pair}.csv"
def journal_filename(pair): return f"journal_{pair}.csv"
def equity_plot_filename(pair): return f"equity_curve_{pair}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

# ---------------------------
# Data management
# ---------------------------
def load_history(data_file):
    if not os.path.exists(data_file):
        return []
    rows = []
    with open(data_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                size, atr = float(row[0]), float(row[1])
                rows.append((size, atr))
            except (ValueError, IndexError):
                continue
    return rows

def save_to_history(data_file, size, atr_value):
    os.makedirs(os.path.dirname(data_file) or ".", exist_ok=True)
    with open(data_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([size, atr_value])

def save_results(results_file, pair, direction, size, atr_value, ratio, mu, sigma, Z,
                 prob, EV, kelly, used_fraction, rr, confirmation,
                 entry_price, stop_loss, take_profit, equity, pos_size, risk_per_unit):
    file_exists = os.path.exists(results_file)
    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Pair","Datetime","Direction","FVG Size","ATR","Ratio","Mean","StdDev",
                "Z-Score","Probability","EV(account)","Kelly","UsedFraction",
                "RR","Confirmation","RiskPerTrade",
                "Entry","StopLoss","TakeProfit",
                "Equity","PositionSize","RiskPerUnit"
            ])
        writer.writerow([
            pair,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            direction, f"{size:.6f}", f"{atr_value:.6f}", f"{ratio:.6f}",
            f"{mu:.6f}", f"{sigma:.6f}", f"{Z:.6f}",
            f"{prob:.6f}", f"{EV:.6f}", f"{kelly:.6f}", f"{used_fraction:.6f}",
            rr, "yes" if confirmation else "no", DEFAULT_RISK_PER_TRADE,
            f"{entry_price:.6f}" if entry_price is not None else "-",
            f"{stop_loss:.6f}" if stop_loss is not None else "-",
            f"{take_profit:.6f}" if take_profit is not None else "-",
            f"{equity:.2f}", f"{pos_size:.6f}" if pos_size else "-", f"{risk_per_unit:.6f}" if risk_per_unit else "-"
        ])

# ---------------------------
# Journal & performance
# ---------------------------
def init_journal_if_needed(journal_file, starting_equity=None):
    if not os.path.exists(journal_file):
        with open(journal_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Datetime","Entry","StopLoss","TakeProfit","Direction","RR",
                "PositionSize","RiskPerUnit","EquityBefore","ProfitAmount","ProfitR","EquityAfter","Note"
            ])
        # if starting_equity provided, write initial row with equity snapshot? We won't write fake trade.

def append_journal(journal_file, entry_price, stop_loss, take_profit, direction, rr,
                   pos_size, risk_per_unit, equity_before, profit_amount, profit_r, note=""):
    os.makedirs(os.path.dirname(journal_file) or ".", exist_ok=True)
    file_exists = os.path.exists(journal_file)
    if not file_exists:
        init_journal_if_needed(journal_file)
    equity_after = equity_before + profit_amount
    with open(journal_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f"{entry_price:.6f}" if entry_price is not None else "-",
            f"{stop_loss:.6f}" if stop_loss is not None else "-",
            f"{take_profit:.6f}" if take_profit is not None else "-",
            direction, rr,
            f"{pos_size:.6f}" if pos_size else "-", f"{risk_per_unit:.6f}" if risk_per_unit else "-",
            f"{equity_before:.2f}", f"{profit_amount:.2f}", f"{profit_r:.6f}", f"{equity_after:.2f}", note
        ])
    return equity_after

def read_journal(journal_file):
    if not os.path.exists(journal_file):
        return []
    rows = []
    with open(journal_file, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append(r)
            except Exception:
                continue
    return rows

# Performance metrics helpers
def compute_performance(journal_rows):
    """
    journal_rows: list of dicts from csv.DictReader
    Each row fields: ProfitAmount, ProfitR, EquityAfter
    """
    if not journal_rows:
        return {}
    trades = []
    for r in journal_rows:
        try:
            profit = float(r.get("ProfitAmount", "0") or 0)
            profit_r = float(r.get("ProfitR", "0") or 0)
            equity_after = float(r.get("EquityAfter", "0") or 0)
            trades.append({"profit": profit, "profit_r": profit_r, "equity_after": equity_after})
        except ValueError:
            continue

    if not trades:
        return {}

    total_trades = len(trades)
    wins = sum(1 for t in trades if t["profit"] > 0)
    losses = total_trades - wins
    winrate = wins / total_trades if total_trades else 0.0
    gross_profit = sum(t["profit"] for t in trades if t["profit"] > 0)
    gross_loss = sum(t["profit"] for t in trades if t["profit"] < 0)
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss < 0 else float("inf")
    avg_r = statistics.mean([t["profit_r"] for t in trades])
    expectancy = avg_r  # average R per trade recorded
    # equity series
    equity_series = [t["equity_after"] for t in trades if t["equity_after"] is not None]
    max_equity = max(equity_series) if equity_series else None
    # compute max drawdown
    peak = equity_series[0] if equity_series else None
    max_dd = 0.0
    peak = -float("inf")
    trough = float("inf")
    running_peak = -float("inf")
    running_max_dd = 0.0
    running_peak = equity_series[0] if equity_series else 0.0
    running_max_dd = 0.0
    peak = equity_series[0] if equity_series else 0.0
    for e in equity_series:
        if e > running_peak:
            running_peak = e
        dd = (running_peak - e) / running_peak if running_peak and running_peak != 0 else 0.0
        if dd > running_max_dd:
            running_max_dd = dd
    max_drawdown = running_max_dd * 100.0  # percent
    final_equity = equity_series[-1] if equity_series else None

    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "winrate": winrate,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "expectancy_R": expectancy,
        "max_drawdown_pct": max_drawdown,
        "final_equity": final_equity,
        "equity_series": equity_series
    }

def plot_equity_curve(equity_series, pair):
    if not equity_series:
        print("No equity data to plot.")
        return None
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(equity_series)+1), equity_series, marker='o')
    plt.title(f"Equity Curve ({pair})")
    plt.xlabel("Trade #")
    plt.ylabel("Equity")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fname = equity_plot_filename(pair)
    plt.savefig(fname)
    plt.close()
    return fname

# ---------------------------
# Utilities - user IO
# ---------------------------
def safe_float(prompt: str, allow_empty=False):
    while True:
        val = input(prompt).strip()
        if allow_empty and val == "":
            return None
        try:
            return float(val)
        except ValueError:
            print("Invalid number. Try again (or leave empty if allowed).")

def yes_no(prompt: str):
    return input(prompt).strip().lower() in ("y", "yes")

# ---------------------------
# Interactive main menu
# ---------------------------
def show_menu():
    print("\n==== FVG Quant Bot Menu ====")
    print("1) New setup & analyze (generate entry/SL/TP/size)")
    print("2) Record trade result to journal (Win/Loss or custom P&L)")
    print("3) Show performance summary & plot equity curve")
    print("4) Exit")
    choice = input("Choose option [1-4]: ").strip()
    return choice

def option_analyze(pair, data_file, results_file):
    print(f"\n--- ANALYZE ({pair}) ---")
    equity = safe_float("Enter account equity: ")
    h1 = safe_float("Enter High of candle 1: ")
    l1 = safe_float("Enter Low of candle 1: ")
    h3 = safe_float("Enter High of candle 3: ")
    l3 = safe_float("Enter Low of candle 3: ")
    atr_value = safe_float("Enter current ATR: ")
    rr = safe_float("Enter Reward/Risk ratio (e.g., 2 means RR 1:2): ")
    confirmation = yes_no("Confirmation LQ + MSS present? (y/n): ")
    entry_price = safe_float("Enter planned entry price: ")
    # SL choice
    auto_sl = yes_no("Use automatic SL based on ATR? (y/n): ")
    if auto_sl:
        atr_mult = safe_float("ATR multiplier for SL (e.g., 1.5): ")
        stop_loss = entry_price - atr_value * atr_mult if entry_price is not None else None
    else:
        stop_loss = safe_float("Enter stop loss price: ")
    # FVG detection
    fvg_bull = max(0.0, l3 - h1)
    fvg_bear = max(0.0, l1 - h3)
    if fvg_bull > 0:
        direction = "Bullish"
        size = fvg_bull
    elif fvg_bear > 0:
        direction = "Bearish"
        size = fvg_bear
    else:
        direction = "No valid FVG"
        size = 0.0
    # TP calculation
    if stop_loss is not None and entry_price is not None:
        risk_per_unit = abs(entry_price - stop_loss)
        take_profit = entry_price + rr * risk_per_unit if direction == "Bullish" else entry_price - rr * risk_per_unit
    else:
        take_profit = None
        risk_per_unit = None
    # Save to per-pair history if valid
    if size > 0 and atr_value > 0:
        save_to_history(data_file, size, atr_value)
    # Load pair history stats (only if enough)
    history = load_history(data_file)
    if len(history) >= MIN_HISTORY_FOR_STATS:
        ratios = [s/atr for s, atr in history if atr > 0]
        mu = statistics.mean(ratios)
        sigma = statistics.pstdev(ratios) if len(ratios) > 1 else 0.3
        sigma = sigma if sigma > 1e-9 else 0.3
    else:
        ratios = []
        mu, sigma = 1.0, 0.3
    # Core calcs
    size, ratio, Z, P_no_conf, P_with_conf = fvg_probability(size, atr_value, mu, sigma, confirmation=confirmation)
    EV = expected_value(P_with_conf, rr, risk=DEFAULT_RISK_PER_TRADE)
    Kelly = kelly_criterion(P_with_conf, rr)
    used_fraction = min(Kelly, DEFAULT_RISK_PER_TRADE)
    pos_size, risk_unit = calc_position_size(equity, used_fraction, entry_price, stop_loss)
    # Output
    print("\n=== Analysis Results ===")
    print(f"Pair: {pair}")
    print(f"Direction: {direction}")
    print(f"FVG size: {size:.6f}")
    print(f"ATR: {atr_value:.6f}")
    print(f"Ratio (size/ATR): {ratio:.6f}")
    print(f"Using history points: {len(history)} (need >= {MIN_HISTORY_FOR_STATS} for robust stats)")
    print(f"Mean ratio (μ): {mu:.6f}")
    print(f"Std dev (σ): {sigma:.6f}")
    print(f"Z-score: {Z:.6f}")
    print(f"Probability (with confirmation): {P_with_conf*100:.2f}%")
    print(f"Expected Value (account-relative): {EV:.6f}")
    print(f"Kelly (info): {Kelly*100:.2f}%")
    print(f"Used fraction (capped to {DEFAULT_RISK_PER_TRADE*100:.2f}%): {used_fraction*100:.2f}%")
    if entry_price is not None:
        print(f"Entry: {entry_price:.6f}")
    if stop_loss is not None:
        print(f"Stop Loss: {stop_loss:.6f}")
    if take_profit is not None:
        print(f"Take Profit: {take_profit:.6f}")
    if pos_size:
        print(f"Position Size (units): {pos_size:.6f}")
        print(f"Risk per unit: {risk_unit:.6f}")
    # Save results to results file
    save_results(results_file, pair, direction, size, atr_value, ratio, mu, sigma, Z,
                 P_with_conf, EV, Kelly, used_fraction, rr, confirmation,
                 entry_price, stop_loss, take_profit, equity, pos_size, risk_unit)
    print("Saved analysis to results file.")

def option_record_trade(pair, journal_file):
    print(f"\n--- RECORD TRADE ({pair}) ---")
    init_journal_if_needed(journal_file)
    # Read last equity if exists
    rows = read_journal(journal_file)
    equity_before = None
    if rows:
        try:
            equity_before = float(rows[-1].get("EquityAfter", rows[-1].get("EquityBefore", "0")) or 0)
        except Exception:
            equity_before = None
    if equity_before is None:
        equity_before = safe_float("No previous equity found. Enter starting equity: ")
    entry_price = safe_float("Entry price (or blank): ", allow_empty=True)
    stop_loss = safe_float("Stop loss (or blank): ", allow_empty=True)
    take_profit = safe_float("Take profit (or blank): ", allow_empty=True)
    direction = input("Direction (Bullish/Bearish): ").strip() or "-"
    rr = safe_float("RR used (e.g., 2): ", allow_empty=True) or 0.0
    pos_size = safe_float("Position size (units) used (or blank): ", allow_empty=True)
    risk_per_unit = safe_float("Risk per unit (abs(entry-stop)) (or blank): ", allow_empty=True)
    # Profit input: let user pick R or currency
    print("Input realized result:")
    p_choice = input("Enter 'r' for R-units (e.g., 2 means +2R) or 'c' for currency amount: ").strip().lower()
    profit_amount = 0.0
    profit_r = 0.0
    if p_choice == 'r':
        profit_r = safe_float("Enter profit in R (positive for win, negative for loss): ")
        # Compute profit_amount if pos_size & risk_per_unit known
        if pos_size and risk_per_unit:
            profit_amount = profit_r * risk_per_unit * pos_size  # R * (risk per unit) * units
        else:
            profit_amount = safe_float("Position size or risk per unit missing; enter realized P/L in currency: ")
    else:
        profit_amount = safe_float("Enter realized P/L in account currency (positive for win, negative for loss): ")
        if pos_size and risk_per_unit and risk_per_unit != 0:
            profit_r = profit_amount / (risk_per_unit * pos_size)
        else:
            profit_r = 0.0
    note = input("Optional note (or leave blank): ").strip()
    equity_after = append_journal(journal_file, entry_price, stop_loss, take_profit, direction, rr,
                                 pos_size, risk_per_unit, equity_before, profit_amount, profit_r, note)
    print(f"Trade recorded. Equity before: {equity_before:.2f} -> after: {equity_after:.2f}")

def option_performance(pair, journal_file):
    print(f"\n--- PERFORMANCE ({pair}) ---")
    rows = read_journal(journal_file)
    if not rows:
        print("No journal data found for this pair.")
        return
    perf = compute_performance(rows)
    if not perf:
        print("Unable to compute performance (no valid trades).")
        return
    print(f"Total trades: {perf['total_trades']}")
    print(f"Wins: {perf['wins']}, Losses: {perf['losses']}, Winrate: {perf['winrate']*100:.2f}%")
    print(f"Gross profit: {perf['gross_profit']:.2f}, Gross loss: {perf['gross_loss']:.2f}")
    pf = perf['profit_factor']
    pf_str = f"{pf:.3f}" if pf != float("inf") else "inf"
    print(f"Profit Factor: {pf_str}")
    print(f"Expectancy (avg R per trade): {perf['expectancy_R']:.4f}")
    print(f"Max Drawdown: {perf['max_drawdown_pct']:.2f}%")
    print(f"Final Equity: {perf['final_equity']:.2f}")
    # Plot equity curve
    fname = plot_equity_curve(perf.get("equity_series", []), pair)
    if fname:
        print(f"Equity curve saved: {fname}")

def main():
    print("=== FVG Quant Bot (multi-pair) with Journal & Performance ===")
    pair = input("Enter pair name (e.g., BTCUSDT): ").strip().upper()
    if not pair:
        print("Pair name required.")
        return
    data_file = data_filename(pair)
    results_file = results_filename(pair)
    journal_file = journal_filename(pair)
    init_journal_if_needed(journal_file)
    while True:
        choice = show_menu()
        if choice == '1':
            option_analyze(pair, data_file, results_file)
        elif choice == '2':
            option_record_trade(pair, journal_file)
        elif choice == '3':
            option_performance(pair, journal_file)
        elif choice == '4':
            print("Exiting. Bye.")
            break
        else:
            print("Invalid choice. Please choose 1-4.")

if __name__ == "__main__":
    main()
