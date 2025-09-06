import math
import csv
import os
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

DATA_FILE = "fvg_data.csv"
RESULTS_FILE = "results.csv"
RISK_PER_TRADE = 0.01  # 1% max per trade

# --- Helpers ---
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

def expected_value(prob, rr, risk=RISK_PER_TRADE):
    return prob * (rr * risk) - (1 - prob) * risk

def kelly_criterion(prob, rr):
    b = rr
    if b <= 0:
        return 0.0
    k = (prob * (b + 1) - 1) / b
    return max(0.0, k)

def calc_position_size(equity, used_fraction, entry_price, stop_loss):
    if not entry_price or not stop_loss:
        return None, None
    risk_per_unit = abs(entry_price - stop_loss)
    if risk_per_unit <= 0:
        return None, None
    risk_amount = equity * used_fraction
    size = risk_amount / risk_per_unit
    return size, risk_per_unit

# --- Data Management ---
def load_history():
    if not os.path.exists(DATA_FILE):
        return []
    rows = []
    with open(DATA_FILE, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                size, atr = float(row[0]), float(row[1])
                rows.append((size, atr))
            except (ValueError, IndexError):
                continue
    return rows

def save_to_history(size, atr_value):
    with open(DATA_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([size, atr_value])

def save_results(direction, size, atr_value, ratio, mu, sigma, Z, prob, EV, kelly, used_fraction,
                 rr, confirmation, entry_price, stop_loss, take_profit, equity, pos_size, risk_per_unit):
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Datetime","Direction","FVG Size","ATR","Ratio","Mean","StdDev",
                "Z-Score","Probability","EV(account)","Kelly","UsedFraction",
                "RR","Confirmation","RiskPerTrade",
                "Entry","StopLoss","TakeProfit",
                "Equity","PositionSize","RiskPerUnit"
            ])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            direction, f"{size:.6f}", f"{atr_value:.6f}", f"{ratio:.6f}",
            f"{mu:.6f}", f"{sigma:.6f}", f"{Z:.6f}",
            f"{prob:.6f}", f"{EV:.6f}", f"{kelly:.6f}", f"{used_fraction:.6f}",
            rr, "yes" if confirmation else "no", RISK_PER_TRADE,
            f"{entry_price:.4f}" if entry_price else "-",
            f"{stop_loss:.4f}" if stop_loss else "-",
            f"{take_profit:.4f}" if take_profit else "-",
            equity, f"{pos_size:.4f}" if pos_size else "-", f"{risk_per_unit:.6f}" if risk_per_unit else "-"
        ])

# --- Plotting ---
def plot_distribution(ratios, mu, sigma):
    plt.figure(figsize=(8,5))
    plt.hist(ratios, bins=15, density=True, alpha=0.6, edgecolor='black')
    plt.axvline(mu, linestyle='--', label=f"Mean μ={mu:.2f}")
    plt.axvline(mu + sigma, linestyle='--', label=f"μ+σ={mu + sigma:.2f}")
    plt.axvline(mu - sigma, linestyle='--', label=f"μ-σ={mu - sigma:.2f}")
    plt.title("Distribution of FVG/ATR Ratio (History)")
    plt.xlabel("Ratio (FVG/ATR)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    filename = f"fvg_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename)
    print(f"\n📊 Distribution chart saved as: {filename}")

# --- Utilities ---
def safe_float(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid number. Please try again.")

def yes_no(prompt: str) -> bool:
    return input(prompt).strip().lower() in ("y", "yes")

# --- Main ---
if __name__ == "__main__":
    print("=== Hedge Fund FVG Quant Bot (SL/TP, Position Sizing, Chart & CSV Export) ===")

    # Inputs
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
        stop_loss = entry_price - atr_value * atr_mult if entry_price else None
    else:
        stop_loss = safe_float("Enter stop loss price: ")

    # Detect FVG
    fvg_bull = max(0.0, l3 - h1)
    fvg_bear = max(0.0, l1 - h3)
    if fvg_bull > 0:
        direction = "Bullish"
        size = fvg_bull
    elif fvg_bear > 0:
        direction = "Bearish"
        size = fvg_bear
    else:
        direction, size = "No valid FVG", 0.0

    # TP calculation
    if stop_loss and entry_price:
        risk_per_unit = abs(entry_price - stop_loss)
        take_profit = entry_price + rr * risk_per_unit if direction == "Bullish" else entry_price - rr * risk_per_unit
    else:
        take_profit, risk_per_unit = None, None

    # History
    if size > 0 and atr_value > 0:
        save_to_history(size, atr_value)

    history = load_history()
    if len(history) >= 2:
        ratios = [s/atr for s, atr in history if atr > 0]
        if ratios:
            mu = statistics.mean(ratios)
            sigma = statistics.pstdev(ratios) if len(ratios) > 1 else 0.3
            sigma = sigma if sigma > 1e-9 else 0.3
        else:
            ratios, mu, sigma = [], 1.0, 0.3
    else:
        ratios, mu, sigma = [], 1.0, 0.3

    # Core calculations
    size, ratio, Z, P_no_conf, P_with_conf = fvg_probability(size, atr_value, mu, sigma, confirmation=confirmation)
    EV = expected_value(P_with_conf, rr, risk=RISK_PER_TRADE)
    Kelly = kelly_criterion(P_with_conf, rr)
    used_fraction = min(Kelly, RISK_PER_TRADE)

    # Position size
    pos_size, risk_per_unit = calc_position_size(equity, used_fraction, entry_price, stop_loss)

    # Output
    print("\n=== Results ===")
    print(f"FVG Direction: {direction}")
    print(f"FVG size: {size:.4f}")
    print(f"Current ATR: {atr_value:.4f}")
    print(f"Size/ATR ratio: {ratio:.4f}")
    print(f"Mean ratio history (μ): {mu:.4f}")
    print(f"Std dev ratio history (σ): {sigma:.4f}")
    print(f"Z-score: {Z:.4f}")
    print(f"Probability (with confirmation): {P_with_conf*100:.2f}%")
    print(f"Expected Value (account-relative): {EV:.6f}")
    print(f"Kelly Fraction (info): {Kelly*100:.2f}% of equity")
    print(f"Used fraction (capped to 1% risk): {used_fraction*100:.2f}% of equity")
    if entry_price and stop_loss and take_profit:
        print(f"Entry Price: {entry_price:.4f}")
        print(f"Stop Loss: {stop_loss:.4f}")
        print(f"Take Profit: {take_profit:.4f}")
    if pos_size:
        print(f"Position Size: {pos_size:.4f} units")
        print(f"Risk per unit: {risk_per_unit:.6f}")

    # Save results
    save_results(direction, size, atr_value, ratio, mu, sigma, Z,
                 P_with_conf, EV, Kelly, used_fraction, rr, confirmation,
                 entry_price, stop_loss, take_profit, equity, pos_size, risk_per_unit)

    # Plot if enough history
    if len(ratios) >= 5:
        plot_distribution(ratios, mu, sigma)
    else:
        print("\n⚠️ Not enough history to plot the distribution (need ≥ 5 ratio points).")
