
import math
import csv
import os
import statistics
import matplotlib
matplotlib.use("Agg")  # untuk Termux / server tanpa GUI
import matplotlib.pyplot as plt
from datetime import datetime

DATA_FILE = "fvg_data.csv"
RESULTS_FILE = "results.csv"

# --- Fungsi Statistik ---
def norm_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

def fvg_probability(size, atr_value, mu, sigma, base=0.70, scale=0.30, conf_bonus=0.15, confirmation=False):
    ratio = size / atr_value if atr_value > 0 else 0
    Z = (ratio - mu) / sigma if sigma > 0 else 0
    p = norm_cdf(Z)
    adj = (p - 0.5) * scale
    P_no_conf = base + adj
    P_with_conf = P_no_conf + (conf_bonus if confirmation else 0)
    return size, ratio, Z, P_no_conf, P_with_conf

def expected_value(prob, rr, risk=1):
    return prob * (rr * risk) - (1 - prob) * risk

def kelly_criterion(prob, rr):
    q = 1 - prob
    b = rr
    kelly = (prob * (b + 1) - 1) / b if b > 0 else 0
    return max(0, kelly)

# --- Data Management ---
def load_history():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, "r") as f:
        reader = csv.reader(f)
        return [(float(size), float(atr)) for size, atr in reader]

def save_to_history(size, atr_value):
    with open(DATA_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([size, atr_value])

def save_results(direction, size, atr_value, ratio, mu, sigma, Z, prob, EV, Kelly):
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Datetime","Direction","FVG Size","ATR","Ratio","Mean","StdDev","Z-Score","Probability","EV","Kelly"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), direction, size, atr_value, ratio, mu, sigma, Z, prob, EV, Kelly])

# --- Plotting ---
def plot_distribution(ratios, mu, sigma):
    plt.figure(figsize=(8,5))
    plt.hist(ratios, bins=15, density=True, alpha=0.6, color='blue', edgecolor='black')
    plt.axvline(mu, color='red', linestyle='--', label=f"Mean μ={mu:.2f}")
    plt.axvline(mu+sigma, color='green', linestyle='--', label=f"μ+σ={mu+sigma:.2f}")
    plt.axvline(mu-sigma, color='green', linestyle='--', label=f"μ-σ={mu-sigma:.2f}")
    plt.title("Distribusi Ratio FVG/ATR (History)")
    plt.xlabel("Ratio (FVG/ATR)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    filename = f"fvg_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename)
    print(f"\n📊 Grafik distribusi disimpan sebagai: {filename}")

# --- Main ---
if __name__ == "__main__":
    print("=== Hedge Fund FVG Quant Bot (Hybrid + Grafik + Export CSV) ===")

    h1 = float(input("Masukkan High candle 1: "))
    l1 = float(input("Masukkan Low candle 1: "))
    h3 = float(input("Masukkan High candle 3: "))
    l3 = float(input("Masukkan Low candle 3: "))

    atr_value = float(input("Masukkan ATR saat ini: "))
    rr = float(input("Masukkan Reward/Risk ratio (contoh 2 untuk RR 1:2): "))
    conf_input = input("Ada konfirmasi LQ + MSS? (y/n): ").lower()
    confirmation = conf_input == "y"

    # Deteksi bullish / bearish
    fvg_bull = max(0, l3 - h1)
    fvg_bear = max(0, l1 - h3)

    if fvg_bull > 0:
        direction = "Bullish"
        size = fvg_bull
    elif fvg_bear > 0:
        direction = "Bearish"
        size = fvg_bear
    else:
        direction = "Tidak ada FVG valid"
        size = 0

    if size > 0 and atr_value > 0:
        save_to_history(size, atr_value)

    history = load_history()
    if len(history) >= 2:
        ratios = [s/atr for s, atr in history if atr > 0]
        mu = statistics.mean(ratios)
        sigma = statistics.pstdev(ratios)
    else:
        ratios = []
        mu, sigma = 1.0, 0.3

    size, ratio, Z, P_no_conf, P_with_conf = fvg_probability(size, atr_value, mu, sigma, confirmation=confirmation)
    EV = expected_value(P_with_conf, rr)
    Kelly = kelly_criterion(P_with_conf, rr)

    # Output
    print("\n=== Hasil Perhitungan ===")
    print(f"Arah FVG: {direction}")
    print(f"FVG size: {size:.2f}")
    print(f"ATR saat ini: {atr_value:.2f}")
    print(f"Ratio size/ATR: {ratio:.2f}")
    print(f"Mean ratio history (μ): {mu:.2f}")
    print(f"Std dev ratio history (σ): {sigma:.2f}")
    print(f"Z-score: {Z:.2f}")
    print(f"Probabilitas (dengan konfirmasi): {P_with_conf*100:.1f}%")
    print(f"Expected Value: {EV:.2f}R")
    print(f"Kelly Fraction: {Kelly*100:.1f}% dari modal")

    # Simpan ke results.csv
    save_results(direction, size, atr_value, ratio, mu, sigma, Z, P_with_conf, EV, Kelly)

    if len(ratios) >= 5:
        plot_distribution(ratios, mu, sigma)
    else:
        print("\n⚠️ Data history masih sedikit, grafik belum bisa ditampilkan.")
