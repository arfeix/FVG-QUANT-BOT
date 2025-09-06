📘 FVG QUANT BOT

A quantitative trading assistant for analyzing Fair Value Gaps (FVG), risk management with Kelly Criterion, and trade journaling.
Supports multi-pair tracking, risk-adjusted position sizing, performance tracking, and equity curve plotting.


---

🚀 Features

Multi-pair support (BTCUSDT, ETHUSDT, etc.) → each pair has its own history & journal.

Automatic calculation of:

FVG detection (Bullish / Bearish).

Probability estimation (Z-score, normal distribution).

Expected Value (EV).

Kelly Criterion for optimal fraction.

Position sizing based on account equity and chosen risk %.


Trade journaling:

Tracks results (Win/Loss, Profit/Loss).

Updates account equity automatically.

Exports journal_PAIR.csv for each pair.


Performance analytics:

Winrate, Profit Factor, Expectancy, Max Drawdown.

Equity curve plotted automatically (equity_PAIR_YYYYMMDD.png).




---

⚙️ Installation

1. Clone repository

git clone https://github.com/arfeix/FVG-QUANT-BOT.git
cd FVG-QUANT-BOT

2. Create virtual environment (recommended)

python3 -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

3. Install dependencies

pip install -r requirements.txt

If requirements.txt is not yet included, install manually:

pip install matplotlib


---

▶️ Usage

Run the bot:

python bot.py

Then follow the prompts:

Enter pair name (e.g., BTCUSDT).

Enter candle highs/lows, ATR, RR, confirmation, entry, stop loss, take profit.

Bot will calculate and export results.

After the trade finishes, input Win/Loss → journal will be updated.



---

📂 File Outputs

fvg_data_PAIR.csv → historical FVG/ATR ratios.

results_PAIR.csv → calculated trade setups.

journal_PAIR.csv → trade results + equity tracking.

equity_PAIR_YYYYMMDD.png → performance chart.


Example:

FVG-QUANT-BOT/
│── bot.py
│── fvg_data_BTCUSDT.csv
│── results_BTCUSDT.csv
│── journal_BTCUSDT.csv
│── equity_BTCUSDT_20250906.png


---

📊 Example Output

=== Results (BTCUSDT) ===
FVG Direction: Bearish
FVG size: 1665.90
ATR: 226.00
Size/ATR ratio: 7.37
Probability (with confirmation): 85.00%
Expected Value: +0.41R
Kelly Fraction: 82.0% (used: 1% risk)
Position Size: 0.0255 units
Entry: 111266.0 | SL: 111657.5 | TP: 109308.5

=== Performance Summary (BTCUSDT) ===
Total Trades: 25
Winrate: 56.0%
Profit Factor: 1.85
Expectancy: +0.21 R/trade
Max Drawdown: -3.5%
Final Equity: $1,257.00

📈 Equity curve saved: equity_BTCUSDT_20250906.png


---

🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss.


---

📜 License

MIT License.
