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

### 1. Clone repository
```bash
git clone https://github.com/arfeix/FVG-QUANT-BOT.git
cd FVG-QUANT-BOT
```

### 2. Create virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
