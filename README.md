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

```bash
git clone https://github.com/arfeix/FVG-QUANT-BOT.git
cd FVG-QUANT-BOT
```

2. Create virtual environment (recommended)

Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows (Command Prompt)

```bash
python -m venv venv
venv\Scripts\activate
```

Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. Upgrade pip (recommended)

```bash
pip install --upgrade pip
```

4. Install dependencies

```bash
pip install -r requirements.txt
```

Jika file requirements.txt belum ada, buat dengan isi:

```txt
matplotlib
```

Lalu jalankan lagi:

```bash
pip install -r requirements.txt
```

5. Run the bot

```bash
python bot.py
```



---

🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss.


---

📜 License

MIT License.


