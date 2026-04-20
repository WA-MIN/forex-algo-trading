# Forex-Algo-Trading
Research pipeline comparing rule-based strategies on 1-minute FX data. The selection criterion is mean fold net Sharpe after transaction costs.
## What this is
Seven currency pairs (plus EURGBP), 2015 to 2025, 1-minute bars sourced from HistData.com. The pipeline runs from raw download through feature engineering, label construction, walk-forward cross-validation, and cost-aware backtesting. Every strategy is evaluated by the same engine with the same metric definitions.

The question: under realistic spreads and strict time-series cross-validation, which strategy produces the best risk-adjusted net performance at the 1-minute level?
## Quick Start (Windows + PyCharm)
### 1. Clone the repository
```bash
git clone https://github.com/Kanyal-HarsH/forex-algo-trading.git
cd forex-algo-trading
```
### 2. Create virtual environment
```bash
python -m venv venv
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
If ```requirements.txt``` is missing:
```bash
pip install pandas numpy pyarrow scikit-learn matplotlib seaborn pyyaml
```
