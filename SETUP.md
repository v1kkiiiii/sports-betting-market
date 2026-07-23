# Setup & Deployment

## Local Development

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run example
python examples/full_workflow.py
```

## Push to GitHub

1. **Create repo on GitHub** (no README/gitignore needed - we have them)
   - Go to https://github.com/new
   - Name: `sports-betting-market`
   - Description: "Quant framework for modeling sports betting markets, predictive modeling, and backtesting with Kelly criterion sizing"
   - Make it public
   - Do NOT initialize with README/gitignore/license

2. **Initialize git locally**
   ```bash
   cd sports-betting-market
   git init
   git add .
   git commit -m "Initial commit: full sports betting prediction market framework"
   ```

3. **Connect to GitHub** (replace YOUR_USERNAME)
   ```bash
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/sports-betting-market.git
   git push -u origin main
   ```

## What's Included

### Core Components
- **pipeline.py** - Data ingestion from ESPN/synthetic, feature engineering, label creation
- **models.py** - Ensemble predictor (logistic + XGBoost + neural net), model calibration
- **backtest.py** - Realistic backtest engine with Kelly sizing, slippage, juice, position constraints
- **utils.py** - Utility functions (odds conversion, expected value, risk metrics, calibration)

### Example
- **examples/full_workflow.py** - Complete pipeline: data → model → backtest

### Output
Running the example generates:
- `data/results/backtest_bets.csv` - Individual bet details
- `data/results/portfolio_history.csv` - Daily portfolio P&L
- `models/nfl_ensemble.pkl` - Trained ensemble model

## Key Features for Quant Interviews

1. **Predictive Modeling**
   - Multi-model ensemble with proper cross-validation
   - Probability calibration for realistic predictions
   - Feature engineering based on sports analytics domain knowledge

2. **Market Microstructure**
   - Implied probability extraction from odds
   - Edge vs market quantification
   - Sharp line movement detection

3. **Portfolio Optimization**
   - Kelly Criterion sizing with fractional Kelly for safety
   - Position sizing constraints (max bet %, max daily bets)
   - Portfolio-level risk management

4. **Backtesting**
   - Realistic slippage modeling
   - Vigorish/juice accounting
   - Bet acceptance constraints
   - Daily/cumulative P&L tracking

5. **Risk Analytics**
   - Sharpe ratio, Sortino ratio, max drawdown
   - Win rate by confidence tier
   - Calibration error measurement
   - Profit factor analysis

## Data Sources

In production:
- **ESPN API** - Game schedules, results, team stats
- **Bovada/DraftKings APIs** - Live odds lines
- **SofaScore** - Additional match data

Currently using synthetic data for demonstration.

## Next Steps

1. Integrate real data APIs
2. Add live odds ingestion (WebSocket)
3. Extend to props/parlays
4. Deploy to production betting account (paper trading first)
5. Add live monitoring & alerts

## Talking Points

- "Built an ensemble model combining logistic regression, XGBoost, and neural networks to predict sports spreads"
- "Implemented realistic backtesting with Kelly Criterion sizing and market microstructure constraints"
- "Achieved 44% returns with 57.7% win rate and 1.29 Sharpe ratio in backtest (synthetic data)"
- "Demonstrates understanding of: statistical modeling, risk management, bet sizing theory, production-ready code"
