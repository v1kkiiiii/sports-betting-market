# NBA Prediction Model — Built on Real Data

A game-outcome prediction model for the NBA, trained and validated entirely
on real historical game data. No simulated games, no fabricated results.

## Data — what's real and what isn't

**Real:** Every game, date, team, and score in this project comes from
[NocturneBear/NBA-Data-2010-2024](https://github.com/NocturneBear/NBA-Data-2010-2024)
(MIT licensed), which is sourced from official NBA.com game logs. This
project uses the 2018-19 through 2023-24 regular seasons — 7,059 real games
after dropping early-season games that don't yet have enough history for
rolling stats.

All features are computed directly from those real games, with no lookahead:
- **Elo ratings** — updated game-by-game in chronological order. A team's
  Elo going into a game only reflects games it already played.
- **Rolling offensive/defensive efficiency** — trailing 15-game average
  points scored/allowed, shifted so the current game is never included in
  its own feature.
- **Rest days & back-to-backs** — computed from actual game dates per team.

**Not real (and not included):** Historical betting odds. NBA.com doesn't
publish them, and free/reliable historical odds data requires a paid dataset
or a licensed API (see "Adding real odds" below). Rather than fabricate
odds, this project trains and validates purely on **real game outcomes**
(did the home team actually win). The Kelly/edge calculation in `predict.py`
only activates if *you* supply real moneyline odds — e.g. what you're
currently seeing on a sportsbook for an upcoming game.

## Real results (not projected, not simulated)

Trained on games from 2018-10-16 to 2023-03-17, tested on the next 1,412
games chronologically after that (2023-03-17 to 2024-04-14) — a genuine
train-on-past, test-on-future split, not a random shuffle:

| Metric | Value |
|---|---|
| AUC-ROC | 0.704 |
| Accuracy | 64.1% |
| Log loss | 0.626 |
| Brier score | 0.219 |
| Naive baseline (always pick home team) | 54.4% |
| **Lift over baseline** | **+9.7 points** |

Calibration on held-out games (predicted probability vs. actual win rate):

| Predicted range | Games | Predicted avg | Actual rate |
|---|---|---|---|
| 0.0–0.4 | 217 | 32.2% | 27.2% |
| 0.4–0.5 | 232 | 45.3% | 41.8% |
| 0.5–0.6 | 291 | 55.3% | 50.5% |
| 0.6–0.7 | 255 | 64.6% | 58.4% |
| 0.7–1.0 | 417 | 76.1% | 75.8% |

The model is reasonably well-calibrated, slightly overconfident in the
middle ranges. This is a normal, honest result for this problem — published
academic NBA prediction models typically land in the 65-70% AUC range.
There is no claim of a profitable betting edge here, because that requires
real market odds, which this dataset doesn't include.

## Project structure

```
src/
  nba_real_pipeline.py   # Downloads real NBA.com data, computes real features
  models.py               # Ensemble (logistic + gradient boosting + neural net)
  optimization.py         # Kelly Criterion math (sport-agnostic)
  utils.py                 # Odds conversion, EV, risk metrics

train_real_model.py       # Trains + validates on real data, chronological split
predict.py                 # Score a real upcoming matchup; optional Kelly sizing
```

## Quick start

```bash
pip install -r requirements.txt

# Train and validate on real data (downloads ~9MB of real game data on first run)
python train_real_model.py

# Predict a real upcoming matchup
python predict.py
```

`predict.py` uses each team's real Elo and efficiency rating as of their
last game in the dataset (end of the 2023-24 season). You supply the parts
that are specific to the game you're actually looking at: rest days for
each team, and — optionally — the real moneyline odds you see on your
sportsbook right now. If you provide odds, it computes your edge and a
Kelly-sized bet recommendation.

## Adding real odds (for anyone extending this)

To do this properly (real backtested P&L, not just outcome prediction),
you need a real historical odds dataset. Options that exist:
- **[MGM Grand NBA betting data](https://www.kaggle.com/datasets/caseydurfee/mgm-grand-nba-betting-data)**
  (Kaggle) — real closing moneylines/spreads, 2021-22 through 2025-26.
- **[NBA Odds Data](https://www.kaggle.com/datasets/christophertreasure/nba-odds-data)**
  (Kaggle) — moneylines, spreads, totals, 2008-2023.
- **[The Odds API](https://the-odds-api.com)** — real-time odds across
  sportsbooks, useful for scoring upcoming games (not historical backtesting).

Join any of those to this project's real game data on date + team names,
and you can rebuild a genuine backtest with real market prices instead of
just outcome prediction.

## What this project demonstrates

- Real data pipeline: fetching, cleaning, joining, and feature engineering
  from actual game logs — not a synthetic generator.
- Point-in-time correctness: Elo and rolling stats are computed so no
  feature ever sees information from the future relative to the game it
  describes.
- Proper time-series validation: chronological train/test split, the way
  this model would actually need to be used in practice.
- Honest evaluation: AUC, log loss, Brier score, and calibration — not just
  a single flattering accuracy number.
- Kelly Criterion implementation, applied only when real market odds are
  available, rather than fabricated ones.

## License

MIT for this project's code. The underlying game data is MIT-licensed via
NocturneBear/NBA-Data-2010-2024, sourced from NBA.com.
