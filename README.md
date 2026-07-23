# nba-predict

A model that predicts NBA game outcomes using real historical data. Started this because I wanted a project that actually used real data end to end instead of another toy dataset, so everything here is trained and tested on real NBA.com game logs, not anything simulated.

## What it does

Feed it two teams and it'll give you a win probability for the home team, based on:
- Elo ratings (updated after every real game, chronologically (so a team's rating never "knows" about games that haven't happened yet)
- Rolling offensive/defensive efficiency over their last 15 games
- Rest days / back-to-back detection

If you also give it real moneyline odds (like what you'd see on a sportsbook), it'll calculate your edge and tell you what to bet using the Kelly Criterion, sized down to 1/4 Kelly so you don't blow up your bankroll on model overconfidence.

## The data

Real games, 2018-19 through 2023-24 seasons: about 7,000 games after filtering out early-season games that don't have enough history yet for the rolling stats. Data comes from [NocturneBear's NBA dataset](https://github.com/NocturneBear/NBA-Data-2010-2024) on GitHub, which pulls from official NBA.com box scores.

One thing I want to be upfront about: this doesn't include real betting odds. Historical odds data isn't free anywhere reliable — it's either paywalled or you need to scrape it yourself over time. So I trained the model on **actual game outcomes** (did the home team win) instead of pretending I had real market data to bet against. If you want to bet with this, you plug in your own real odds when you use `predict.py`.

## Results

Trained on games through March 2023, tested on the ~1,400 games after that (chronological split, not random, since randomly shuffling would let the model "see the future" during training, which is cheating for time-series data like this).

- **AUC-ROC: 0.70**
- **Accuracy: 64%**
- Beats a "just always pick the home team" baseline by about 10 percentage points

Not going to pretend this is some crazy edge — 64% accuracy on NBA games is a solid, believable number (most published models land somewhere around there), not the kind of inflated result you get from overfitting on fake data. Calibration is decent too, when the model says 70% win probability, real games in that bucket won about 76% of the time.

## Running it

```bash
pip install -r requirements.txt

# trains on real data (downloads it the first time, ~9mb) and shows honest metrics
python train_real_model.py

# score a real matchup you're looking at
python predict.py
```

`predict.py` uses each team's real Elo/efficiency as of their last game in the dataset (end of 2023-24). You just tell it the rest days for each team and, if you have them, the actual odds you're seeing right now.

## Stack

Python, pandas, scikit-learn (logistic regression + gradient boosting + a small neural net, ensembled together), scipy for the Kelly optimization.
