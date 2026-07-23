"""
NBA Bet Predictor — built on REAL data.

Team strength (Elo, rolling offensive/defensive efficiency) comes from real
NBA.com game logs, 2018-19 through 2023-24 (the latest available in the free
dataset this project uses — see README for how to extend to the current
season). You supply the game-specific details that change: rest days, and
if you want an edge/Kelly calculation, the real odds YOU see on your
sportsbook right now — this tool does not fabricate odds.

Two modes:
  1. CLI interactive:      python predict.py
  2. Programmatic:         from predict import predict_matchup
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from src.nba_real_pipeline import NBARealDataPipeline
from src.models import EnsemblePredictor
from src.utils import kelly_criterion, expected_value

MODEL_PATH = Path(__file__).parent / "models" / "nba_real_ensemble.pkl"
FEATURE_COLS = ["elo_diff", "net_rtg_diff", "rest_diff", "rest_advantage_pts"]

_team_stats_cache = None
_model_cache = None


def american_to_decimal(american: float) -> float:
    if american > 0:
        return (american + 100) / 100
    else:
        return 100 / (-american) + 1


def get_latest_real_team_stats() -> pd.DataFrame:
    """
    Real Elo rating and rolling offensive/defensive efficiency for every
    team, as of their last real game in the dataset (end of 2023-24 season).
    This is the actual computed state from real box scores — not invented.
    """
    global _team_stats_cache
    if _team_stats_cache is not None:
        return _team_stats_cache

    print("[INFO] Loading real NBA data to compute latest team stats...")
    pipeline = NBARealDataPipeline()
    df = pipeline.process(start_season="2018-19", end_season="2023-24")

    home_side = df[["home_team", "date", "home_elo", "home_roll_pts_for", "home_roll_pts_against"]].rename(
        columns={"home_team": "team", "home_elo": "elo",
                 "home_roll_pts_for": "roll_pts_for", "home_roll_pts_against": "roll_pts_against"}
    )
    away_side = df[["away_team", "date", "away_elo", "away_roll_pts_for", "away_roll_pts_against"]].rename(
        columns={"away_team": "team", "away_elo": "elo",
                 "away_roll_pts_for": "roll_pts_for", "away_roll_pts_against": "roll_pts_against"}
    )
    combined = pd.concat([home_side, away_side]).sort_values("date")
    latest = combined.groupby("team").tail(1).set_index("team")

    _team_stats_cache = latest
    return latest


def load_real_model() -> EnsemblePredictor:
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}.\n"
            f"Run `python train_real_model.py` first — it trains on real NBA "
            f"data and saves the model here."
        )

    print("[INFO] Loading model trained on real NBA data...")
    _model_cache = EnsemblePredictor.load(str(MODEL_PATH))
    return _model_cache


def predict_matchup(
    home_team: str,
    away_team: str,
    home_rest_days: int,
    away_rest_days: int,
    home_moneyline_american: float = None,
    away_moneyline_american: float = None,
    bankroll: float = 10000,
    kelly_safety_factor: float = 0.25,
) -> dict:
    """
    Predict a real matchup using each team's real, latest computed Elo and
    efficiency ratings. Odds are optional — if you supply the real moneyline
    you see on your sportsbook, this also computes edge and a Kelly-sized
    recommendation. Without odds, it just returns the model's probability.
    """
    stats = get_latest_real_team_stats()
    model = load_real_model()

    home_team = home_team.upper()
    away_team = away_team.upper()

    for t in (home_team, away_team):
        if t not in stats.index:
            raise ValueError(
                f"'{t}' not found in real team data. Available teams: "
                f"{sorted(stats.index.tolist())}"
            )

    home = stats.loc[home_team]
    away = stats.loc[away_team]

    elo_diff = home["elo"] - away["elo"]
    net_rtg_diff = (
        (home["roll_pts_for"] - home["roll_pts_against"]) -
        (away["roll_pts_for"] - away["roll_pts_against"])
    )
    rest_diff = home_rest_days - away_rest_days
    home_b2b = home_rest_days <= 1
    away_b2b = away_rest_days <= 1
    rest_advantage_pts = (-5 if home_b2b else 0) - (-5 if away_b2b else 0)

    X = pd.DataFrame([{
        "elo_diff": elo_diff,
        "net_rtg_diff": net_rtg_diff,
        "rest_diff": rest_diff,
        "rest_advantage_pts": rest_advantage_pts,
    }])

    model_prob = float(model.predict_proba(X)[0])

    result = {
        "matchup": f"{away_team} @ {home_team}",
        "home_team_real_elo": round(float(home["elo"]), 1),
        "away_team_real_elo": round(float(away["elo"]), 1),
        "home_recent_net_rating": round(float(home["roll_pts_for"] - home["roll_pts_against"]), 1),
        "away_recent_net_rating": round(float(away["roll_pts_for"] - away["roll_pts_against"]), 1),
        "model_home_win_prob": round(model_prob, 4),
    }

    if home_moneyline_american is not None and away_moneyline_american is not None:
        home_ml_decimal = american_to_decimal(home_moneyline_american)
        away_ml_decimal = american_to_decimal(away_moneyline_american)

        home_implied_raw = 1.0 / home_ml_decimal
        away_implied_raw = 1.0 / away_ml_decimal
        total = home_implied_raw + away_implied_raw
        market_implied_prob = home_implied_raw / total  # vig removed

        edge = model_prob - market_implied_prob
        ev = expected_value(model_prob, home_ml_decimal)

        full_kelly = kelly_criterion(model_prob, home_ml_decimal, kelly_fraction=1.0)
        rec_kelly = full_kelly * kelly_safety_factor
        rec_bet = max(0, rec_kelly * bankroll)
        should_bet = bool(edge > 0.02 and rec_bet > 0)

        result.update({
            "market_implied_prob": round(market_implied_prob, 4),
            "edge": round(edge, 4),
            "expected_value_per_dollar": round(ev, 4),
            "full_kelly_fraction": round(full_kelly, 4),
            "recommended_kelly_fraction": round(rec_kelly, 4),
            "recommended_bet_size": round(rec_bet, 2),
            "should_bet": should_bet,
            "recommendation": (
                f"BET ${rec_bet:,.2f} on {home_team} (moneyline)"
                if should_bet else "PASS — edge too small or negative"
            )
        })
    else:
        result["recommendation"] = (
            "Model probability only — supply real moneyline odds to get an edge/Kelly recommendation."
        )

    return result


def run_cli():
    print("\n" + "="*70)
    print("NBA BET PREDICTOR — built on real NBA.com game data (2018-24)")
    print("="*70)

    stats = get_latest_real_team_stats()
    model = load_real_model()

    print(f"\nTeams with real current data: {', '.join(sorted(stats.index.tolist()))}")
    print("(Elo/efficiency as of each team's last real game in the dataset — 2023-24 season)\n")

    home_team = input("Home team abbreviation (e.g. LAL): ").strip().upper()
    away_team = input("Away team abbreviation (e.g. BOS): ").strip().upper()

    home_rest = int(input("Home team rest days (1 = back-to-back) [2]: ").strip() or "2")
    away_rest = int(input("Away team rest days [2]: ").strip() or "2")

    has_odds = input("\nDo you have real moneyline odds from a sportsbook? (y/n) [n]: ").strip().lower() == "y"

    home_ml, away_ml, bankroll = None, None, 10000
    if has_odds:
        home_ml = float(input("Home moneyline (American, e.g. -150): ").strip())
        away_ml = float(input("Away moneyline (American, e.g. +130): ").strip())
        bankroll = float(input("Your bankroll ($) [10000]: ").strip() or "10000")

    try:
        result = predict_matchup(
            home_team=home_team, away_team=away_team,
            home_rest_days=home_rest, away_rest_days=away_rest,
            home_moneyline_american=home_ml, away_moneyline_american=away_ml,
            bankroll=bankroll,
        )
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        return

    print("\n" + "="*70)
    print(f"RESULT: {result['matchup']}")
    print("="*70)
    print(f"  {home_team} real Elo: {result['home_team_real_elo']}  |  "
          f"recent net rating: {result['home_recent_net_rating']:+.1f}")
    print(f"  {away_team} real Elo: {result['away_team_real_elo']}  |  "
          f"recent net rating: {result['away_recent_net_rating']:+.1f}")
    print(f"\n  Model win probability ({home_team}): {result['model_home_win_prob']*100:.1f}%")

    if "edge" in result:
        print(f"  Market implied probability:      {result['market_implied_prob']*100:.1f}%")
        print(f"  Edge:                             {result['edge']*100:+.2f}%")
        print(f"  Expected value per $1 bet:        ${result['expected_value_per_dollar']:.3f}")
        print(f"  Full Kelly fraction:               {result['full_kelly_fraction']*100:.2f}%")
        print(f"  Recommended (1/4 Kelly):           {result['recommended_kelly_fraction']*100:.2f}%")

    print(f"\n  >>> {result['recommendation']} <<<\n")


if __name__ == "__main__":
    run_cli()
