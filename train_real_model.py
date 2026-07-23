"""
Train the ensemble model on REAL NBA game data and validate honestly.

Key differences from a naive approach:
  - CHRONOLOGICAL train/test split (not random) — you can't train on future
    games and test on past ones, that's leakage. Real trading/betting models
    must be validated the way they'd actually be used: train on the past,
    predict the future.
  - Label is the REAL game outcome (home team won), not a fabricated spread
    cover, since this dataset has no real historical odds.
  - All reported metrics (AUC, accuracy, log-loss, calibration) are computed
    on real, held-out games the model never saw during training.

Run: python train_real_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss

from src.nba_real_pipeline import NBARealDataPipeline
from src.models import EnsemblePredictor

FEATURE_COLS = ["elo_diff", "net_rtg_diff", "rest_diff", "rest_advantage_pts"]
MODEL_PATH = Path(__file__).parent / "models" / "nba_real_ensemble.pkl"


def chronological_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Split by date, not randomly — train on earlier games, test on later ones."""
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def main():
    print("\n" + "="*70)
    print("TRAINING ON REAL NBA DATA (2018-19 through 2023-24)")
    print("="*70)

    # ---- 1. Load real data ----
    pipeline = NBARealDataPipeline()
    df = pipeline.process(start_season="2018-19", end_season="2023-24")

    print(f"\n[DATA] {len(df)} real NBA games")
    print(f"[DATA] Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"[DATA] Real home win rate: {df['home_won'].mean()*100:.1f}%")

    # ---- 2. Chronological split (critical — no shuffling sports data) ----
    df_train, df_test = chronological_split(df, test_frac=0.2)

    print(f"\n[SPLIT] Train: {len(df_train)} games ({df_train['date'].min().date()} to {df_train['date'].max().date()})")
    print(f"[SPLIT] Test:  {len(df_test)} games ({df_test['date'].min().date()} to {df_test['date'].max().date()})")
    print(f"[SPLIT] This is a TIME split — test games all happen after every training game.")

    X_train = df_train[FEATURE_COLS].fillna(0)
    y_train = df_train["home_won"]
    X_test = df_test[FEATURE_COLS].fillna(0)
    y_test = df_test["home_won"]

    # ---- 3. Train ----
    print(f"\n[TRAIN] Fitting ensemble (logistic + XGBoost-style GBM + neural net)...")
    model = EnsemblePredictor(weights={"logistic": 0.30, "gbm": 0.50, "nn": 0.20})
    model.fit(X_train, y_train, calibrate=True)

    # ---- 4. Honest evaluation on real, unseen, future games ----
    print(f"\n[EVAL] Scoring on {len(df_test)} real held-out games the model never saw...")

    test_probs = model.predict_proba(X_test)
    test_preds = model.predict(X_test)

    auc = roc_auc_score(y_test, test_probs)
    acc = accuracy_score(y_test, test_preds)
    ll = log_loss(y_test, test_probs)
    brier = brier_score_loss(y_test, test_probs)

    # Baseline: always predict "home wins" (since home teams win more often)
    baseline_acc = max(y_test.mean(), 1 - y_test.mean())

    print(f"\n{'='*70}")
    print("REAL RESULTS (held-out games, chronological split)")
    print(f"{'='*70}")
    print(f"  AUC-ROC:              {auc:.4f}")
    print(f"  Accuracy:             {acc:.4f}")
    print(f"  Log loss:             {ll:.4f}")
    print(f"  Brier score:          {brier:.4f}  (lower is better; 0.25 = coin flip)")
    print(f"  Naive baseline acc:   {baseline_acc:.4f}  (always predict home team wins)")
    print(f"  Model lift over naive baseline: {(acc - baseline_acc)*100:+.1f} percentage points")

    if acc <= baseline_acc:
        print(f"\n  [HONEST NOTE] The model does NOT beat the naive 'always bet home' baseline.")
        print(f"  This is common — beating a well-calibrated market/baseline consistently is hard.")
        print(f"  This is the real, unembellished result on real data.")

    # ---- 5. Calibration check ----
    print(f"\n[CALIBRATION] Comparing predicted probability buckets to real outcomes:")
    df_test_eval = df_test.copy()
    df_test_eval["pred_prob"] = test_probs
    df_test_eval["bucket"] = pd.cut(df_test_eval["pred_prob"], bins=[0, 0.4, 0.5, 0.6, 0.7, 1.0])

    calib = df_test_eval.groupby("bucket", observed=True).agg(
        n_games=("home_won", "count"),
        predicted_avg=("pred_prob", "mean"),
        actual_rate=("home_won", "mean")
    )
    print(calib.to_string())

    # ---- 6. Save model ----
    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save(str(MODEL_PATH))

    # ---- 7. Save real predictions for inspection ----
    out_path = Path(__file__).parent / "data" / "results" / "nba_real_test_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_test_eval[["date", "home_team", "away_team", "home_score", "away_score",
                  "home_won", "pred_prob"]].to_csv(out_path, index=False)
    print(f"\n[SAVED] Real test predictions: {out_path}")
    print(f"[SAVED] Model: {MODEL_PATH}")

    print(f"\n{'='*70}")
    print("DONE — every number above came from real games, real dates, real scores.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
