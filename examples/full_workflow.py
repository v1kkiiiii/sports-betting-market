"""
Full workflow example: Data -> Model -> Backtest
Run from repo root: python examples/full_workflow.py
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 1)[0] + "/..")

from src.pipeline import DataPipeline
from src.models import EnsemblePredictor
from src.backtest import Backtest
import pandas as pd
import numpy as np


def main():
    print("\n" + "="*60)
    print("SPORTS BETTING PREDICTION MARKET - FULL WORKFLOW")
    print("="*60)
    
    # ========== 1. DATA PIPELINE ==========
    print("\n[STEP 1] Loading and processing data...")
    pipeline = DataPipeline(sport="nfl")
    
    # Generate synthetic data (in production, use fetch_espn_data)
    df = pipeline.process(season=2023, use_synthetic=True)
    print(f"  Loaded {len(df)} games")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Check data
    print(f"\n  Feature availability:")
    feature_cols = [
        "elo_diff", "rest_diff", "line_movement",
        "closing_line_value", "home_moneyline_movement",
        "home_implied_prob", "elo_ratio"
    ]
    for col in feature_cols:
        pct_available = (df[col].notna().sum() / len(df)) * 100
        print(f"    {col:30s}: {pct_available:5.1f}%")
    
    # ========== 2. TRAIN/TEST SPLIT ==========
    print("\n[STEP 2] Splitting data...")
    train_size = int(0.8 * len(df))
    df_train = df.iloc[:train_size].copy()
    df_test = df.iloc[train_size:].copy()
    
    print(f"  Train set: {len(df_train)} games ({df_train['date'].min()} - {df_train['date'].max()})")
    print(f"  Test set:  {len(df_test)} games ({df_test['date'].min()} - {df_test['date'].max()})")
    
    # ========== 3. MODEL TRAINING ==========
    print("\n[STEP 3] Training ensemble model...")
    
    X_train = df_train[feature_cols].fillna(0)
    y_train = df_train["home_covered"]
    
    ensemble = EnsemblePredictor(
        weights={"logistic": 0.25, "gbm": 0.50, "nn": 0.25}
    )
    ensemble.fit(X_train, y_train, calibrate=True)
    
    # ========== 4. VALIDATION ==========
    print("\n[STEP 4] Validating on test data...")
    
    X_test = df_test[feature_cols].fillna(0)
    y_test = df_test["home_covered"]
    
    test_probs = ensemble.predict_proba(X_test)
    test_preds = ensemble.predict(X_test)
    
    # Metrics
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
    
    auc = roc_auc_score(y_test, test_probs)
    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds, zero_division=0)
    rec = recall_score(y_test, test_preds, zero_division=0)
    
    print(f"  AUC-ROC:  {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:   {rec:.4f}")
    
    # ========== 5. BACKTEST ==========
    print("\n[STEP 5] Running backtest with Kelly sizing...")
    
    # Add predictions to test data
    df_test["model_prob"] = test_probs
    
    bt = Backtest(
        model=ensemble,
        initial_capital=10000,
        kelly_fraction=0.25,
        min_edge=0.02,  # Need 2% edge
        juice=0.045,    # 4.5% bookmaker margin
        max_daily_bets=20,
        max_bet_size_pct=0.05,
        name="Ensemble Kelly Strategy"
    )
    
    results = bt.run(df_test, verbose=True)
    
    # ========== 6. EXPORT RESULTS ==========
    print("\n[STEP 6] Exporting results...")
    
    import os
    os.makedirs("data/results", exist_ok=True)
    
    # Save bet details
    bets_df = results.to_dataframe()
    bets_df.to_csv("data/results/backtest_bets.csv", index=False)
    print(f"  Saved {len(bets_df)} bets to data/results/backtest_bets.csv")
    
    # Save portfolio history
    portfolio_df = pd.DataFrame([
        {
            "date": p.date,
            "capital": p.capital,
            "daily_pnl": p.daily_pnl,
            "daily_return": p.daily_return,
            "winning_bets": p.winning_bets,
            "total_bets": p.total_bets
        }
        for p in results.portfolio_history
    ])
    portfolio_df.to_csv("data/results/portfolio_history.csv", index=False)
    print(f"  Saved portfolio history to data/results/portfolio_history.csv")
    
    # Save model
    import os
    os.makedirs("models", exist_ok=True)
    ensemble.save("models/nfl_ensemble.pkl")
    print(f"  Saved model to models/nfl_ensemble.pkl")
    
    # ========== 7. ANALYSIS ==========
    print("\n[STEP 7] Key insights...")
    
    if len(bets_df) > 0:
        # Win rate by confidence tier
        print(f"\n  Win rate by confidence tier:")
        bets_df["confidence_tier"] = pd.cut(
            bets_df["predicted_prob"],
            bins=[0, 0.55, 0.60, 0.65, 1.0],
            labels=["Marginal", "Moderate", "Strong", "Very Strong"]
        )
        
        for tier in ["Marginal", "Moderate", "Strong", "Very Strong"]:
            tier_data = bets_df[bets_df["confidence_tier"] == tier]
            if len(tier_data) > 0:
                tier_wr = tier_data["outcome"].mean()
                tier_count = len(tier_data)
                print(f"    {tier:15s}: {tier_wr*100:5.1f}% ({tier_count} bets)")
        
        # Edge analysis
        print(f"\n  Edge analysis:")
        positive_edge = bets_df[bets_df["edge_vs_market"] > 0]
        print(f"    Bets with +edge: {len(positive_edge)}/{len(bets_df)}")
        if len(positive_edge) > 0:
            print(f"    Win rate on +edge bets: {positive_edge['outcome'].mean()*100:.1f}%")
            print(f"    Avg edge: {positive_edge['edge_vs_market'].mean()*100:.2f}%")
    
    print("\n" + "="*60)
    print("WORKFLOW COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
