"""
NBA Prediction Market Workflow
Complete pipeline: NBA data → model training → Kelly optimization → backtest
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 1)[0] + "/..")

from src.nba_pipeline import NBADataPipeline
from src.models import EnsemblePredictor
from src.backtest import Backtest
from src.optimization import GrowthOptimizer
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score


def main():
    print("\n" + "="*80)
    print("NBA SPORTS BETTING PREDICTION MARKET")
    print("="*80)
    
    # ========== 1. LOAD NBA DATA ==========
    print("\n[STEP 1] Fetching NBA data...")
    nba_pipeline = NBADataPipeline()
    
    # Process NBA data (tries real ESPN data, falls back to synthetic)
    df = nba_pipeline.process_nba_data(season=2024, use_synthetic=False)
    
    if df.empty:
        print("[INFO] Real API unavailable, generating synthetic 2024-25 NBA season data")
        df = nba_pipeline.process_nba_data(season=2024, use_synthetic=True)
    
    print(f"✓ Loaded {len(df)} NBA games")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    
    # Sample games
    print(f"\n  Sample games:")
    for _, game in df.head(3).iterrows():
        winner = game['home_team'] if game['spread_result'] > game['spread'] else game['away_team']
        print(f"    {game['date'].date()}: {game['home_team']} ({game['home_score']:.0f}) vs "
              f"{game['away_team']} ({game['away_score']:.0f}) - Winner: {winner}")
    
    # ========== 2. NBA-SPECIFIC FEATURES ==========
    print(f"\n[STEP 2] NBA-specific features analysis...")
    
    features = [
        ("elo_diff", "Team strength differential"),
        ("net_rtg_diff", "Net rating advantage"),
        ("rest_diff", "Rest days difference"),
        ("rest_advantage_pts", "B2B impact on points"),
        ("spread", "Opening spread"),
        ("home_implied_prob", "Market probability"),
    ]
    
    print(f"\n  Feature summary:")
    for feat, desc in features:
        if feat in df.columns:
            print(f"    {feat:20s} ({desc:30s}): μ={df[feat].mean():7.2f} σ={df[feat].std():6.2f}")
    
    # ========== 3. TRAIN/TEST SPLIT ==========
    print(f"\n[STEP 3] Train/Test split...")
    
    split_idx = int(0.75 * len(df))  # 75/25 for better test set
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    
    print(f"  Train: {len(df_train)} games | Test: {len(df_test)} games")
    print(f"  Train dates: {df_train['date'].min().date()} → {df_train['date'].max().date()}")
    print(f"  Test dates:  {df_test['date'].min().date()} → {df_test['date'].max().date()}")
    
    # ========== 4. MODEL TRAINING ==========
    print(f"\n[STEP 4] Training ensemble model...")
    
    feature_cols = ["elo_diff", "net_rtg_diff", "rest_diff", "rest_advantage_pts",
                    "spread", "home_implied_prob"]
    
    X_train = df_train[feature_cols].fillna(0)
    y_train = df_train["home_covered"]
    
    ensemble = EnsemblePredictor(
        weights={"logistic": 0.25, "gbm": 0.55, "nn": 0.20}
    )
    ensemble.fit(X_train, y_train, calibrate=True)
    
    # ========== 5. TEST SET VALIDATION ==========
    print(f"\n[STEP 5] Test set evaluation...")
    
    X_test = df_test[feature_cols].fillna(0)
    y_test = df_test["home_covered"]
    
    test_probs = ensemble.predict_proba(X_test)
    test_preds = ensemble.predict(X_test)
    
    auc = roc_auc_score(y_test, test_probs)
    acc = accuracy_score(y_test, test_preds)
    win_rate = test_preds.mean()
    
    print(f"  AUC-ROC:        {auc:.4f}")
    print(f"  Accuracy:       {acc:.4f}")
    print(f"  Predicted WR:   {win_rate*100:.1f}%")
    
    # ========== 6. KELLY CRITERION ANALYSIS ==========
    print(f"\n[STEP 6] Kelly Criterion optimization...")
    
    empirical_wr = y_test.mean()
    avg_home_ml = df_test["home_ml_close"].mean()
    
    # Use growth rate optimizer
    growth_opt = GrowthOptimizer(win_prob=empirical_wr, odds=avg_home_ml)
    optimal_kelly, max_growth = growth_opt.find_optimal_kelly()
    
    print(f"  Empirical test win rate: {empirical_wr*100:.1f}%")
    print(f"  Average odds (moneyline): {avg_home_ml:.3f}")
    print(f"  Optimal Kelly fraction: {optimal_kelly:.4f}")
    print(f"  Maximum growth rate: {max_growth:.4f}")
    print(f"\n  Kelly recommendations:")
    print(f"    Full Kelly:     {optimal_kelly:.4f} (risky, high volatility)")
    print(f"    Half Kelly:     {optimal_kelly*0.5:.4f} (moderate risk)")
    print(f"    Quarter Kelly:  {optimal_kelly*0.25:.4f} (conservative, recommended) ✓")
    print(f"    Eighth Kelly:   {optimal_kelly*0.125:.4f} (very safe)")
    
    recommended_kelly = optimal_kelly * 0.25
    
    # ========== 7. EXPECTED VALUE CALCULATION ==========
    print(f"\n[STEP 7] Expected value and position sizing...")
    
    # Calculate expected value for recommended Kelly
    test_probs = ensemble.predict_proba(X_test)
    test_edges = []
    
    for i, (prob, implied_prob) in enumerate(zip(test_probs, df_test["home_implied_prob"].values)):
        edge = prob - implied_prob
        test_edges.append(edge)
    
    positive_edges = sum(1 for e in test_edges if e > 0.02)
    avg_positive_edge = np.mean([e for e in test_edges if e > 0.02]) if positive_edges > 0 else 0
    
    print(f"  Bets with >2% edge: {positive_edges}/{len(test_edges)}")
    print(f"  Average edge (positive only): {avg_positive_edge*100:.2f}%")
    
    # Simulate expected returns
    expected_return = 0
    for i, row in df_test.iterrows():
        prob = test_probs[i - split_idx] if i >= split_idx else 0
        implied = row["home_implied_prob"]
        edge = prob - implied
        
        if edge > 0.02:  # Only bet if edge > 2%
            odds = row["home_ml_close"]
            bet_size = recommended_kelly * 10000
            potential_return = bet_size * (odds - 1) if row["home_covered"] else -bet_size
            expected_return += potential_return * prob
    
    print(f"\n  Projected performance:")
    print(f"    Expected return per edge: ${expected_return/max(positive_edges, 1):,.2f}")
    print(f"    Projected ROI: {expected_return/10000*100:,.1f}% (if using full capital)")
    
    # Save predictions
    import os
    os.makedirs("data/results", exist_ok=True)
    
    predictions_df = pd.DataFrame({
        "date": df_test["date"].values,
        "home_team": df_test["home_team"].values,
        "away_team": df_test["away_team"].values,
        "spread": df_test["spread"].values,
        "model_prob": test_probs,
        "implied_prob": df_test["home_implied_prob"].values,
        "edge": test_edges,
        "actual_result": df_test["home_covered"].values,
        "closing_odds": df_test["home_ml_close"].values,
    })
    
    predictions_df.to_csv("data/results/nba_predictions.csv", index=False)
    print(f"\n  Saved predictions to: data/results/nba_predictions.csv")
    
    # Summary stats
    sharpe = 0  # Placeholder
    profit_factor = 0
    
    # ========== 9. SUMMARY ==========
    print(f"\n" + "="*80)
    print("NBA PREDICTION MARKET SUMMARY")
    print("="*80)
    
    print(f"""
✓ NBA Data Pipeline
  - Fetches real ESPN NBA data (fallback: synthetic 2024-25 season)
  - NBA-specific features: efficiency ratings, back-to-back penalties, rest
  - {len(df)} games processed

✓ Ensemble Model
  - Logistic regression (25%) + XGBoost (55%) + Neural net (20%)
  - Calibrated for probability accuracy
  - AUC-ROC: {auc:.4f}

✓ Kelly Criterion Optimization
  - Optimal Kelly: {optimal_kelly:.4f}
  - Recommended (1/4 Kelly): {recommended_kelly:.4f}
  - Protection against overbetting

✓ Kelly Criterion Applied
  - Optimal Kelly: {optimal_kelly:.4f}
  - Recommended (1/4 Kelly): {recommended_kelly:.4f}
  - Bets with +edge: {positive_edges}/{len(test_edges)}

NEXT STEPS:
  1. Integrate real-time ESPN odds feed
  2. Add player injury/status tracking
  3. Implement live in-game betting
  4. Deploy to paper trading (DraftKings/FanDuel)
  5. Scale to production account
""")


if __name__ == "__main__":
    main()
