"""
Kelly Criterion Optimization Example
Finds optimal Kelly fraction by sweeping and evaluating performance.
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 1)[0] + "/..")

from src.pipeline import DataPipeline
from src.models import EnsemblePredictor
from src.backtest import Backtest
from src.optimization import KellySweep, PortfolioKellyOptimizer, GrowthOptimizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    print("\n" + "="*70)
    print("KELLY CRITERION OPTIMIZATION")
    print("="*70)
    
    # ========== 1. LOAD DATA & TRAIN MODEL ==========
    print("\n[STEP 1] Loading data and training model...")
    pipeline = DataPipeline(sport="nfl")
    df = pipeline.process(season=2023, use_synthetic=True)
    
    train_size = int(0.8 * len(df))
    df_train = df.iloc[:train_size].copy()
    df_test = df.iloc[train_size:].copy()
    
    feature_cols = [
        "elo_diff", "rest_diff", "line_movement",
        "closing_line_value", "home_moneyline_movement",
        "home_implied_prob", "elo_ratio"
    ]
    
    X_train = df_train[feature_cols].fillna(0)
    y_train = df_train["home_covered"]
    
    ensemble = EnsemblePredictor(
        weights={"logistic": 0.25, "gbm": 0.50, "nn": 0.25}
    )
    ensemble.fit(X_train, y_train, calibrate=True)
    
    print(f"  Model trained on {len(df_train)} games")
    print(f"  Testing on {len(df_test)} games\n")
    
    # ========== 2. KELLY SWEEP ==========
    print("[STEP 2] Sweeping Kelly fractions...")
    print("  Testing kelly_fraction values from 0.05 to 1.0\n")
    
    # Define backtest function that can accept kelly_fraction
    def run_backtest_with_kelly(kelly_fraction, df_test_data):
        bt = Backtest(
            model=ensemble,
            initial_capital=10000,
            kelly_fraction=kelly_fraction,
            min_edge=0.02,
            juice=0.045,
            max_daily_bets=20,
            max_bet_size_pct=0.05,
            name=f"Kelly {kelly_fraction:.2f}"
        )
        return bt.run(df_test_data, verbose=False)
    
    sweep = KellySweep(kelly_fractions=np.linspace(0.05, 1.0, 20))
    
    # Run sweep (this wraps the backtest function)
    results_list = []
    
    for kf in sweep.kelly_fractions:
        bt = Backtest(
            model=ensemble,
            initial_capital=10000,
            kelly_fraction=kf,
            min_edge=0.02,
            juice=0.045,
            max_daily_bets=20,
            max_bet_size_pct=0.05
        )
        
        results = bt.run(df_test, verbose=False)
        
        bets = results.bets
        if not bets:
            results_list.append({
                "kelly_fraction": kf,
                "total_return": 0,
                "sharpe": 0,
                "win_rate": 0,
                "max_drawdown": 0,
                "num_bets": 0,
            })
            continue
        
        pnls = np.array([b.pnl for b in bets])
        outcomes = np.array([b.outcome for b in bets])
        
        total_return = pnls.sum() / 10000
        win_rate = outcomes.mean()
        
        daily_returns = [p.daily_return for p in results.portfolio_history]
        daily_rets = np.array(daily_returns)
        if len(daily_rets) > 1 and np.std(daily_rets) > 0:
            sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
        else:
            sharpe = 0
        
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / (np.abs(running_max) + 1)
        max_dd = drawdowns.min() if len(drawdowns) > 0 else 0
        
        results_list.append({
            "kelly_fraction": kf,
            "total_return": total_return,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "max_drawdown": max_dd,
            "num_bets": len(bets),
        })
        
        print(f"  Kelly {kf:.2f}: Return={total_return*100:6.1f}% Sharpe={sharpe:6.2f} WR={win_rate*100:5.1f}% Bets={len(bets)}")
    
    df_sweep = pd.DataFrame(results_list)
    
    # ========== 3. FIND OPTIMAL ==========
    print("\n[STEP 3] Analyzing results...")
    
    best_sharpe_idx = df_sweep["sharpe"].idxmax()
    best_return_idx = df_sweep["total_return"].idxmax()
    
    optimal_kelly_sharpe = df_sweep.loc[best_sharpe_idx, "kelly_fraction"]
    optimal_kelly_return = df_sweep.loc[best_return_idx, "kelly_fraction"]
    
    print(f"\n  SHARPE RATIO OPTIMIZATION:")
    print(f"    Optimal Kelly: {optimal_kelly_sharpe:.3f}")
    print(f"    Sharpe Ratio: {df_sweep.loc[best_sharpe_idx, 'sharpe']:.3f}")
    print(f"    Return: {df_sweep.loc[best_sharpe_idx, 'total_return']*100:.1f}%")
    print(f"    Win Rate: {df_sweep.loc[best_sharpe_idx, 'win_rate']*100:.1f}%")
    
    print(f"\n  RETURN OPTIMIZATION:")
    print(f"    Optimal Kelly: {optimal_kelly_return:.3f}")
    print(f"    Return: {df_sweep.loc[best_return_idx, 'total_return']*100:.1f}%")
    print(f"    Sharpe Ratio: {df_sweep.loc[best_return_idx, 'sharpe']:.3f}")
    print(f"    Win Rate: {df_sweep.loc[best_return_idx, 'win_rate']*100:.1f}%")
    
    # ========== 4. PORTFOLIO-LEVEL OPTIMIZATION ==========
    print("\n[STEP 4] Portfolio-level Kelly optimization...")
    
    # Extract odds and predicted probabilities for all test games
    X_test = df_test[feature_cols].fillna(0)
    test_probs = ensemble.predict_proba(X_test)
    test_odds = df_test["moneyline_home_close"].values
    
    pf_optimizer = PortfolioKellyOptimizer()
    
    # Optimize using portfolio constraints
    allocations, info = pf_optimizer.optimize_with_constraints(
        win_probs=test_probs,
        odds=test_odds,
        kelly_fraction=0.25,
        max_allocation_pct=0.05,
        max_total_exposure_pct=0.20,
        method="kelly"
    )
    
    print(f"  Portfolio has {len(test_probs)} potential bets")
    print(f"  Using kelly_fraction=0.25 (fractional Kelly for safety)")
    print(f"\n  Allocation summary:")
    print(f"    Bets with non-zero allocation: {(allocations > 0).sum()}")
    print(f"    Total portfolio exposure: {info['total_exposure']*100:.1f}%")
    print(f"    Average allocation per bet: {allocations[allocations > 0].mean()*100:.2f}%")
    print(f"    Max allocation per bet: {allocations.max()*100:.2f}%")
    print(f"    Expected portfolio value: {info['expected_value']:.4f}")
    
    # ========== 5. GROWTH RATE ANALYSIS ==========
    print("\n[STEP 5] Long-term growth rate analysis...")
    
    # Use empirical win rate and average odds
    empirical_wr = df_test["home_covered"].mean()
    avg_odds = df_test["moneyline_home_close"].mean()
    
    growth_opt = GrowthOptimizer(win_prob=empirical_wr, odds=avg_odds)
    optimal_kelly_growth, max_growth = growth_opt.find_optimal_kelly()
    
    print(f"  Empirical win rate: {empirical_wr*100:.1f}%")
    print(f"  Average odds received: {avg_odds:.3f}")
    print(f"\n  Analytical Kelly optimization:")
    print(f"    Optimal Kelly (full): {optimal_kelly_growth:.3f}")
    print(f"    Max growth rate: {max_growth:.4f}")
    print(f"    1/4 Kelly (recommended): {optimal_kelly_growth*0.25:.3f}")
    print(f"    Growth at 1/4 Kelly: {growth_opt.growth_rate(optimal_kelly_growth*0.25):.4f}")
    
    # ========== 6. SUMMARY & RECOMMENDATIONS ==========
    print("\n[STEP 6] Recommendations...")
    
    recommended_kelly = optimal_kelly_sharpe * 0.25  # Use 1/4 of optimal for safety
    
    print(f"\n  RECOMMENDED STRATEGY:")
    print(f"    Use Kelly Fraction: {recommended_kelly:.3f}")
    print(f"    Rationale: {optimal_kelly_sharpe:.3f} (optimal) * 0.25 (safety factor)")
    print(f"    This provides:")
    print(f"      - Lower drawdown risk")
    print(f"      - Smoother equity curve")
    print(f"      - Protection against model overfitting")
    print(f"      - Still captures 25% of theoretical edge")
    
    # ========== 7. EXPORT RESULTS ==========
    print("\n[STEP 7] Exporting results...")
    
    import os
    os.makedirs("data/results", exist_ok=True)
    
    df_sweep.to_csv("data/results/kelly_sweep.csv", index=False)
    print(f"  Saved Kelly sweep results to data/results/kelly_sweep.csv")
    
    # Export allocations
    allocations_df = pd.DataFrame({
        "game_idx": np.arange(len(allocations)),
        "predicted_prob": test_probs,
        "odds": test_odds,
        "kelly_allocation_pct": allocations * 100,
        "bet_amount": allocations * 10000
    })
    allocations_df.to_csv("data/results/optimal_allocations.csv", index=False)
    print(f"  Saved optimal allocations to data/results/optimal_allocations.csv")
    
    print("\n" + "="*70)
    print("KELLY OPTIMIZATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
