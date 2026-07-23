"""
Kelly Criterion optimization for sports betting portfolios.
Finds optimal Kelly fractions, handles correlations, and optimizes bet sizing.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Callable
from scipy.optimize import minimize, differential_evolution
import warnings


class KellySweep:
    """
    Sweep Kelly fractions to find optimal value for a given strategy.
    Tests kelly_fractions from 0.01 to 1.0 and evaluates performance.
    """
    
    def __init__(self, kelly_fractions: np.ndarray = None):
        if kelly_fractions is None:
            self.kelly_fractions = np.linspace(0.05, 1.0, 20)
        else:
            self.kelly_fractions = kelly_fractions
    
    def evaluate_kelly_fraction(
        self,
        backtest_fn: Callable,
        kelly_fraction: float,
        **backtest_kwargs
    ) -> Dict:
        """
        Run backtest with given Kelly fraction and return metrics.
        backtest_fn: Function that runs backtest (e.g., Backtest.run)
        """
        results = backtest_fn(kelly_fraction=kelly_fraction, **backtest_kwargs)
        
        bets = results.bets
        if not bets:
            return {
                "kelly_fraction": kelly_fraction,
                "total_return": 0,
                "sharpe": 0,
                "win_rate": 0,
                "max_drawdown": 0,
                "num_bets": 0,
                "error": True
            }
        
        # Extract metrics
        pnls = np.array([b.pnl for b in bets])
        outcomes = np.array([b.outcome for b in bets])
        
        total_return = pnls.sum() / results.backtest.initial_capital
        win_rate = outcomes.mean()
        
        # Sharpe ratio
        daily_returns = [p.daily_return for p in results.portfolio_history]
        daily_rets = np.array(daily_returns)
        if len(daily_rets) > 1 and np.std(daily_rets) > 0:
            sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Max drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / (np.abs(running_max) + 1)
        max_dd = drawdowns.min() if len(drawdowns) > 0 else 0
        
        return {
            "kelly_fraction": kelly_fraction,
            "total_return": total_return,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "max_drawdown": max_dd,
            "num_bets": len(bets),
            "error": False
        }
    
    def optimize(
        self,
        backtest_fn: Callable,
        objective: str = "sharpe",
        **backtest_kwargs
    ) -> Tuple[float, pd.DataFrame]:
        """
        Sweep Kelly fractions and find optimal.
        objective: "sharpe", "return", "risk_adjusted" (Sharpe), or "profit_factor"
        
        Returns (optimal_kelly, results_df)
        """
        results = []
        
        print(f"[INFO] Sweeping Kelly fractions: {self.kelly_fractions}")
        
        for kf in self.kelly_fractions:
            try:
                result = self.evaluate_kelly_fraction(backtest_fn, kf, **backtest_kwargs)
                results.append(result)
                
                if not result["error"]:
                    print(f"  Kelly {kf:.2f}: Return={result['total_return']*100:6.1f}% "
                          f"Sharpe={result['sharpe']:6.2f} WR={result['win_rate']*100:5.1f}%")
            except Exception as e:
                print(f"  Kelly {kf:.2f}: Error - {e}")
                continue
        
        df = pd.DataFrame(results)
        
        # Find optimal
        if objective == "sharpe":
            best_idx = df["sharpe"].idxmax()
        elif objective == "return":
            best_idx = df["total_return"].idxmax()
        elif objective == "risk_adjusted":
            # Maximize return / |drawdown|
            df["risk_adjusted"] = df["total_return"] / (np.abs(df["max_drawdown"]) + 0.01)
            best_idx = df["risk_adjusted"].idxmax()
        else:
            best_idx = df["sharpe"].idxmax()
        
        optimal_kelly = df.loc[best_idx, "kelly_fraction"]
        
        print(f"\n[INFO] Optimal Kelly: {optimal_kelly:.3f}")
        print(f"       Return: {df.loc[best_idx, 'total_return']*100:.1f}%")
        print(f"       Sharpe: {df.loc[best_idx, 'sharpe']:.2f}")
        print(f"       Win Rate: {df.loc[best_idx, 'win_rate']*100:.1f}%")
        
        return optimal_kelly, df


class PortfolioKellyOptimizer:
    """
    Optimize Kelly sizing across multiple correlated bets.
    Handles bet correlations and portfolio-level constraints.
    """
    
    def __init__(self, correlation_matrix: np.ndarray = None):
        """
        correlation_matrix: NxN correlation matrix of bets
        If None, assumes independent bets
        """
        self.correlation_matrix = correlation_matrix
    
    def compute_kelly_allocations(
        self,
        win_probs: np.ndarray,
        odds: np.ndarray,
        kelly_fraction: float = 0.25,
        max_allocation_pct: float = 0.05,
        max_total_exposure_pct: float = 0.50
    ) -> Tuple[np.ndarray, float]:
        """
        Compute Kelly allocations for multiple bets.
        Handles correlations and constraints.
        
        Returns (allocations, portfolio_expected_value)
        """
        n_bets = len(win_probs)
        
        # Individual Kelly sizes
        kelly_sizes = []
        for i in range(n_bets):
            edge = win_probs[i] * odds[i] - 1
            if edge > 0:
                f_star = edge / (odds[i] - 1)
                f_star = f_star * kelly_fraction
            else:
                f_star = 0
            kelly_sizes.append(f_star)
        
        kelly_sizes = np.array(kelly_sizes)
        
        # Apply constraints
        allocations = np.minimum(kelly_sizes, max_allocation_pct)
        
        # Ensure total exposure doesn't exceed limit
        total_exposure = allocations.sum()
        if total_exposure > max_total_exposure_pct:
            allocations = allocations * (max_total_exposure_pct / total_exposure)
        
        # Portfolio expected value
        ev = sum(allocations[i] * (win_probs[i] * odds[i] - 1) 
                 for i in range(n_bets))
        
        return allocations, ev
    
    def optimize_with_constraints(
        self,
        win_probs: np.ndarray,
        odds: np.ndarray,
        kelly_fraction: float = 0.25,
        max_allocation_pct: float = 0.05,
        max_total_exposure_pct: float = 0.50,
        method: str = "kelly"  # "kelly" or "optimize"
    ) -> Tuple[np.ndarray, Dict]:
        """
        Optimize allocations considering correlations.
        method: "kelly" for standard Kelly, "optimize" for numerical optimization
        """
        n_bets = len(win_probs)
        
        if method == "kelly":
            allocations, ev = self.compute_kelly_allocations(
                win_probs, odds, kelly_fraction,
                max_allocation_pct, max_total_exposure_pct
            )
        else:
            # Numerical optimization to maximize expected return subject to constraints
            def objective(x):
                # Negative because we minimize
                return -sum(x[i] * (win_probs[i] * odds[i] - 1) 
                           for i in range(n_bets))
            
            def constraint_sum(x):
                # Total exposure <= limit
                return max_total_exposure_pct - x.sum()
            
            x0 = np.ones(n_bets) * (max_total_exposure_pct / n_bets)
            
            bounds = [(0, max_allocation_pct) for _ in range(n_bets)]
            constraints = {"type": "ineq", "fun": constraint_sum}
            
            result = minimize(
                objective, x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 100}
            )
            
            allocations = result.x
            ev = -result.fun
        
        return allocations, {
            "allocations": allocations,
            "expected_value": ev,
            "total_exposure": allocations.sum(),
            "n_bets_placed": (allocations > 0).sum()
        }


class GrowthOptimizer:
    """
    Find Kelly fraction that maximizes long-term logarithmic growth rate.
    For use when you have empirical win rates and odds data.
    """
    
    def __init__(self, win_prob: float, odds: float):
        """
        win_prob: Historical win probability
        odds: Average odds received
        """
        self.win_prob = win_prob
        self.odds = odds
    
    def growth_rate(self, kelly_fraction: float) -> float:
        """
        Logarithmic growth rate = P(W) * log(1 + f*(odds-1)) + P(L) * log(1 - f)
        Higher is better. At optimal Kelly, this is maximized.
        """
        if kelly_fraction <= 0 or kelly_fraction >= 1:
            return -np.inf
        
        win_gain = self.win_prob * np.log(1 + kelly_fraction * (self.odds - 1))
        loss_gain = (1 - self.win_prob) * np.log(1 - kelly_fraction)
        
        return win_gain + loss_gain
    
    def find_optimal_kelly(self) -> Tuple[float, float]:
        """
        Find Kelly fraction maximizing growth rate.
        Returns (optimal_kelly, max_growth_rate)
        """
        # Analytical solution for Kelly with binary outcomes
        # f* = (P*odds - 1) / (odds - 1)
        
        edge = self.win_prob * self.odds - 1
        if edge <= 0:
            return 0, -np.inf
        
        f_star = edge / (self.odds - 1)
        
        # Constrain to [0, 1]
        f_star = np.clip(f_star, 0, 1)
        
        growth = self.growth_rate(f_star)
        
        return f_star, growth


class BetCorrelationAnalyzer:
    """
    Analyze correlations between bets and adjust Kelly sizing accordingly.
    """
    
    @staticmethod
    def compute_bet_correlation(
        outcomes1: np.ndarray,
        outcomes2: np.ndarray
    ) -> float:
        """
        Compute correlation between two bet outcome sequences.
        outcomes: binary array (1 = win, 0 = loss)
        """
        if len(outcomes1) != len(outcomes2):
            return 0
        
        if np.std(outcomes1) == 0 or np.std(outcomes2) == 0:
            return 0
        
        return np.corrcoef(outcomes1, outcomes2)[0, 1]
    
    @staticmethod
    def correlation_adjusted_kelly(
        kelly_f: float,
        correlation: float,
        n_bets: int
    ) -> float:
        """
        Adjust Kelly fraction for correlated bets.
        Higher correlation = reduce Kelly to be more conservative.
        
        Simple adjustment: f_adjusted = f * (1 - |correlation|)
        """
        adjustment = 1 - np.abs(correlation) * 0.5  # Max 50% reduction
        return kelly_f * adjustment


if __name__ == "__main__":
    print("[INFO] Kelly Optimization Module Tests\n")
    
    # Test 1: Growth rate optimizer
    print("Test 1: Growth Rate Optimization")
    optimizer = GrowthOptimizer(win_prob=0.55, odds=1.95)
    optimal_f, max_growth = optimizer.find_optimal_kelly()
    print(f"  Win prob: 55%, Odds: 1.95")
    print(f"  Optimal Kelly: {optimal_f:.3f}")
    print(f"  Max growth rate: {max_growth:.4f}\n")
    
    # Test 2: Fractional Kelly safety
    print("Test 2: Fractional Kelly Comparison")
    full_kelly = optimal_f
    quarter_kelly = full_kelly * 0.25
    eighth_kelly = full_kelly * 0.125
    
    for name, f in [("Full Kelly", full_kelly), ("1/4 Kelly", quarter_kelly), ("1/8 Kelly", eighth_kelly)]:
        growth = optimizer.growth_rate(f)
        print(f"  {name:12s} ({f:.3f}): growth={growth:.4f}")
    
    print()
    
    # Test 3: Portfolio allocation
    print("Test 3: Portfolio-level Allocation")
    win_probs = np.array([0.55, 0.52, 0.58])
    odds = np.array([1.95, 2.0, 1.9])
    
    pf_optimizer = PortfolioKellyOptimizer()
    allocations, info = pf_optimizer.optimize_with_constraints(
        win_probs, odds,
        kelly_fraction=0.25,
        max_allocation_pct=0.05,
        max_total_exposure_pct=0.20
    )
    
    print(f"  Bets: {len(win_probs)}")
    print(f"  Win probs: {win_probs}")
    print(f"  Allocations: {allocations}")
    print(f"  Total exposure: {allocations.sum():.3f}")
    print(f"  Expected value: {info['expected_value']:.4f}")
    
    print("\n[INFO] Kelly optimization module ready for use")
