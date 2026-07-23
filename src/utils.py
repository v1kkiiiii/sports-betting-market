"""
Utility functions for sports betting market modeling.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List


def american_to_decimal(american_odds: float) -> float:
    """
    Convert American odds to decimal odds.
    Positive: decimal = (american + 100) / 100
    Negative: decimal = 100 / (-american) + 1
    """
    if american_odds > 0:
        return (american_odds + 100) / 100
    else:
        return 100 / (-american_odds) + 1


def decimal_to_american(decimal_odds: float) -> float:
    """Convert decimal odds back to American format."""
    if decimal_odds >= 2.0:
        return (decimal_odds - 1) * 100
    else:
        return -100 / (decimal_odds - 1)


def implied_probability(decimal_odds: float, account_for_vig: bool = True) -> float:
    """
    Extract implied probability from decimal odds.
    account_for_vig: Remove bookmaker margin (vig) to get true probability.
    """
    implied = 1.0 / decimal_odds
    
    if account_for_vig:
        # Typical vig is 4.5%
        # Simplified: scale up implied probability
        implied = implied / (1 - 0.045)
        implied = min(implied, 1.0)  # Cap at 100%
    
    return implied


def expected_value(win_prob: float, decimal_odds: float) -> float:
    """
    Expected value of a $1 bet.
    EV = P(win) * (odds - 1) - P(loss) * 1
    EV = P(win) * (odds - 1) - (1 - P(win))
    """
    return win_prob * (decimal_odds - 1) - (1 - win_prob)


def kelly_criterion(win_prob: float, decimal_odds: float, kelly_fraction: float = 1.0) -> float:
    """
    Kelly Criterion: f = (edge) / (odds - 1)
    where edge = P(win) * odds - 1
    
    Returns fraction of bankroll to wager.
    Typical use: kelly_fraction = 0.25 (quarter Kelly for safety)
    """
    edge = win_prob * decimal_odds - 1
    if edge <= 0:
        return 0
    
    f_star = edge / (decimal_odds - 1)
    return f_star * kelly_fraction


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio.
    returns: array of period returns
    periods_per_year: 252 for daily, 52 for weekly, 12 for monthly
    """
    if len(returns) < 2:
        return 0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    
    if np.std(excess_returns) == 0:
        return 0
    
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)


def sortino_ratio(returns: np.ndarray, target_return: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Sortino ratio: only penalizes downside volatility.
    """
    excess_returns = returns - target_return / periods_per_year
    downside = np.minimum(excess_returns, 0)
    downside_std = np.sqrt(np.mean(downside ** 2))
    
    if downside_std == 0:
        return 0
    
    return np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year)


def max_drawdown(cumulative_returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown.
    cumulative_returns: array of cumulative returns (e.g., [0, 0.05, 0.03, 0.08])
    """
    if len(cumulative_returns) == 0:
        return 0
    
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / np.abs(running_max + 1e-8)
    
    return drawdown.min()


def profit_factor(wins: np.ndarray, losses: np.ndarray) -> float:
    """
    Profit factor = Total Wins / Total Losses
    > 1.5 is considered good
    """
    total_losses = np.abs(losses).sum()
    total_wins = wins.sum()
    
    if total_losses == 0:
        return np.inf if total_wins > 0 else 0
    
    return total_wins / total_losses


def win_rate(outcomes: np.ndarray) -> float:
    """
    Win rate = (# wins) / (# total bets)
    outcomes: binary array where 1 = win, 0 = loss
    """
    if len(outcomes) == 0:
        return 0
    return outcomes.mean()


def calibration_error(predicted_probs: np.ndarray, actual_outcomes: np.ndarray, bins: int = 5) -> float:
    """
    Expected Calibration Error (ECE).
    Measures how well predicted probabilities match actual frequencies.
    Lower is better (perfect = 0).
    """
    bin_boundaries = np.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    total = len(predicted_probs)
    
    for lower, upper in zip(bin_lowers, bin_uppers):
        in_bin = (predicted_probs >= lower) & (predicted_probs < upper)
        
        if in_bin.sum() == 0:
            continue
        
        # Accuracy in bin
        accuracy_in_bin = actual_outcomes[in_bin].mean()
        # Average confidence in bin
        avg_confidence = predicted_probs[in_bin].mean()
        # Proportion of predictions in bin
        proportion_in_bin = in_bin.sum() / total
        
        ece += proportion_in_bin * np.abs(accuracy_in_bin - avg_confidence)
    
    return ece


def compute_roc_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Compute AUC-ROC score (requires sklearn).
    """
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, y_pred_proba)
    except ImportError:
        print("[WARN] sklearn not available, returning 0")
        return 0


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_func=np.mean,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for any statistic.
    Returns (lower_bound, upper_bound)
    """
    bootstrap_stats = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
    
    return lower, upper


if __name__ == "__main__":
    # Quick tests
    print("[TEST] American/Decimal conversion:")
    print(f"  -110 (American) -> {american_to_decimal(-110):.2f} (Decimal)")
    print(f"  1.909 (Decimal) -> {decimal_to_american(1.909):.0f} (American)")
    
    print("\n[TEST] Implied Probability:")
    print(f"  1.91 decimal odds -> {implied_probability(1.91)*100:.1f}% implied prob")
    
    print("\n[TEST] Expected Value:")
    print(f"  60% win prob, 1.95 odds -> EV = {expected_value(0.6, 1.95):.3f}")
    
    print("\n[TEST] Kelly Criterion:")
    print(f"  60% win prob, 1.95 odds, full Kelly -> {kelly_criterion(0.6, 1.95)*100:.1f}%")
    print(f"  60% win prob, 1.95 odds, quarter Kelly -> {kelly_criterion(0.6, 1.95, 0.25)*100:.1f}%")
    
    print("\n[TEST] Risk Metrics:")
    returns = np.array([0.01, -0.005, 0.015, -0.002, 0.02])
    print(f"  Returns: {returns}")
    print(f"  Sharpe: {sharpe_ratio(returns):.2f}")
    print(f"  Sortino: {sortino_ratio(returns):.2f}")
    
    print("\n[TEST] Calibration:")
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([0.8, 0.3, 0.9, 0.7, 0.4])
    ece = calibration_error(y_pred, y_true, bins=2)
    print(f"  ECE (Expected Calibration Error): {ece:.3f}")
