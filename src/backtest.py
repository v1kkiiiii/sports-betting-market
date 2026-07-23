"""
Realistic backtesting engine for betting strategies.
Handles slippage, juice, position sizing (Kelly), and portfolio constraints.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple


@dataclass
class BetResult:
    """Result of a single bet."""
    game_id: int
    date: pd.Timestamp
    predicted_prob: float
    closing_odds: float
    closing_line: float
    bet_size: float
    outcome: int  # 1 if bet won, 0 if lost
    pnl: float
    edge_vs_market: float
    accepted: bool  # Whether bet was accepted


@dataclass
class PortfolioState:
    """Portfolio state at a point in time."""
    date: pd.Timestamp
    capital: float
    invested: float
    available: float
    cash: float
    total_bets: int
    winning_bets: int
    daily_pnl: float
    daily_return: float


class Backtest:
    """
    Realistic sports betting backtest engine.
    
    Features:
    - Kelly criterion sizing
    - Vigorish (juice) modeling
    - Slippage and line movement
    - Bet acceptance limits
    - Maximum position sizing constraints
    """
    
    def __init__(
        self,
        model,
        initial_capital: float = 10000,
        kelly_fraction: float = 0.25,
        min_edge: float = 0.02,  # 2% edge minimum to bet
        juice: float = 0.045,  # 4.5% bookmaker margin
        max_daily_bets: int = 20,
        max_bet_size_pct: float = 0.05,  # Max 5% per bet
        max_portfolio_exposure_pct: float = 0.50,  # Max 50% of capital at risk
        slippage_bps: float = 20,  # 20 bps line slippage on average
        name: str = "Strategy"
    ):
        self.model = model
        self.initial_capital = initial_capital
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.juice = juice
        self.max_daily_bets = max_daily_bets
        self.max_bet_size_pct = max_bet_size_pct
        self.max_portfolio_exposure_pct = max_portfolio_exposure_pct
        self.slippage_bps = slippage_bps
        self.name = name
        
        # State
        self.bets: List[BetResult] = []
        self.portfolio_history: List[PortfolioState] = []
        self.capital = initial_capital
        
    def kelly_size(
        self,
        prob_win: float,
        odds_decimal: float,
        kelly_fraction: float = None
    ) -> float:
        """
        Kelly Criterion position sizing.
        f* = (edge) / (odds - 1)
        Fractionalized to kelly_fraction for safety.
        """
        if kelly_fraction is None:
            kelly_fraction = self.kelly_fraction
        
        # Check if edge exists
        # If prob * odds > 1, we have positive expectation
        edge = prob_win * odds_decimal - 1
        if edge <= 0:
            return 0
        
        # Kelly: f = (prob * odds - 1) / (odds - 1)
        f = edge / (odds_decimal - 1)
        
        # Fractional Kelly
        return max(0, f * kelly_fraction)
    
    def compute_edge_vs_market(
        self,
        model_prob: float,
        implied_prob: float,
        odds_decimal: float
    ) -> float:
        """
        Edge = model probability - implied probability
        as % of potential winnings
        """
        return (model_prob - implied_prob) / max(implied_prob, 0.01)
    
    def account_for_slippage(self, odds: float, bet_direction: str = "under") -> float:
        """
        Model realistic slippage between signal and execution.
        Assumes bets occur at slightly worse odds.
        """
        slippage_factor = 1 + (self.slippage_bps / 10000)
        
        if bet_direction == "over":
            # Bet on favorite, odds get worse (higher)
            return odds * slippage_factor
        else:
            # Bet on underdog, odds get worse (lower)
            return odds / slippage_factor
    
    def size_bet(
        self,
        model_prob: float,
        implied_prob: float,
        odds_decimal: float,
        available_capital: float,
        daily_bet_count: int,
    ) -> Tuple[float, bool]:
        """
        Determine bet size using Kelly, respecting constraints.
        Returns (bet_size, was_accepted)
        """
        # Check daily bet limit
        if daily_bet_count >= self.max_daily_bets:
            return 0, False
        
        # Check minimum edge
        edge = self.compute_edge_vs_market(model_prob, implied_prob, odds_decimal)
        if edge < self.min_edge:
            return 0, False
        
        # Kelly sizing
        kelly_size = self.kelly_size(model_prob, odds_decimal)
        if kelly_size <= 0:
            return 0, False
        
        # Apply portfolio constraints
        max_bet = available_capital * self.max_bet_size_pct
        max_portfolio_exposure = available_capital * self.max_portfolio_exposure_pct
        
        bet_size = min(
            kelly_size * available_capital,
            max_bet,
            max_portfolio_exposure
        )
        
        return bet_size, True
    
    def process_game(
        self,
        game_idx: int,
        row: pd.Series,
        daily_bet_count: int,
        capital: float
    ) -> Tuple[Optional[BetResult], int]:
        """
        Process a single game for betting.
        Returns (BetResult or None, updated daily bet count)
        """
        try:
            # Get available features dynamically (works with NFL, NBA, etc)
            available_features = [
                "elo_diff", "rest_diff", "line_movement",
                "closing_line_value", "home_moneyline_movement",
                "home_implied_prob", "elo_ratio",
                # NBA-specific
                "net_rtg_diff", "rest_advantage_pts", "spread_movement", "ou_diff"
            ]
            features_to_use = [f for f in available_features if f in row.index]
            
            # Get model prediction with available features
            X_game = row[features_to_use].fillna(0).values.reshape(1, -1)
            
            model_prob = self.model.predict_proba(X_game)[0]
            implied_prob = row.get("home_implied_prob", 0.5)
            odds_decimal = row.get("moneyline_home_close", 2.0)
            
            # Size bet
            bet_size, accepted = self.size_bet(
                model_prob,
                implied_prob,
                odds_decimal,
                capital,
                daily_bet_count
            )
            
            if not accepted or bet_size <= 0:
                return None, daily_bet_count
            
            # Apply slippage
            slippage_odds = self.account_for_slippage(odds_decimal, "under")
            
            # Simulate outcome
            outcome = int(row.get("home_covered", 0))  # 1 if home covered, 0 otherwise
            
            # P&L calculation
            if outcome == 1:
                # Bet won
                pnl = bet_size * (slippage_odds - 1)
            else:
                # Bet lost
                pnl = -bet_size
            
            # Apply juice if we won (bookmaker takes cut)
            if outcome == 1:
                pnl = pnl * (1 - self.juice / 2)  # Simplified vig
            
            edge = self.compute_edge_vs_market(model_prob, implied_prob, odds_decimal)
            
            bet_result = BetResult(
                game_id=game_idx,
                date=row["date"],
                predicted_prob=model_prob,
                closing_odds=odds_decimal,
                closing_line=row.get("line_close", 0),
                bet_size=bet_size,
                outcome=outcome,
                pnl=pnl,
                edge_vs_market=edge,
                accepted=True
            )
            
            return bet_result, daily_bet_count + 1
        
        except Exception as e:
            print(f"[WARN] Error processing game {game_idx}: {e}")
            return None, daily_bet_count
    
    def run(self, df: pd.DataFrame, verbose: bool = True) -> "BacktestResults":
        """
        Run backtest over game sequence.
        df: DataFrame with games in chronological order.
        """
        if verbose:
            print(f"\n[INFO] Running backtest: {self.name}")
            print(f"      Initial capital: ${self.initial_capital:.2f}")
            print(f"      Kelly fraction: {self.kelly_fraction}")
            print(f"      Min edge: {self.min_edge*100:.1f}%")
        
        df = df.sort_values("date").reset_index(drop=True)
        
        self.capital = self.initial_capital
        self.bets = []
        self.portfolio_history = []
        
        current_date = None
        daily_bet_count = 0
        daily_pnl = 0
        
        for idx, row in df.iterrows():
            # Reset daily counters
            if current_date != row["date"]:
                if current_date is not None:
                    daily_return = daily_pnl / self.capital if self.capital > 0 else 0
                    self.portfolio_history.append(PortfolioState(
                        date=current_date,
                        capital=self.capital,
                        invested=0,  # Simplified
                        available=self.capital,
                        cash=self.capital,
                        total_bets=len(self.bets),
                        winning_bets=sum(1 for b in self.bets if b.outcome == 1),
                        daily_pnl=daily_pnl,
                        daily_return=daily_return
                    ))
                
                current_date = row["date"]
                daily_bet_count = 0
                daily_pnl = 0
            
            # Process game
            bet_result, daily_bet_count = self.process_game(
                idx, row, daily_bet_count, self.capital
            )
            
            if bet_result:
                self.bets.append(bet_result)
                self.capital += bet_result.pnl
                daily_pnl += bet_result.pnl
        
        # Close last day
        if current_date:
            daily_return = daily_pnl / self.capital if self.capital > 0 else 0
            self.portfolio_history.append(PortfolioState(
                date=current_date,
                capital=self.capital,
                invested=0,
                available=self.capital,
                cash=self.capital,
                total_bets=len(self.bets),
                winning_bets=sum(1 for b in self.bets if b.outcome == 1),
                daily_pnl=daily_pnl,
                daily_return=daily_return
            ))
        
        results = BacktestResults(
            backtest=self,
            bets=self.bets,
            portfolio_history=self.portfolio_history,
            final_capital=self.capital
        )
        
        if verbose:
            print(results.summary())
        
        return results


class BacktestResults:
    """Results and analysis of backtest."""
    
    def __init__(self, backtest: Backtest, bets: List[BetResult],
                 portfolio_history: List[PortfolioState], final_capital: float):
        self.backtest = backtest
        self.bets = bets
        self.portfolio_history = portfolio_history
        self.final_capital = final_capital
    
    def summary(self) -> str:
        """Summary statistics."""
        if not self.bets:
            return "[WARN] No bets placed"
        
        df_bets = pd.DataFrame([
            {
                "date": b.date,
                "predicted_prob": b.predicted_prob,
                "outcome": b.outcome,
                "pnl": b.pnl,
                "edge": b.edge_vs_market,
                "bet_size": b.bet_size
            }
            for b in self.bets
        ])
        
        wins = (df_bets["outcome"] == 1).sum()
        total = len(df_bets)
        win_rate = wins / total if total > 0 else 0
        
        avg_pnl = df_bets["pnl"].mean()
        total_pnl = df_bets["pnl"].sum()
        cumulative_return = total_pnl / self.backtest.initial_capital
        
        # Sharpe ratio
        daily_returns = [p.daily_return for p in self.portfolio_history]
        daily_ret_array = np.array(daily_returns)
        if len(daily_ret_array) > 1:
            sharpe = np.mean(daily_ret_array) / np.std(daily_ret_array) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Max drawdown
        cumulative = np.cumsum([b.pnl for b in self.bets])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / np.abs(running_max + 1)
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        
        summary_text = f"""
==== BACKTEST RESULTS ====
Strategy: {self.backtest.name}
Initial Capital: ${self.backtest.initial_capital:,.2f}
Final Capital: ${self.final_capital:,.2f}
Total Return: {cumulative_return*100:.2f}% (${total_pnl:,.2f})

Betting Statistics:
- Total Bets: {total}
- Winning Bets: {wins}
- Win Rate: {win_rate*100:.1f}%
- Average Win: ${df_bets[df_bets['outcome']==1]['pnl'].mean():,.2f}
- Average Loss: ${df_bets[df_bets['outcome']==0]['pnl'].mean():,.2f}
- Profit Factor: {df_bets[df_bets['outcome']==1]['pnl'].sum() / abs(df_bets[df_bets['outcome']==0]['pnl'].sum()):.2f}

Risk Metrics:
- Sharpe Ratio: {sharpe:.2f}
- Max Drawdown: {max_drawdown*100:.1f}%
- Avg Daily Return: {np.mean(daily_ret_array)*100:.3f}%
- Volatility: {np.std(daily_ret_array)*100:.3f}%

Edge Analysis:
- Avg Edge vs Market: {df_bets['edge'].mean()*100:.2f}%
- Avg Predicted Prob: {df_bets['predicted_prob'].mean():.3f}
        """
        return summary_text
    
    def to_dataframe(self) -> pd.DataFrame:
        """Export bets to DataFrame."""
        return pd.DataFrame([
            {
                "date": b.date,
                "predicted_prob": b.predicted_prob,
                "closing_odds": b.closing_odds,
                "closing_line": b.closing_line,
                "bet_size": b.bet_size,
                "outcome": b.outcome,
                "pnl": b.pnl,
                "edge_vs_market": b.edge_vs_market
            }
            for b in self.bets
        ])


if __name__ == "__main__":
    from pipeline import DataPipeline
    from models import EnsemblePredictor
    
    # Load data and model
    print("[INFO] Loading data...")
    pipeline = DataPipeline("nfl")
    df = pipeline.process(season=2023, use_synthetic=True)
    
    # Split
    train_size = int(0.8 * len(df))
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]
    
    print("[INFO] Training model...")
    ensemble = EnsemblePredictor()
    ensemble.fit(df_train[["elo_diff", "rest_diff", "line_movement",
                           "closing_line_value", "home_moneyline_movement",
                           "home_implied_prob", "elo_ratio"]],
                 df_train["home_covered"])
    
    # Run backtest
    print("[INFO] Running backtest...")
    bt = Backtest(
        ensemble,
        initial_capital=10000,
        kelly_fraction=0.25,
        min_edge=0.02,
        name="Ensemble Kelly Strategy"
    )
    results = bt.run(df_test)
    
    # Export
    results.to_dataframe().to_csv("data/results/backtest_results.csv", index=False)
    print("\n[INFO] Results saved to data/results/backtest_results.csv")
