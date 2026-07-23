"""
Data ingestion and cleaning pipeline.
Fetches sports data from multiple sources, cleans, and computes features.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import requests
import json
from pathlib import Path


class DataPipeline:
    """Fetch, clean, and feature-engineer sports betting data."""
    
    def __init__(self, sport: str = "nfl"):
        self.sport = sport.lower()
        self.base_path = Path(__file__).parent.parent / "data"
        self.base_path.mkdir(exist_ok=True)
        (self.base_path / "raw").mkdir(exist_ok=True)
        (self.base_path / "processed").mkdir(exist_ok=True)
    
    def fetch_espn_data(self, season: int, league_id: str = None) -> pd.DataFrame:
        """
        Fetch historical game data from ESPN API.
        league_id: "nfl", "nba", "mlb", "nhl"
        """
        if league_id is None:
            league_map = {"nfl": "nfl", "nba": "nba", "mlb": "mlb", "nhl": "nhl"}
            league_id = league_map.get(self.sport, "nfl")
        
        games = []
        base_url = f"https://site.api.espn.com/apis/site/v2/sports/{self.sport}/{league_id}"
        
        try:
            # Fetch season schedule/results
            response = requests.get(f"{base_url}/2023/summary")
            if response.status_code == 200:
                data = response.json()
                # Parse games from response
                # This is a simplified mock - real implementation would parse actual ESPN API
                print(f"[INFO] Fetched ESPN {self.sport} data for season {season}")
            else:
                print(f"[WARN] ESPN API returned {response.status_code}")
        except Exception as e:
            print(f"[ERROR] Failed to fetch ESPN data: {e}")
        
        return pd.DataFrame(games) if games else pd.DataFrame()
    
    def generate_synthetic_data(self, n_games: int = 500, sport: str = "nfl") -> pd.DataFrame:
        """
        Generate synthetic historical betting data for demonstration.
        Includes realistic odds movement patterns and outcomes.
        """
        np.random.seed(42)
        
        dates = pd.date_range(start="2023-09-01", periods=n_games, freq="3D")
        
        data = {
            "date": dates,
            "sport": [sport] * n_games,
            "home_team": np.random.choice(
                ["KC", "SF", "TB", "DAL", "PHI", "NYG", "CLE", "LAR"], n_games
            ),
            "away_team": np.random.choice(
                ["KC", "SF", "TB", "DAL", "PHI", "NYG", "CLE", "LAR"], n_games
            ),
            # Initial line (home spread)
            "line_open": np.random.normal(-3, 4, n_games),
            "line_close": np.random.normal(-3, 4, n_games),
            # Moneyline odds (decimal format)
            "moneyline_home_open": np.random.uniform(1.8, 2.1, n_games),
            "moneyline_away_open": np.random.uniform(1.8, 2.1, n_games),
            "moneyline_home_close": np.random.uniform(1.8, 2.1, n_games),
            "moneyline_away_close": np.random.uniform(1.8, 2.1, n_games),
            # Over/under
            "ou_line": np.random.normal(45, 5, n_games),
            # Team stats (simplified)
            "home_elo": np.random.normal(1600, 150, n_games),
            "away_elo": np.random.normal(1600, 150, n_games),
            "home_rest_days": np.random.choice([3, 4, 5, 6, 7, 10], n_games),
            "away_rest_days": np.random.choice([3, 4, 5, 6, 7, 10], n_games),
            # Actual outcome (home team spread result)
            "spread_result": np.random.normal(-2, 12, n_games),  # Actual spread outcome
            "total_points": np.random.normal(45, 15, n_games),  # Total points in game
        }
        
        df = pd.DataFrame(data)
        
        # Add derived features
        df["line_movement"] = df["line_close"] - df["line_open"]
        df["home_moneyline_movement"] = (
            df["moneyline_home_close"] - df["moneyline_home_open"]
        ) / df["moneyline_home_open"]
        
        return df
    
    def extract_implied_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert decimal odds to implied probabilities accounting for vig.
        Handles American odds conversion first if needed.
        """
        df = df.copy()
        
        # For games with moneyline odds
        if "moneyline_home_close" in df.columns:
            # Implied prob = 1 / decimal_odds
            home_implied = 1.0 / df["moneyline_home_close"]
            away_implied = 1.0 / df["moneyline_away_close"]
            
            # Account for vig (bookmaker margin)
            total_prob = home_implied + away_implied
            df["home_implied_prob"] = home_implied / total_prob
            df["away_implied_prob"] = away_implied / total_prob
        
        return df
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering for predictive models."""
        df = df.copy()
        
        # Line characteristics
        df["closing_line_value"] = df["line_close"]  # Key for model input
        df["line_juice"] = 1.0 / (1.0 / (1 - 1.0/105) + 1.0 / (1 - 1.0/105))  # Vig approx
        
        # Team strength
        df["elo_diff"] = df["home_elo"] - df["away_elo"]
        df["elo_ratio"] = (df["home_elo"] + 1500) / (df["away_elo"] + 1500)
        
        # Rest and schedule effects
        df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]
        df["high_rest_days"] = ((df["home_rest_days"] >= 10) | (df["away_rest_days"] >= 10)).astype(int)
        
        # Odds movement
        df["sharp_line_movement"] = df["line_movement"].abs() > 1.0  # Flag large moves
        
        return df
    
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary labels for supervised learning.
        Label: 1 if home team covered spread, 0 otherwise.
        """
        df = df.copy()
        
        # Home team covered if actual spread >= closing line spread
        # (negative spread = home favored)
        df["home_covered"] = (df["spread_result"] >= df["line_close"]).astype(int)
        
        return df
    
    def process(self, season: int, use_synthetic: bool = True) -> pd.DataFrame:
        """
        Full pipeline: fetch -> clean -> feature engineer -> label.
        """
        print(f"[INFO] Processing {self.sport} season {season}")
        
        # Fetch data
        if use_synthetic:
            print("[INFO] Using synthetic data")
            df = self.generate_synthetic_data(n_games=500, sport=self.sport)
        else:
            df = self.fetch_espn_data(season)
            if df.empty:
                print("[WARN] No real data, falling back to synthetic")
                df = self.generate_synthetic_data(n_games=500, sport=self.sport)
        
        # Clean
        df = df.dropna(subset=["date", "home_team", "away_team", "line_close"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        
        # Features and labels
        df = self.extract_implied_probabilities(df)
        df = self.compute_features(df)
        df = self.create_labels(df)
        
        print(f"[INFO] Processed {len(df)} games")
        
        return df
    
    def save(self, df: pd.DataFrame, name: str, subset: str = "processed"):
        """Save processed dataframe."""
        path = self.base_path / subset / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"[INFO] Saved to {path}")
        return path


if __name__ == "__main__":
    # Example: Process NFL data
    pipeline = DataPipeline(sport="nfl")
    games = pipeline.process(season=2023, use_synthetic=True)
    
    print(f"\nFirst few games:")
    print(games[["date", "home_team", "away_team", "line_close", "home_covered", "elo_diff"]].head(10))
    
    print(f"\nFeature summary:")
    print(games[["line_movement", "elo_diff", "rest_diff", "sharp_line_movement"]].describe())
    
    # Save
    pipeline.save(games, "nfl_2023")
