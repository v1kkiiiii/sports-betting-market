"""
NBA data pipeline — REAL DATA.

Source: NocturneBear/NBA-Data-2010-2024 (MIT licensed), sourced from official
NBA.com game logs. Covers regular season games 2010-11 through 2023-24,
~16,600 real games with box scores.

This pulls the raw team-game totals CSV, reconstructs one row per game
(home + away paired), and computes rolling/point-in-time features:
  - Elo ratings, updated game-by-game (no lookahead)
  - Rolling offensive/defensive rating (last 15 games, prior to this game)
  - Real rest days from actual game dates
  - Real back-to-back detection

No betting odds exist in this dataset (NBA.com doesn't publish historical
lines), so there is no synthetic odds generation here — see README for how
to plug in a real odds source. The label used for modeling is the real,
actual game outcome (home_won), not a simulated spread cover.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request

RAW_DATA_URL = (
    "https://raw.githubusercontent.com/NocturneBear/NBA-Data-2010-2024/"
    "main/regular_season_totals_2010_2024.csv"
)


class NBARealDataPipeline:
    """Fetches and processes real historical NBA game data (2010-2024)."""

    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).parent.parent / "data" / "nba"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.raw_path = self.cache_dir / "raw" / "nba_totals_2010_2024.csv"
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)

    def download_raw_data(self, force: bool = False) -> Path:
        """Download the real NBA game totals CSV (cached after first run)."""
        if self.raw_path.exists() and not force:
            print(f"[INFO] Using cached real NBA data: {self.raw_path}")
            return self.raw_path

        print(f"[INFO] Downloading real NBA game data (2010-2024)...")
        urllib.request.urlretrieve(RAW_DATA_URL, self.raw_path)
        size_mb = self.raw_path.stat().st_size / 1e6
        print(f"[INFO] Downloaded {size_mb:.1f} MB to {self.raw_path}")
        return self.raw_path

    def load_and_merge_games(self) -> pd.DataFrame:
        """
        Load raw team-game rows and merge into one row per game
        (home team stats + away team stats side by side).
        """
        raw_path = self.download_raw_data()
        df = pd.read_csv(raw_path)
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

        df["is_home"] = df["MATCHUP"].str.contains("vs.")

        home = df[df["is_home"]].copy()
        away = df[~df["is_home"]].copy()

        merged = home.merge(away, on="GAME_ID", suffixes=("_home", "_away"))
        merged = merged.rename(columns={"GAME_DATE_home": "date"})
        merged = merged.drop(columns=["GAME_DATE_away", "is_home_home", "is_home_away"])
        merged = merged.sort_values("date").reset_index(drop=True)

        return merged

    def compute_elo_ratings(
        self,
        df: pd.DataFrame,
        k_factor: float = 20,
        home_advantage: float = 100,
        initial_elo: float = 1500
    ) -> pd.DataFrame:
        """
        Compute real Elo ratings updated game-by-game in chronological order.
        The Elo BEFORE each game is what gets used as a feature (no lookahead —
        you can't know a team's post-game rating before the game happens).
        """
        df = df.sort_values("date").reset_index(drop=True)
        elo = {}

        home_elo_pre = []
        away_elo_pre = []

        for _, row in df.iterrows():
            home_team = row["TEAM_ABBREVIATION_home"]
            away_team = row["TEAM_ABBREVIATION_away"]

            h_elo = elo.get(home_team, initial_elo)
            a_elo = elo.get(away_team, initial_elo)

            home_elo_pre.append(h_elo)
            away_elo_pre.append(a_elo)

            expected_home = 1 / (1 + 10 ** ((a_elo - (h_elo + home_advantage)) / 400))

            home_won = 1 if row["PTS_home"] > row["PTS_away"] else 0

            elo[home_team] = h_elo + k_factor * (home_won - expected_home)
            elo[away_team] = a_elo + k_factor * ((1 - home_won) - (1 - expected_home))

        df["home_elo"] = home_elo_pre
        df["away_elo"] = away_elo_pre

        return df

    def compute_rolling_efficiency(self, df: pd.DataFrame, window: int = 15) -> pd.DataFrame:
        """
        Compute rolling points-for/points-against per team using only PRIOR
        games (shifted, so there's no lookahead). Used as an efficiency proxy
        since this dataset doesn't include possession counts for true ORTG/DRTG.
        """
        df = df.sort_values("date").reset_index(drop=True)

        home_log = df[["date", "GAME_ID", "TEAM_ABBREVIATION_home", "PTS_home", "PTS_away"]].rename(
            columns={"TEAM_ABBREVIATION_home": "team", "PTS_home": "pts_for", "PTS_away": "pts_against"}
        )
        away_log = df[["date", "GAME_ID", "TEAM_ABBREVIATION_away", "PTS_away", "PTS_home"]].rename(
            columns={"TEAM_ABBREVIATION_away": "team", "PTS_away": "pts_for", "PTS_home": "pts_against"}
        )
        team_log = pd.concat([home_log, away_log]).sort_values(["team", "date"]).reset_index(drop=True)

        team_log["roll_pts_for"] = (
            team_log.groupby("team")["pts_for"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=3).mean())
        )
        team_log["roll_pts_against"] = (
            team_log.groupby("team")["pts_against"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=3).mean())
        )

        home_roll = team_log.rename(columns={
            "team": "TEAM_ABBREVIATION_home",
            "roll_pts_for": "home_roll_pts_for",
            "roll_pts_against": "home_roll_pts_against"
        })[["GAME_ID", "TEAM_ABBREVIATION_home", "home_roll_pts_for", "home_roll_pts_against"]]

        away_roll = team_log.rename(columns={
            "team": "TEAM_ABBREVIATION_away",
            "roll_pts_for": "away_roll_pts_for",
            "roll_pts_against": "away_roll_pts_against"
        })[["GAME_ID", "TEAM_ABBREVIATION_away", "away_roll_pts_for", "away_roll_pts_against"]]

        df = df.merge(home_roll, on=["GAME_ID", "TEAM_ABBREVIATION_home"], how="left")
        df = df.merge(away_roll, on=["GAME_ID", "TEAM_ABBREVIATION_away"], how="left")

        return df

    def compute_rest_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """Real rest days computed from actual game dates per team."""
        df = df.sort_values("date").reset_index(drop=True)

        home_log = df[["date", "GAME_ID", "TEAM_ABBREVIATION_home"]].rename(
            columns={"TEAM_ABBREVIATION_home": "team"})
        away_log = df[["date", "GAME_ID", "TEAM_ABBREVIATION_away"]].rename(
            columns={"TEAM_ABBREVIATION_away": "team"})
        team_log = pd.concat([home_log, away_log]).sort_values(["team", "date"]).reset_index(drop=True)

        team_log["prev_game_date"] = team_log.groupby("team")["date"].shift(1)
        team_log["rest_days"] = (team_log["date"] - team_log["prev_game_date"]).dt.days

        home_rest = team_log.rename(columns={
            "team": "TEAM_ABBREVIATION_home", "rest_days": "home_rest_days"
        })[["GAME_ID", "TEAM_ABBREVIATION_home", "home_rest_days"]]

        away_rest = team_log.rename(columns={
            "team": "TEAM_ABBREVIATION_away", "rest_days": "away_rest_days"
        })[["GAME_ID", "TEAM_ABBREVIATION_away", "away_rest_days"]]

        df = df.merge(home_rest, on=["GAME_ID", "TEAM_ABBREVIATION_home"], how="left")
        df = df.merge(away_rest, on=["GAME_ID", "TEAM_ABBREVIATION_away"], how="left")

        df["home_back_to_back"] = (df["home_rest_days"] == 1).astype(int)
        df["away_back_to_back"] = (df["away_rest_days"] == 1).astype(int)

        return df

    def compute_features_and_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assemble final modeling features from the real, computed data above."""
        df = df.copy()

        df["elo_diff"] = df["home_elo"] - df["away_elo"]

        df["net_rtg_diff"] = (
            (df["home_roll_pts_for"] - df["home_roll_pts_against"]) -
            (df["away_roll_pts_for"] - df["away_roll_pts_against"])
        )

        df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]
        df["rest_advantage_pts"] = (df["home_back_to_back"] * -5) - (df["away_back_to_back"] * -5)

        # Real outcome — not simulated
        df["home_won"] = (df["PTS_home"] > df["PTS_away"]).astype(int)
        df["point_margin"] = df["PTS_home"] - df["PTS_away"]
        df["total_points"] = df["PTS_home"] + df["PTS_away"]

        df["home_team"] = df["TEAM_ABBREVIATION_home"]
        df["away_team"] = df["TEAM_ABBREVIATION_away"]
        df["home_score"] = df["PTS_home"]
        df["away_score"] = df["PTS_away"]

        return df

    def process(
        self,
        start_season: str = "2018-19",
        end_season: str = "2023-24",
        rolling_window: int = 15,
    ) -> pd.DataFrame:
        """
        Full pipeline: download real data -> merge -> compute real features.
        Elo and rolling stats are computed over the FULL history (so ratings
        are warmed up before the season range starts), then the dataset is
        filtered down to the requested season range.
        """
        print(f"[INFO] Loading real NBA data ({start_season} to {end_season})...")

        full_merged = self.load_and_merge_games()
        all_seasons = sorted(full_merged["SEASON_YEAR_home"].unique())
        if start_season not in all_seasons:
            print(f"[WARN] {start_season} not in data, available seasons: {all_seasons}")

        full_merged = self.compute_elo_ratings(full_merged)
        full_merged = self.compute_rolling_efficiency(full_merged, window=rolling_window)
        full_merged = self.compute_rest_days(full_merged)

        df = full_merged[
            (full_merged["SEASON_YEAR_home"] >= start_season) &
            (full_merged["SEASON_YEAR_home"] <= end_season)
        ].reset_index(drop=True)

        df = self.compute_features_and_labels(df)

        before = len(df)
        df = df.dropna(subset=["net_rtg_diff", "home_rest_days", "away_rest_days"]).reset_index(drop=True)
        print(f"[INFO] Dropped {before - len(df)} early-season rows lacking rolling history")
        print(f"[INFO] Final dataset: {len(df)} real games with complete features")

        return df

    def save(self, df: pd.DataFrame, name: str = "nba_real_processed"):
        path = self.cache_dir / "processed" / f"{name}.csv"
        path.parent.mkdir(exist_ok=True)
        df.to_csv(path, index=False)
        print(f"[INFO] Saved to {path}")
        return path


if __name__ == "__main__":
    pipeline = NBARealDataPipeline()

    print("\n" + "="*70)
    print("REAL NBA DATA PIPELINE")
    print("="*70 + "\n")

    df = pipeline.process(start_season="2018-19", end_season="2023-24")

    print(f"\nSample real games:")
    print(df[["date", "home_team", "away_team", "home_score", "away_score",
               "home_won", "elo_diff", "net_rtg_diff"]].head(10))

    print(f"\nReal outcome distribution:")
    print(f"  Home win rate: {df['home_won'].mean()*100:.1f}%")
    print(f"  Avg point margin (home): {df['point_margin'].mean():.2f}")
    print(f"  Avg total points: {df['total_points'].mean():.1f}")

    pipeline.save(df)
