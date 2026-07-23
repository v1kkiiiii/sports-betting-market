"""
Predictive models for sports betting.
Ensemble approach: logistic regression + gradient boosting + neural network.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import pickle
from pathlib import Path


class BaseModel:
    """Base class for sports prediction models."""
    
    def __init__(self, name: str = "BaseModel"):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
    
    def get_features(self, df: pd.DataFrame):
        """Extract feature columns for modeling."""
        feature_cols = [
            "elo_diff", "rest_diff", "line_movement", 
            "closing_line_value", "home_moneyline_movement",
            "home_implied_prob", "elo_ratio"
        ]
        # Only use available features
        available = [col for col in feature_cols if col in df.columns]
        self.feature_names = available
        return df[available].fillna(0).values
    
    def fit(self, X, y):
        raise NotImplementedError
    
    def predict_proba(self, X):
        raise NotImplementedError
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class LogisticModel(BaseModel):
    """Simple logistic regression baseline."""
    
    def __init__(self, c: float = 1.0):
        super().__init__("LogisticRegression")
        self.model = LogisticRegression(C=c, max_iter=1000, random_state=42)
    
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        return self
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]


class GradientBoostModel(BaseModel):
    """Gradient boosting classifier for non-linear patterns."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 5, learning_rate: float = 0.05):
        super().__init__("GradientBoosting")
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            subsample=0.8
        )
    
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        return self
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]


class NeuralNetModel(BaseModel):
    """Simple neural network (implemented with sklearn MLPClassifier)."""
    
    def __init__(self, hidden_layers: tuple = (64, 32), learning_rate: float = 0.001):
        super().__init__("NeuralNet")
        from sklearn.neural_network import MLPClassifier
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            learning_rate_init=learning_rate,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            batch_size=32
        )
    
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        return self
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]


class EnsemblePredictor:
    """
    Ensemble combining logistic regression, gradient boosting, and neural net.
    Weighted average of predictions with calibration for probability estimates.
    """
    
    def __init__(self, weights: dict = None):
        self.models = {
            "logistic": LogisticModel(c=1.0),
            "gbm": GradientBoostModel(n_estimators=100, max_depth=5),
            "nn": NeuralNetModel(hidden_layers=(64, 32))
        }
        
        # Weights for ensemble
        self.weights = weights or {"logistic": 0.25, "gbm": 0.50, "nn": 0.25}
        self.feature_names = None
        self.is_trained = False
        self.calibrator = None
    
    def fit(self, X, y, calibrate: bool = True):
        """
        Train all base models and calibrate ensemble output.
        X: DataFrame with features
        y: Series/array of labels (1 = home covered, 0 = home didn't cover)
        """
        if isinstance(X, pd.DataFrame):
            feature_cols = self._get_features(X)
            X_array = X[feature_cols].fillna(0).values
        else:
            X_array = X
        
        y_array = y.values if isinstance(y, pd.Series) else y
        
        # Train individual models
        for name, model in self.models.items():
            print(f"[INFO] Training {name}...")
            model.get_features(pd.DataFrame(X_array))  # Set feature names
            model.fit(X_array, y_array)
            cv_scores = cross_val_score(
                model.model, 
                X_array, 
                y_array, 
                cv=5, 
                scoring="roc_auc"
            )
            print(f"      Cross-val AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Calibrate ensemble if requested
        if calibrate:
            print("[INFO] Calibrating ensemble...")
            # Use 20% holdout for calibration
            split_idx = int(0.8 * len(X_array))
            X_train = X_array[:split_idx]
            y_train = y_array[:split_idx]
            X_cal = X_array[split_idx:]
            y_cal = y_array[split_idx:]
            
            # Get base predictions on calibration set
            ensemble_preds = self._ensemble_predict_proba(X_cal)
            
            # Calibrate using simple logistic calibration
            # Fit a logistic regression to map ensemble output to true prob
            cal_model = LogisticRegression(random_state=42)
            cal_model.fit(ensemble_preds.reshape(-1, 1), y_cal)
            self.calibrator = cal_model
        
        self.is_trained = True
        print("[INFO] Ensemble training complete")
        return self
    
    def _get_features(self, df: pd.DataFrame):
        """Get list of available feature columns."""
        feature_cols = [
            "elo_diff", "rest_diff", "line_movement",
            "closing_line_value", "home_moneyline_movement",
            "home_implied_prob", "elo_ratio"
        ]
        return [col for col in feature_cols if col in df.columns]
    
    def _ensemble_predict_proba(self, X):
        """Unweighted ensemble probabilities."""
        preds = np.zeros(len(X))
        for name, model in self.models.items():
            weight = self.weights[name]
            preds += weight * model.predict_proba(X)
        return preds
    
    def predict_proba(self, X):
        """
        Predict home team cover probability.
        Returns array of probabilities [0, 1].
        """
        if isinstance(X, pd.DataFrame):
            feature_cols = self._get_features(X)
            X_array = X[feature_cols].fillna(0).values
        else:
            X_array = X
        
        preds = self._ensemble_predict_proba(X_array)
        
        # Apply calibration if available
        if self.calibrator:
            # Calibrator is a logistic regression mapping ensemble output to true prob
            preds = self.calibrator.predict_proba(preds.reshape(-1, 1))[:, 1]
        
        return np.clip(preds, 0.01, 0.99)  # Avoid extreme probabilities
    
    def predict(self, X, threshold: float = 0.5):
        """Predict home team cover (1) or not (0)."""
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def save(self, path: str):
        """Save ensemble to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[INFO] Saved ensemble to {path}")
    
    @staticmethod
    def load(path: str):
        """Load ensemble from disk."""
        with open(path, "rb") as f:
            ensemble = pickle.load(f)
        print(f"[INFO] Loaded ensemble from {path}")
        return ensemble


def evaluate_model(model, X_test, y_test):
    """Compute standard evaluation metrics."""
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
    
    preds_proba = model.predict_proba(X_test)
    preds = model.predict(X_test)
    
    y_test_array = y_test.values if isinstance(y_test, pd.Series) else y_test
    
    metrics = {
        "auc": roc_auc_score(y_test_array, preds_proba),
        "accuracy": accuracy_score(y_test_array, preds),
        "precision": precision_score(y_test_array, preds, zero_division=0),
        "recall": recall_score(y_test_array, preds, zero_division=0),
    }
    
    return metrics


if __name__ == "__main__":
    from pipeline import DataPipeline
    
    # Load data
    print("[INFO] Loading data...")
    pipeline = DataPipeline("nfl")
    df = pipeline.process(season=2023, use_synthetic=True)
    
    # Split train/test
    train_size = int(0.8 * len(df))
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]
    
    feature_cols = [
        "elo_diff", "rest_diff", "line_movement",
        "closing_line_value", "home_moneyline_movement",
        "home_implied_prob", "elo_ratio"
    ]
    
    X_train = df_train[feature_cols].fillna(0)
    y_train = df_train["home_covered"]
    X_test = df_test[feature_cols].fillna(0)
    y_test = df_test["home_covered"]
    
    # Train ensemble
    print("\n[INFO] Training ensemble...")
    ensemble = EnsemblePredictor()
    ensemble.fit(X_train, y_train, calibrate=True)
    
    # Evaluate
    print("\n[INFO] Evaluating on test set...")
    metrics = evaluate_model(ensemble, X_test, y_test)
    print(f"Test metrics: {metrics}")
    
    # Save
    Path("models").mkdir(exist_ok=True)
    ensemble.save("models/nfl_ensemble.pkl")
