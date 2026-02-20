"""
LightGBM Expert Model for IPF Progression Prediction
Uses hand-crafted and demographic features
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Tuple, List
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


class LightGBMExpert:
    """
    LightGBM Expert for progression prediction
    Uses hand-crafted features and demographics
    """
    
    def __init__(
        self,
        params: Dict = None,
        class_weights: np.ndarray = None
    ):
        """
        Args:
            params: LightGBM parameters
            class_weights: Class weights for imbalanced data [weight_neg, weight_pos]
        """
        if params is None:
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 15,
                'learning_rate': 0.01,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'max_depth': -1,
                'min_data_in_leaf': 5,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
                'verbose': -1,
                'seed': 42
            }
        
        self.params = params
        self.model = None
        self.class_weights = class_weights
        self.feature_names = None
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        feature_names: List[str] = None,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
        verbose: bool = True
    ):
        """
        Train LightGBM model
        
        Args:
            X_train: Training features (N_train, n_features)
            y_train: Training labels (N_train,)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            feature_names: List of feature names
            num_boost_round: Maximum number of boosting rounds
            early_stopping_rounds: Early stopping patience
            verbose: Print training progress
        """
        self.feature_names = feature_names
        
        # Prepare sample weights
        sample_weights = None
        if self.class_weights is not None:
            sample_weights = np.where(
                y_train == 1,
                self.class_weights[1],
                self.class_weights[0]
            )
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            weight=sample_weights,
            feature_name=feature_names
        )
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                reference=train_data,
                feature_name=feature_names
            )
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        # Train model
        callbacks = []
        if early_stopping_rounds and X_val is not None:
            callbacks.append(lgb.early_stopping(early_stopping_rounds))
        
        if verbose:
            callbacks.append(lgb.log_evaluation(period=50))
        
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        if verbose:
            print(f"\nBest iteration: {self.model.best_iteration}")
            print(f"Best score: {self.model.best_score}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions
        
        Args:
            X: Features (N, n_features)
        
        Returns:
            probabilities: (N,) probability of class 1
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        return self.model.predict(X, num_iteration=self.model.best_iteration)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Get binary predictions
        
        Args:
            X: Features (N, n_features)
            threshold: Classification threshold
        
        Returns:
            predictions: (N,) binary predictions
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance
        
        Args:
            importance_type: 'split' or 'gain'
        
        Returns:
            DataFrame with feature names and importance
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        importance = self.model.feature_importance(importance_type=importance_type)
        
        feature_names = self.feature_names if self.feature_names else [f'f{i}' for i in range(len(importance))]
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def save_model(self, path: str):
        """Save model to file"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        self.model.save_model(path)
    
    def load_model(self, path: str):
        """Load model from file"""
        self.model = lgb.Booster(model_file=path)


class LightGBMExpertTrainer:
    """Trainer wrapper for LightGBM Expert with cross-validation support"""
    
    def __init__(
        self,
        params: Dict = None,
        class_weights: np.ndarray = None
    ):
        self.params = params
        self.class_weights = class_weights
        self.model = LightGBMExpert(params=params, class_weights=class_weights)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        feature_names: List[str] = None,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
        verbose: bool = True
    ):
        """Train the model"""
        self.model.fit(
            X_train, y_train,
            X_val, y_val,
            feature_names=feature_names,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose
        )
    
    def predict(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray]:
        """
        Get predictions
        
        Returns:
            y_pred_proba: predicted probabilities
        """
        y_pred_proba = self.model.predict_proba(X)
        return y_pred_proba
    
    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Returns:
            metrics: Dict with AUC, accuracy, F1
        """
        y_pred_proba = self.model.predict_proba(X)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        metrics = {
            'auc': roc_auc_score(y_true, y_pred_proba),
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
        
        return metrics
