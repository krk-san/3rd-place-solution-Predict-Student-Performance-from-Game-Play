import os
import pickle
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl


class BaseGBDTTrainer(metaclass=ABCMeta):
    def __init__(self, df_train: pl.DataFrame, feature_cols: List[str], q: int, fold: Optional[int] = None, df_valid: Optional[pl.DataFrame] = None, params: Dict[str, Any] = {}):
        """Abstract base class for GBDT-Trainer.

        Args:
            df_train (pl.DataFrame): Train data.
            feature_cols (List[str]): Feature columns.
            q (int): Question number.
            fold (Optional[int], optional): Fold number (when cross-validation). Defaults to None.
            df_valid (Optional[pl.DataFrame], optional): Validation data. Defaults to None.
            params (Dict[str, Any], optional): Model parameters. Defaults to {}.
        """
        # Attributes.
        self.q = q
        self.fold = fold

        # Prepare train and valid inputs.
        self.X_train, self.y_train = self._prepare_input(df=df_train, feature_cols=feature_cols, q=q)
        if df_valid is not None:
            self.X_valid, self.y_valid = self._prepare_input(df=df_valid, feature_cols=feature_cols, q=q)

        # Create a model instance.
        self._model = self._set_model(params)
        self._feature_importances = None
        self._best_iterations = None

    @abstractmethod
    def _set_model(self, params: Dict[str, Any]) -> Any:
        """Create a model intance.

        Args:
            params (Dict[str, Any]): Model parameters.

        Returns:
            Any: Model instance.
        """
        pass

    @abstractmethod
    def train(self) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probablities.

        Args:
            X (np.ndarray): Feature inputs.

        Returns:
            np.ndarray: Predicted probablities.
        """
        pass

    @staticmethod
    def _prepare_input(df: pl.DataFrame, feature_cols: List[str], q: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare model inputs.

        Args:
            df (pl.DataFrame): Dataframe.
            feature_cols (List[str]): Feature columns.
            q (int): Question number.

        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """
        label_col = f"correct_{q}"
        X = df.select(feature_cols).to_numpy()
        y = df.select(label_col).to_numpy().reshape(-1)
        return X, y

    def get_model(self) -> Any:
        return self._model

    def get_feature_importances(self) -> Optional[np.ndarray]:
        return self._feature_importances

    def get_best_iterations(self) -> Optional[int]:
        return self._best_iterations

    def get_y_yhat_valid(self) -> Tuple[np.ndarray, np.ndarray]:
        yhat_valid = self._predict_proba(self.X_valid)
        return self.y_valid, yhat_valid

    def save_model(self, save_dir) -> None:
        os.makedirs(save_dir, exist_ok=True)
        if self.fold is not None:
            file = f"{save_dir}/fold{self.fold}_q{self.q}.pkl"
        else:
            file = f"{save_dir}/overall_q{self.q}.pkl"
        pickle.dump(self._model, open(file, "wb"))
