from typing import Any, Dict

import numpy as np
from xgboost import XGBClassifier

from ._base_gbdt_trainer import BaseGBDTTrainer


class XGBoostTrainer(BaseGBDTTrainer):
    def _set_model(self, params: Dict[str, Any]) -> XGBClassifier:
        """Create a model intance.

        Args:
            params (Dict[str, Any]): Model parameters.

        Returns:
            Any: Model instance.
        """
        return XGBClassifier(**params)

    def train(self) -> None:
        """Train XGBoost."""
        eval_set = [(self.X_valid, self.y_valid)] if self.y_valid is not None else None
        self._model = self._model.fit(self.X_train, self.y_train, eval_set=eval_set, verbose=0)
        self._best_iterations = self._model.best_iteration
        self._feature_importances = self._model.feature_importances_

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probablities.

        Args:
            X (np.ndarray): Feature inputs.

        Returns:
            np.ndarray: Predicted probablities.
        """
        return self._model.predict_proba(X)[:, 1]
