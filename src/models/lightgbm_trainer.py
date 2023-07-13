from typing import Any, Dict

import numpy as np
import lightgbm as lgb
from lightgbm.sklearn import LGBMRegressor

from ._base_gbdt_trainer import BaseGBDTTrainer


class LightGBMTrainer(BaseGBDTTrainer):
    def _set_model(self, params: Dict[str, Any]) -> LGBMRegressor:
        """Create a model intance.

        Args:
            params (Dict[str, Any]): Model parameters.

        Returns:
            Any: Model instance.
        """
        self.early_stopping_rounds = params.get("early_stopping_rounds", 50)
        _params = params.copy()
        _params.pop("early_stopping_rounds", None)
        return LGBMRegressor(**_params)

    def train(self) -> None:
        """Train LightGBM."""
        eval_set = [(self.X_valid, self.y_valid)] if self.y_valid is not None else None
        self._model = self._model.fit(self.X_train, self.y_train, eval_set=eval_set, callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)])
        self._best_iterations = self._model.best_iteration_
        self._feature_importances = self._model.feature_importances_

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probablities.

        Args:
            X (np.ndarray): Feature inputs.

        Returns:
            np.ndarray: Predicted probablities.
        """
        return self._model.predict(X)
