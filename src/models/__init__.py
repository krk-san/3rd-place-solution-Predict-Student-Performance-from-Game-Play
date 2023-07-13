from .xgboost_trainer import XGBoostTrainer
from .lightgbm_trainer import LightGBMTrainer


TRAINERS = {"XGBoost": XGBoostTrainer, "LGBM": LightGBMTrainer}
