import argparse
import datetime
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl
import yaml
from tqdm import tqdm

from models import TRAINERS
from postprocess import get_model_best_score
from preprocess import create_inputs, seed_everything


def train(
    df1: pl.DataFrame, df2: pl.DataFrame, df3: pl.DataFrame, features1: List[str], features2: List[str], features3: List[str], folds_mapping: Dict[str, Dict[int, list]], cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train models for each question.

    Args:
        df1 (pl.DataFrame): Dataframe of features and labels for grp-level "0-4".
        df2 (pl.DataFrame):  Dataframe of features and labels  for grp-level "5-12".
        df3 (pl.DataFrame):  Dataframe of features and labels  for grp-level "13-22".
        features1 (List[str]): Feature columns for grp-level "0-4".
        features2 (List[str]): Feature columns for grp-level "5-12".
        features3 (List[str]): Feature columns for grp-level "13-22".
        cfg (Dict[str, Any]): Config.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: True labels and oof prediction probablities.
    """

    # Store for feature creation.
    preds_dict = [defaultdict(float) for _ in range(18)]
    level_grp_max_time = [defaultdict(float) for _ in range(2)]
    level_grp_min_time = defaultdict(float)

    # Store for train results.
    models = {}
    results = [[[], []] for _ in range(18)]
    features_dict = {}

    # Base features.
    feature_cols_all_grp = {
        "0-4": features1,
        "5-12": features2,
        "13-22": features3,
    }

    # Base dataframe.
    df_all_grp = {
        "0-4": df1,
        "5-12": df2,
        "13-22": df3,
    }

    # Get the fold number for cross-validation. Only the data provided by the competition is used for the final evaluation.
    session_ids_folds_ori = folds_mapping["ori"]
    session_ids_folds_add = folds_mapping.get("add", {})
    session_ids_for_cv_score = []
    for fold in range(cfg["NUM_FOLDS"]):
        session_ids_for_cv_score += session_ids_folds_ori[fold]

    # Train per question.
    for q in tqdm(range(1, 19)):
        # Get target grp features
        if q <= 3:
            grp = "0-4"
        elif q <= 13:
            grp = "5-12"
        elif q <= 22:
            grp = "13-22"

        feature_cols = feature_cols_all_grp[grp].copy()
        df = df_all_grp[grp]

        # Feature creation that cannot be pre-computed.
        # Prediction probablities for previous questions.
        if cfg.get("USE_PREDS") and q > 1:
            [feature_cols.extend([f"preds_{t}"]) for t in range(1, q)]
            df = df.with_columns(
                [
                    *[pl.col("session_id").map_dict(preds_dict[t - 1]).alias(f"preds_{t}") for t in range(1, q)],
                ],
            )

        # Elapsed time between the start time of the previous level-grp and the end time of the current level-grp.
        if cfg.get("USE_ET_GAP"):
            # Between "0-4" and "5-12"
            if grp in ["5-12", "13-22"]:
                feature_cols.append("et_gap_1_")
                df = df.with_columns(
                    [
                        pl.col("session_id").map_dict(level_grp_max_time[0]).alias("et_max_1_"),
                        pl.col("session_id").map_dict(level_grp_min_time).alias("et_min_1_"),
                    ]
                )
                df = df.with_columns([(pl.col("et_min_1_") - pl.col("et_max_1_")).alias("et_gap_1_")])

                if cfg.get("USE_ET_GAP_RATIO") and q <= 13:
                    df = df.with_columns((pl.col(["et_gap_1_"]) / pl.col(f"elapsed_time_max_5-12")).alias("et_gap_1_ratio_5-12"))
                    feature_cols.extend(["et_gap_1_ratio_5-12"])

            # Between "5-12" and "13-22"
            if grp in ["13-22"]:
                feature_cols.append("et_gap_2_")
                df = df.with_columns([pl.col("session_id").map_dict(level_grp_max_time[1]).alias("et_max_2_")])
                df = df.with_columns([(pl.col(f"elapsed_time_min_{grp}") - pl.col("et_max_2_")).alias("et_gap_2_")])

                if cfg.get("USE_ET_GAP_RATIO"):
                    df = df.with_columns((pl.col("et_gap_2_") / pl.col(f"elapsed_time_max_13-22")).alias("et_gap_2_ratio_13-22"))
                    feature_cols.extend(["et_gap_2_ratio_13-22"])

        # Prediction binary values.
        if cfg.get("USE_PREDS_BIN") and cfg.get("USE_PREDS"):
            [feature_cols.extend([f"preds_bin_{t}"]) for t in range(1, q)]
            df = df.with_columns(
                [(pl.col(f"preds_{t}") > 0.62).cast(pl.Int8).alias(f"preds_bin_{t}") for t in range(1, q)],
            )

        # Sum of last i prediction probablities.
        if cfg.get("USE_PREDS_SUM") and cfg.get("USE_PREDS"):
            if q > 2:
                [feature_cols.extend([f"preds_sum_from_{q-i}_to_{q-1}"]) for i in range(2, 18) if q - i >= 1]
                for i in range(2, 18):
                    if q - i < 1:
                        continue
                    df = df.with_columns(pl.sum([f"preds_{t}" for t in range(q - i, q)]).alias(f"preds_sum_from_{q-i}_to_{q-1}"))

        features_dict[str(q)] = feature_cols.copy()

        # Cross validation.
        for fold in range(cfg["NUM_FOLDS"]):

            # Split train/valid.
            # Get validation session_ids.
            valid_ids_ori = session_ids_folds_ori[fold]
            valid_ids_add = session_ids_folds_add.get(fold, [])
            valid_ids = valid_ids_ori + valid_ids_add

            # Filter.
            df_train = df.filter(~pl.col("session_id").is_in(valid_ids))
            df_train = df_train.filter(~pl.col([f"correct_{q}"]).is_null())
            df_valid = df.filter(pl.col("session_id").is_in(valid_ids))
            df_valid = df_valid.filter(~pl.col([f"correct_{q}"]).is_null())

            # Train
            model_type = cfg["MODEL_TYPE"]
            trainer = TRAINERS[model_type](df_train=df_train, feature_cols=features_dict[str(q)], q=q, fold=fold, df_valid=df_valid, params=cfg["PARAMS"])
            trainer.train()

            # Save model.
            if cfg["SAVE"]:
                trainer.save_model(save_dir=cfg["SAVE_DIR"])
            models[(fold, q)] = trainer.get_model()

            # Get predict probablities of validation data for calculate the cv-score.
            y, yhat = trainer.get_y_yhat_valid()
            indices = np.where(np.isin(df_valid.select("session_id").to_numpy(), valid_ids_ori))[0]
            results[q - 1][0].append(y[indices])
            results[q - 1][1].append(yhat[indices])

            # Save some values for feature engineering.
            # Previous prediction probablities
            session_ids_val = df_valid.get_column("session_id")
            for idx in range(len(df_valid)):
                s_id = session_ids_val[idx]
                preds_dict[q - 1][s_id] = yhat[idx]

            # Elapsed time between previous level-grp and current level-grp.
            if cfg.get("USE_ET_GAP"):
                et_max = df_valid.get_column(f"elapsed_time_max_{grp}")
                et_min = df_valid.get_column(f"elapsed_time_min_{grp}")

                for idx in range(len(df_valid)):
                    s_id = valid_ids[idx]
                    if q == 3:
                        level_grp_max_time[0][s_id] = et_max[idx]
                    elif q == 4:
                        level_grp_min_time[s_id] = et_min[idx]
                    elif q == 13:
                        level_grp_max_time[1][s_id] = et_max[idx]

    # Get predict probablities of validation data for calculate the cv-score.
    results = [[np.concatenate(_) for _ in _] for _ in results]
    true = pd.DataFrame(np.stack([_[0] for _ in results]).T)
    true.index = session_ids_for_cv_score
    oof = pd.DataFrame(np.stack([_[1] for _ in results]).T)
    oof.index = session_ids_for_cv_score

    # Save feature columns and oof-predictions.
    if cfg.get("SAVE"):
        # features columns for each question.
        f_save = open(f"{cfg['SAVE_DIR']}/features_dict.pkl", "wb")
        pickle.dump(features_dict, f_save)
        f_save.close()

        # Oof-predictions.
        true.to_csv(f"{cfg['SAVE_DIR']}/true.csv")
        oof.to_csv(f"{cfg['SAVE_DIR']}/oof.csv")
    return true, oof


def main(options: argparse.Namespace):
    # Load Config
    print("********Load config********")
    cfg_path = Path(__file__).resolve().parent.parent / Path("config", options.cfg)
    with open(cfg_path) as file:
        cfg = yaml.safe_load(file)
    print("********Done********")

    # Save option.
    save = not options.no_save
    if save:
        cfg["SAVE"] = True
        cfg["NAME"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg["SAVE_DIR"] = f"{cfg['OUTPUT_ROOT']}/{cfg['NAME']}"
        seed_everything(cfg["SEED"])

    # Prepare inputs.
    print("********Prepare inputs********")
    df1, df2, df3, features1, features2, features3, folds_mapping = create_inputs(cfg)
    print("********Done********")

    # Train.
    print("********Train********")
    true, oof = train(df1, df2, df3, features1, features2, features3, folds_mapping, cfg)
    best_score = get_model_best_score(true, oof, visible=False)
    cfg["Overall F1"] = float(best_score)
    print("********Done********")

    # Save config.
    print("********Save config********")
    if cfg["SAVE"]:
        os.makedirs(cfg["SAVE_DIR"], exist_ok=True)
        with open(f"{cfg['SAVE_DIR']}/config.yml", "w") as f:
            yaml.dump(cfg, f)
        f_save = open(f"{cfg['SAVE_DIR']}/config.pkl", "wb")
        pickle.dump(cfg, f_save)
        f_save.close()
    print("********Done********")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="model.yaml path")
    parser.add_argument("--no_save", action="store_true")

    options = parser.parse_args()
    main(options)
