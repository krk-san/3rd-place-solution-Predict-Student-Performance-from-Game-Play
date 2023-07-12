"""Preprocessing and feature engineering methods."""
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import polars as pl
from const_values import FE_AGG_TARGETS, FLAG_EVENTS_TEXTS_PAIRS
from sklearn.model_selection import KFold


def seed_everything(seed: int = 42) -> None:
    """Set seed.

    Args:
        seed (int, optional): Seed number. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def drop_null_cols(df1: pl.DataFrame, df2: pl.DataFrame, df3: pl.DataFrame, th: float = 0.9) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, List[str], List[str], List[str], Dict[str, Dict[int, list]]]:
    """Drop columns with many null values based on a setting threshold.

    Args:
        df1 (pl.DataFrame): Feature dataframe for level group "0-4".
        df2 (pl.DataFrame): Feature dataframe for level group "5-12".
        df3 (pl.DataFrame): Feature dataframe for level group "13-22".
        th (float, optional): Threshold for percentage of null values. Defaults to 0.9.

    Returns:
        Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, List[str], List[str], List[str]]: Input Dataframes, feature columns and folds mapping.
    """
    null1 = df1.null_count() / len(df1)
    null2 = df2.null_count() / len(df1)
    null3 = df3.null_count() / len(df1)

    null1 = null1.transpose(include_header=True)
    null1 = null1.filter(pl.col("column_0") > th)

    null2 = null2.transpose(include_header=True)
    null2 = null2.filter(pl.col("column_0") > th)

    null3 = null3.transpose(include_header=True)
    null3 = null3.filter(pl.col("column_0") > th)

    drop1 = list(null1["column"])
    drop2 = list(null2["column"])
    drop3 = list(null3["column"])

    for col in df1.columns:
        if (df1[col].n_unique() == 1) and (col not in drop1):
            drop1.append(col)
    for col in df2.columns:
        if (df2[col].n_unique() == 1) and (col not in drop1):
            drop2.append(col)
    for col in df3.columns:
        if (df3[col].n_unique() == 1) and (col not in drop1):
            drop3.append(col)

    features1 = [c for c in df1.columns if c not in drop1 + ["level_group", "session_id"]]
    features2 = [c for c in df2.columns if c not in drop2 + ["level_group", "session_id"]]
    features3 = [c for c in df3.columns if c not in drop3 + ["level_group", "session_id"]]
    print("We will train with", len(features1), len(features2), len(features3), "features")
    all_users = df1["session_id"].unique()
    print("We will train with", len(all_users), "users info")
    return df1, df2, df3, features1, features2, features3


def load_train_and_label_csv(train_path: str, label_path: str, sort_cols: List[str] = ["session_id"]) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Load train and label csv.

    Args:
        train_path (str): Train data path.
        label_path (str): Label data path.
        sort_cols (List[str], optional): Columns on whict to base sorting. Defaults to ["session_id"].

    Returns:
        Tuple[pl.DataFrame, pl.DataFrame]: Train dataframe and label dataframe.
    """
    columns = [
        pl.col("page").cast(pl.Float32),
        ((pl.col("elapsed_time") - pl.col("elapsed_time").shift(1)).fill_null(0).clip(0, 1e9).over(["session_id", "level_group"]).alias("elapsed_time_diff")),
        ((pl.col("screen_coor_x") - pl.col("screen_coor_x").shift(1)).abs().over(["session_id", "level_group"]).alias("location_x_diff")),
        ((pl.col("screen_coor_y") - pl.col("screen_coor_y").shift(1)).abs().over(["session_id", "level_group"]).alias("location_y_diff")),
        ((pl.col("index") / pl.col("elapsed_time")).abs().alias("density")),
        pl.col("fqid").fill_null("fqid_None"),
        pl.col("text_fqid").fill_null("text_fqid_None"),
    ]

    train = pl.read_csv(train_path).sort(sort_cols).drop(["fullscreen", "hq", "music"]).with_columns(columns)

    label = (
        pl.read_csv(label_path)
        .select([pl.col("session_id").str.split("_q").arr.get(0).cast(pl.Int64), pl.col("session_id").str.split("_q").arr.get(1).cast(pl.Int32).alias("qid"), pl.col("correct").cast(pl.UInt8)])
        .sort(["session_id", "qid"])
        .groupby("session_id")
        .agg(pl.col("correct"))
        .select([pl.col("session_id"), *[pl.col("correct").arr.get(i).alias(f"correct_{i + 1}") for i in range(18)]])
    )

    return train, label


def prepare_train_and_label(cfg: Dict[str, Any]) -> Tuple[pl.DataFrame, pl.DataFrame, List[str], List[str]]:
    """Load train data and label data. Additional data is loaded and concatenated according to the config.

    Args:
        cfg (Dict[str, Any]): Config.

    Returns:
        Tuple[pl.DataFrame, pl.DataFrame, List[str], List[str]]: Train dataframe, Label dataframe, session_ids in competition data, and session_ids only in additional data.
    """
    # Load competition datasets.
    train_path = cfg["TRAIN_CSV"]
    label_path = cfg["LABEL_CSV"]
    train, label = load_train_and_label_csv(train_path=train_path, label_path=label_path, sort_cols=cfg.get("SORT_COLS", ["session_id"]))
    original_ids = list(train["session_id"].unique())

    # Load additional datasets(Optional).
    if cfg.get("USE_ADD"):
        train_path = cfg["ADD_TRAIN_CSV"]
        label_path = cfg["ADD_LABEL_CSV"]
        train_add, label_add = load_train_and_label_csv(train_path=train_path, label_path=label_path, sort_cols=cfg.get("SORT_COLS", ["session_id"]))

        # Remove session_ids of those who quit the game in the middle.
        if cfg.get("USE_ADD") == "FULL_LABEL_ONLY":
            label_add = label_add.drop_nulls()

        # Remove session_ids included in the competition dataset.
        label_add = label_add.filter(~pl.col("session_id").is_in(original_ids))
        add_ids = list(label_add["session_id"].unique())
        train_add = train_add.filter(pl.col("session_id").is_in(add_ids))

        train = pl.concat([train, train_add])
        label = pl.concat([label, label_add])

    add_ids = train.filter(~pl.col("session_id").is_in(original_ids)).select("session_id").unique().to_numpy().reshape(-1)

    return train, label, original_ids, add_ids


def join_features_and_labels(df: pl.DataFrame, label: pl.DataFrame) -> pl.DataFrame:
    df = df.join(label, on="session_id", how="left")
    df = df.sort("session_id")
    return df


def assign_fold_number_to_session_id(original_ids: List[str], add_ids: List[str], n_folds: int, seed: int) -> Dict[str, Dict[int, list]]:
    """Assign a fold number to each session_id for cross-validation.Competition data and additional data are handled separately.

    Args:
        original_ids (List[str]): List of session_ids in competition datasets.
        add_ids (List[str]): List of session_ids not in competition datasets.
        n_folds (int): Number of folds.
        seed (int): Number of seed.

    Returns:
        Dict[str, Dict[int, list]]: Mapping of session_id contained in each data type (competition data or additional data) and fold.
    """
    session_ids_folds_ori = defaultdict(list)
    session_ids_folds_add = defaultdict(list)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, valid_idx) in enumerate(kf.split(original_ids, groups=original_ids)):
        session_ids_folds_ori[fold] += [original_ids[i] for i in valid_idx]
    folds_mapping = {"ori": session_ids_folds_ori}

    if add_ids:
        for fold, (_, valid_idx) in enumerate(kf.split(add_ids, groups=add_ids)):
            session_ids_folds_add[fold] += [add_ids[i] for i in valid_idx]
        folds_mapping["add"] = session_ids_folds_ori

    return folds_mapping


def feature_engineer(x: pl.DataFrame, grp: str, feature_suffix: str, cfg: Dict[str, Any]) -> pl.DataFrame:
    """_summary_

    Args:
        x (pl.DataFrame): Dataframe used for feature creation
        grp (str): Level group.This time, just use it for the feature name.
        feature_suffix (str): Feature name suffix.
        cfg (Dict[str, Any]): Config.

    Returns:
        pl.DataFrame: Feature-added dataframe.
    """
    # Delete logs after restarting a game.
    if cfg.get("DROP_RESTART"):
        final_fqid = {"0-4": "chap1_finale_c", "5-12": "chap2_finale_c", "13-22": "chap4_finale_c"}
        x = x.filter(pl.col("elapsed_time") <= (pl.col("elapsed_time").filter(pl.col("fqid") == final_fqid[grp]).first().over("session_id")))

    # Click location features.
    if cfg.get("USE_LOC_DIFF"):
        columns = [
            ((pl.col("screen_coor_x") - pl.col("screen_coor_x").shift(1)).abs().over(["session_id", "level_group"]).alias("location_x_diff")),
            ((pl.col("screen_coor_y") - pl.col("screen_coor_y").shift(1)).abs().over(["session_id", "level_group"]).alias("location_y_diff")),
        ]

        x = x.with_columns(columns)

    # Various aggregate features.
    # base features
    aggs = [
        pl.col("index").count().alias(f"session_number_{feature_suffix}"),
        *[pl.col(c).drop_nulls().n_unique().alias(f"{c}_unique_{feature_suffix}") for c in cfg["CATS"]],
        *[pl.col(c).mean().alias(f"{c}_mean_{feature_suffix}") for c in cfg["NUMS"]],
        *[pl.col(c).min().alias(f"{c}_min_{feature_suffix}") for c in cfg["NUMS"]],
        *[pl.col(c).max().alias(f"{c}_max_{feature_suffix}") for c in cfg["NUMS"]],
        *[pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).mean().alias(f"{c}_ET_mean_{feature_suffix}") for c in cfg["EVENTS"]],
        *[pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).max().alias(f"{c}_ET_max_{feature_suffix}") for c in cfg["EVENTS"]],
        *[pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).min().alias(f"{c}_ET_min_{feature_suffix}") for c in cfg["EVENTS"]],
        *[pl.col("elapsed_time_diff").filter(pl.col("name") == c).mean().alias(f"{c}_ET_mean_{feature_suffix}") for c in cfg["NAMES"]],
        *[pl.col("elapsed_time_diff").filter(pl.col("name") == c).max().alias(f"{c}_ET_max_{feature_suffix}") for c in cfg["NAMES"]],
        *[pl.col("elapsed_time_diff").filter(pl.col("name") == c).min().alias(f"{c}_ET_min_{feature_suffix}") for c in cfg["NAMES"]],
    ]

    # Additional statistical features.
    if cfg.get("USE_MEDIAN"):
        aggs += [
            *[pl.col(c).median().alias(f"{c}_median_{feature_suffix}") for c in cfg["NUMS"]],
            *[pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).median().alias(f"{c}_ET_median_{feature_suffix}") for c in cfg["EVENTS"]],
            *[pl.col("elapsed_time_diff").filter(pl.col("name") == c).median().alias(f"{c}_ET_median_{feature_suffix}") for c in cfg["NAMES"]],
        ]

    if cfg.get("USE_QUANTILE"):
        aggs += [
            *[pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).quantile(0.1, "nearest").alias(f"{c}_ET_quantile1_{feature_suffix}") for c in cfg["EVENTS"]],
            # *[pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).quantile(0.2, "nearest").alias(f"{c}_ET_quantile2_{feature_suffix}") for c in cfg["EVENTS"]],
            *[pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).quantile(0.3, "nearest").alias(f"{c}_ET_quantile3_{feature_suffix}") for c in cfg["EVENTS"]],
            *[pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).quantile(0.7, "nearest").alias(f"{c}_ET_quantile7_{feature_suffix}") for c in cfg["EVENTS"]],
            # *[pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).quantile(0.8, "nearest").alias(f"{c}_ET_quantile8_{feature_suffix}") for c in cfg["EVENTS"]],
            *[pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).quantile(0.9, "nearest").alias(f"{c}_ET_quantile9_{feature_suffix}") for c in cfg["EVENTS"]],
        ]

    if cfg.get("USE_STD"):
        aggs += [
            *[pl.col(c).std().alias(f"{c}_std_{feature_suffix}") for c in cfg["NUMS"]],
            *[pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).std().alias(f"{c}_ET_std_{feature_suffix}") for c in cfg["EVENTS"]],
            *[pl.col("elapsed_time_diff").filter(pl.col("name") == c).std().alias(f"{c}_ET_std_{feature_suffix}") for c in cfg["NAMES"]],
        ]

    # Aggregate for ratio features described below.
    if cfg.get("USE_RATIO"):
        aggs += [
            *[pl.col("event_name").filter(pl.col("event_name") == c).count().alias(f"{c}_count_{feature_suffix}") for c in cfg["EVENTS"]],
            *[pl.col("elapsed_time_diff").filter(pl.col("event_name") == c).sum().alias(f"{c}_ET_sum_{feature_suffix}") for c in cfg["EVENTS"]],
            *[pl.col("name").filter(pl.col("name") == c).count().alias(f"{c}_count_{feature_suffix}") for c in cfg["NAMES"]],
            *[pl.col("elapsed_time_diff").filter(pl.col("name") == c).sum().alias(f"{c}_ET_sum_{feature_suffix}") for c in cfg["NAMES"]],
        ]

    # Aggregate features per level
    if cfg.get("USE_LEVEL"):
        aggs += [
            *[pl.col("level").filter(pl.col("level") == c).count().alias(f"{c}_count_{feature_suffix}") for c in cfg["LEVELS"][grp]],
            *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).sum().alias(f"{c}_sum_{feature_suffix}") for c in cfg["LEVELS"][grp]],
            *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).max().alias(f"{c}_max_{feature_suffix}") for c in cfg["LEVELS"][grp]],
            *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).min().alias(f"{c}_min_{feature_suffix}") for c in cfg["LEVELS"][grp]],
            *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).mean().alias(f"{c}_mean_{feature_suffix}") for c in cfg["LEVELS"][grp]],
        ]

    # Click location features (screen_coor).
    if cfg.get("USE_LOC_DIFF"):
        for key in ["x", "y"]:
            aggs += [
                *[pl.col(f"location_{key}_diff").filter(pl.col("event_name") == c).sum().alias(f"{c}_sum_{key}_{feature_suffix}") for c in cfg["EVENTS"]],
                *[pl.col(f"location_{key}_diff").filter(pl.col("event_name") == c).mean().alias(f"{c}_mean_{key}_{feature_suffix}") for c in cfg["EVENTS"]],
                *[pl.col(f"location_{key}_diff").filter(pl.col("event_name") == c).std().alias(f"{c}_std_{key}_{feature_suffix}") for c in cfg["EVENTS"]],
                *[pl.col(f"location_{key}_diff").filter(pl.col("event_name") == c).max().alias(f"{c}_max_{key}_{feature_suffix}") for c in cfg["EVENTS"]],
                *[pl.col(f"location_{key}_diff").filter(pl.col("event_name") == c).min().alias(f"{c}_min_{key}_{feature_suffix}") for c in cfg["EVENTS"]],
            ]

    # Number of times elapsed time is reversed.
    if cfg.get("USE_ET_MINUS"):
        aggs += [(pl.col("elapsed_time_diff") < 0).sum().alias(f"ET_minus_sum_{feature_suffix}")]

    # Number of times elapsed_time_diff exceeds a certain threshold.
    if cfg.get("USE_ET_DIFF_TH_COUNT"):
        aggs += [*[pl.col("elapsed_time_diff").filter(pl.col("elapsed_time_diff") > th).count().alias(f"ET_OVER_{th}_{feature_suffix}") for th in [10000, 100000, 500000, 1000000, 3600000]]]

    # Please refer to the following public notebook.
    # https://www.kaggle.com/code/janmpia/clues-clicks-eda-features#Clues-clicks-EDA-and-feature-enginnering-(10-new-features-at-the-end
    if cfg.get("USE_RIGHT_FQID"):
        fqid_xy = {
            "tunic": [50, 200, 200, 350],
            "plaque": [500, 590, -150, -80],
            "businesscards": [50, 150, -150, -100],
            "logbook": [-10000, 10000, -10, 40],
            "reader": [-300, -150, -120, -90],
            "right_directory": [-390, -60, -460, -280],
            "tracks": [950, 1100, -500, -320],
            "colorbook": [-10000, 10000, -10000, 10000],
            "reader_flag": [-290, 0, -90, 110],
            "journals_flag": [-10000, 10000, -10000, 10000],
        }
        for fqid_name, xy in fqid_xy.items():
            aggs += [
                pl.col("index")
                .filter(
                    (pl.col("fqid") == fqid_name)
                    & (pl.col("event_name") != "navigate_click")
                    & (~((pl.col("room_coor_x") > xy[0]) & (pl.col("room_coor_x") < xy[1]) & (pl.col("room_coor_y") > xy[2]) & (pl.col("room_coor_y") < xy[3])))
                )
                .count()
                .alias(f"{fqid_name}_mistakes_{feature_suffix}")
            ]

    # Please refer to the following public notebook.
    # https://www.kaggle.com/code/janmpia/person-clicks-eda-features
    if cfg.get("USE_RIGHT_PERSON"):
        rooms_person = {
            "tunic.historicalsociety.closet": [-470, -350, -50, 120],
            "tunic.historicalsociety.basement": [-50, 100, -200, -50],
            "tunic.historicalsociety.entry": [250, 400, 0, 180],
            "tunic.historicalsociety.entry": [50, 200, 0, 180],
            "tunic.historicalsociety.collection": [-200, -90, -75, 180],
            "tunic.historicalsociety.collection_flag": [-200, -90, -75, 180],
            "tunic.capitol_0.hall": [200, 300, -100, 100],
            "tunic.historicalsociety.closet_dirty": [-800, -700, -150, 0],
            "tunic.historicalsociety.frontdesk": [-130, 50, 0, 150],
            "tunic.humanecology.frontdesk": [-320, -180, -220, 65],
            "tunic.drycleaner.frontdesk": [-180, -40, -50, 120],
            "tunic.library.frontdesk": [-450, -350, -20, 260],
            "tunic.capitol_1.hall": [200, 280, -100, 100],
            "tunic.capitol_2.hall": [200, 280, -100, 100],
            "tunic.historicalsociety.cage": [-10, 100, -200, -100],
            "tunic.historicalsociety.cage": [-750, -650, -275, -225],
            "tunic.wildlife.center": [-850, -720, -700, -400],
            "tunic.wildlife.center": [650, 750, -610, -300],
            "tunic.flaghouse.entry": [200, 300, -30, 125],
        }
        for i, (room, xy) in enumerate(rooms_person.items()):
            aggs += [
                pl.col("index")
                .filter(
                    (pl.col("event_name") == "person_click")
                    & (pl.col("room_fqid") == room)
                    & (pl.col("fqid").is_not_null())
                    & (~((pl.col("room_coor_x") > xy[0]) & (pl.col("room_coor_x") < xy[1]) & (pl.col("room_coor_y") > xy[2]) & (pl.col("room_coor_y") < xy[3])))
                )
                .count()
                .alias(f"person{i}_clicks_{feature_suffix}")
            ]

    # Features per fqids containing important keywords.
    # I was going to create one for each group level, but ended up creating only 0-4.
    if cfg.get("USE_LEVELS_FQID"):
        fqid_list = [
            ["plaque", "plaque.face.date"],
            ["archivist", "directory.closeup.archivist", "archivist_glasses"],
            ["logbook", "logbook.page.bingo"],
        ]
        aggs += [
            *[pl.col("elapsed_time_diff").filter(pl.col("fqid").is_in(c)).count().alias(f"level_{c[0]}_count_{feature_suffix}") for c in fqid_list],
            *[pl.col("elapsed_time_diff").filter(pl.col("fqid").is_in(c)).std().alias(f"level_{c[0]}_std_{feature_suffix}") for c in fqid_list],
            *[pl.col("elapsed_time_diff").filter(pl.col("fqid").is_in(c)).sum().alias(f"level_{c[0]}_sum_{feature_suffix}") for c in fqid_list],
            *[pl.col("elapsed_time_diff").filter(pl.col("fqid").is_in(c)).max().alias(f"level_{c[0]}_max_{feature_suffix}") for c in fqid_list],
            *[pl.col("elapsed_time_diff").filter(pl.col("fqid").is_in(c)).min().alias(f"level_{c[0]}_min_{feature_suffix}") for c in fqid_list],
            *[pl.col("elapsed_time_diff").filter(pl.col("fqid").is_in(c)).mean().alias(f"level_{c[0]}_mean_{feature_suffix}") for c in fqid_list],
        ]

    # Aggregate features for each text, fqid and room_fqid.
    # FE_AGG example: ["text", "fqid", "room_fqid"]
    for filter_col in cfg.get("FE_AGG", []):
        targets = FE_AGG_TARGETS[filter_col][grp]
        agg_value_col = "elapsed_time_diff"
        cfg[f"{filter_col}_{grp}"] = targets
        for agg_method in ["count", "sum", "mean", "min"]:
            use_filter_inds = None
            aggs += make_filter_agg_expression(filter_col=filter_col, filter_values=targets, agg_value_col=agg_value_col, agg_method=agg_method, grp=grp, feature_suffix=feature_suffix, use_filter_inds=use_filter_inds)

        # Maximum Level for each text, fqid and room_fqid.
        if cfg.get("USE_LEVEL_AGG"):
            aggs += [
                *[pl.col("level").filter(pl.col(filter_col) == c).max().alias(f"{filter_col}_{i}_grp{grp}_level_{feature_suffix}") for i, c in enumerate(targets)],
            ]

    # Important flag events(fqids) features. Flag events mean events that absolutely must pass.
    # Calculate index counts and elapsed time between flag events.
    if cfg.get("USE_TEXT_FQIDS_DIFF"):
        cols_text_fqids_for_et_sum = []
        cols_text_fqids_for_ind_sum = []

        # text_fqids_diff is given a pair of the current flag fqid and the next flag fqid.
        text_fqids_diff = FLAG_EVENTS_TEXTS_PAIRS[grp]
        for i, text_fqids in enumerate(text_fqids_diff):
            aggs += [
                (pl.col("elapsed_time").filter(pl.col("text_fqid") == text_fqids[1]).min() - pl.col("elapsed_time").filter(pl.col("text_fqid") == text_fqids[0]).max()).alias(f"text_fqid{i}_duration_{feature_suffix}"),
                (pl.col("index").filter(pl.col("text_fqid") == text_fqids[1]).min() - pl.col("index").filter(pl.col("text_fqid") == text_fqids[0]).max()).alias(f"text_fqid{i}_indexcount_{feature_suffix}"),
            ]
            cols_text_fqids_for_et_sum.append(f"text_fqid{i}_duration_{feature_suffix}")
            cols_text_fqids_for_ind_sum.append(f"text_fqid{i}_indexcount_{feature_suffix}")

    df = x.groupby(["session_id"], maintain_order=True).agg(aggs).sort("session_id")

    # Target percentage of total sessions for index counts and time required
    if cfg.get("USE_RATIO"):
        df = df.with_columns(
            *[(pl.col(f"{c}_count_{feature_suffix}") / pl.col(f"session_number_{feature_suffix}")).alias(f"{c}_ratio_{feature_suffix}") for c in cfg["EVENTS"] + cfg["NAMES"]],
            *[(pl.col(f"{c}_ET_sum_{feature_suffix}") / pl.col(f"elapsed_time_max_{feature_suffix}")).alias(f"{c}_ET_ratio_{feature_suffix}") for c in cfg["EVENTS"] + cfg["NAMES"]],
        )

    # Predicted screen aspect ratio
    if cfg.get("USE_SCREEN_RATIO"):
        df = df.with_columns([(pl.col(f"screen_coor_y_max_{feature_suffix}") / pl.col(f"screen_coor_x_max_{feature_suffix}")).alias(f"screen_xy_ratio_{feature_suffix}")])  # これまとめた方が良いんじゃないの？

    # Datetime features.
    if cfg.get("USE_DATE"):
        df = df.with_columns(
            [
                pl.col("session_id").cast(pl.Utf8).str.slice(0, 2).cast(pl.Int8).alias(f"year_{feature_suffix}"),
                pl.col("session_id").cast(pl.Utf8).str.slice(2, 2).cast(pl.Int8).alias(f"month_{feature_suffix}"),
                pl.col("session_id").cast(pl.Utf8).str.slice(4, 2).cast(pl.Int8).alias(f"day_{feature_suffix}"),
                pl.col("session_id").cast(pl.Utf8).str.slice(6, 2).cast(pl.Int8).alias(f"hour_{feature_suffix}"),
            ]
        )

    # Sum of elapsed time and index counts for flag event features
    if cfg.get("USE_TEXT_FQIDS_DIFF_SUM") and cfg.get("USE_TEXT_FQIDS_DIFF"):
        df = df.with_columns(
            [
                pl.sum(cols_text_fqids_for_et_sum).alias(f"text_fqids_et_sum_{feature_suffix}"),
                pl.sum(cols_text_fqids_for_ind_sum).alias(f"text_fqids_ind_sum_{feature_suffix}"),
            ]
        )

    return df


def make_filter_agg_expression(filter_col: str, filter_values: List[str], agg_value_col: str, agg_method: str, grp: str, feature_suffix: str, use_filter_inds: bool = None) -> List[pl.Expr]:
    """Create polars expression which filters records using filter_col and filter_values and creages aggregate features for agg_value_col with agg_method.

    Args:
        filter_col (str): Column for filter.
        filter_values (List[str]): List of values that filter_col should match when filtering.
        agg_value_col (str): Aggregate target column.
        agg_method (str): Aggregate method.
        grp (str): Level group.This time, just use it for the feature name.
        feature_suffix (str): Feature name suffix.
        use_filter_inds (bool, optional): Indices of filter_values to be used this time. Defaults to None.

    Returns:
        List[pl.Expr]: _description_
    """
    if use_filter_inds is None:
        use_filter_inds = range(len(filter_values))
    if agg_method == "count":
        return [pl.col(agg_value_col).filter(pl.col(filter_col) == filter_values[i]).count().alias(f"{agg_value_col}_{filter_col}_{i}_grp{grp}_count_{feature_suffix}") for i in use_filter_inds]
    if agg_method == "sum":
        return [pl.col(agg_value_col).filter(pl.col(filter_col) == filter_values[i]).sum().alias(f"{agg_value_col}_{filter_col}_{i}_grp{grp}_sum_{feature_suffix}") for i in use_filter_inds]
    if agg_method == "mean":
        return [pl.col(agg_value_col).filter(pl.col(filter_col) == filter_values[i]).mean().alias(f"{agg_value_col}_{filter_col}_{i}_grp{grp}_mean_{feature_suffix}") for i in use_filter_inds]
    if agg_method == "min":
        return [pl.col(agg_value_col).filter(pl.col(filter_col) == filter_values[i]).min().alias(f"{agg_value_col}_{filter_col}_{i}_grp{grp}_min_{feature_suffix}") for i in use_filter_inds]
    if agg_method == "max":
        return [pl.col(agg_value_col).filter(pl.col(filter_col) == filter_values[i]).max().alias(f"{agg_value_col}_{filter_col}_{i}_grp{grp}_max_{feature_suffix}") for i in use_filter_inds]


def create_inputs(cfg: Dict[str, Any]) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, List[str], List[str], List[str], Dict[str, Dict[int, list]]]:
    """_summary_

    Args:
        cfg (Dict[str, Any]): Config.

    Returns:
        Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, List[str], List[str], List[str], Dict[str, Dict[int, list]]]: Input Dataframes, feature columns and folds mapping.
    """

    # Load train and label dataframes.
    df, label, original_ids, add_ids = prepare_train_and_label(cfg)

    # Split dataframes by level group.
    df1 = df.filter(pl.col("level_group") == "0-4")
    df2 = df.filter(pl.col("level_group") == "5-12")
    df3 = df.filter(pl.col("level_group") == "13-22")

    # Feature engineering.
    df1 = feature_engineer(df1, grp="0-4", feature_suffix="0-4", cfg=cfg)
    df2 = feature_engineer(df2, grp="5-12", feature_suffix="5-12", cfg=cfg)
    df3 = feature_engineer(df3, grp="13-22", feature_suffix="13-22", cfg=cfg)
    df1, df2, df3, features1, features2, features3 = drop_null_cols(df1=df1, df2=df2, df3=df3, th=cfg.get("TH_DROP_NULL"))

    # Join labels.
    df1 = join_features_and_labels(df1, label)
    df2 = join_features_and_labels(df2, label)
    df3 = join_features_and_labels(df3, label)

    # Assign a fold number to each session_id for cross-validation.
    folds_mapping = assign_fold_number_to_session_id(original_ids=original_ids, add_ids=add_ids, n_folds=cfg["NUM_FOLDS"], seed=cfg["SEED"])
    return df1, df2, df3, features1, features2, features3, folds_mapping
