import sys
from math import log
from typing import Tuple

import lightgbm as lgb
import modin.pandas as pd
import swifter
from matplotlib import pyplot
from pandas import DataFrame
from pandas import read_csv
from pandas import Series

from split import split_data


def choose_params_and_train(train_data_df: DataFrame) -> Tuple[dict, int, lgb.Booster]:
    base_params = {
        "objective": "binary",
        "lambda_l2": 0.01,
        "min_data_in_leaf": 10,
        "is_unbalance": True,
        "verbose": -1,
        "metric": "binary_logloss",
    }
    best_num_rounds = 50
    best_params = base_params
    best_logloss = 1000
    best_model = None
    for num_rounds in [50, 200]:
        for lambda_l2 in [0.01, 0.02, 0.05]:
            for min_data_in_leaf in [10, 25, 50]:
                base_params["lambda_l2"] = lambda_l2
                base_params["min_data_in_leaf"] = [min_data_in_leaf]
                cv_res = lgb.cv(
                    params=base_params,
                    train_set=train_data_df,
                    num_boost_round=num_rounds,
                    return_cvbooster=True,
                )
                if min(cv_res["binary_logloss-mean"]) < best_logloss:
                    best_logloss = min(cv_res["binary_logloss-mean"])
                    best_model = cv_res["cvbooster"].boosters[cv_res["cvbooster"].best_iteration]
                    best_params = base_params.copy()

    print((best_params, best_num_rounds, best_model))
    return (best_params, best_num_rounds, best_model)


def train_and_assess(
    full_train_data_df: DataFrame,
    pred_col: str,
    *,
    plot: bool = False,
    writeout: bool = False,
    test_data_df: DataFrame = None,
) -> dict:
    (train_data_df, valid_data_df) = split_data(0.2, full_train_data_df)
    if type(test_data_df) == DataFrame:
        print("Performing final testing")
        valid_data_df = test_data_df
    full_train_data = lgb.Dataset(
        full_train_data_df[[col for col in full_train_data_df.columns if col != pred_col]],
        full_train_data_df[pred_col],
        params={"verbose": -1},
    )

    train_data = lgb.Dataset(
        train_data_df[[col for col in train_data_df.columns if col != pred_col]],
        train_data_df[pred_col],
        params={"verbose": -1},
    )
    valid_data = lgb.Dataset(
        valid_data_df[[col for col in train_data_df.columns if col != pred_col]],
        valid_data_df[pred_col],
        params={"verbose": -1},
    )
    # print(train_data_df)
    # print(train_data)
    evals_result = dict()
    (params, num_round, gb_model) = choose_params_and_train(full_train_data)
    # gb_model = lgb.train(
    #    params,
    #    train_data,
    #    num_round,
    #    valid_sets=valid_data,
    #    valid_names="valid_set",
    #    callbacks=[lgb.log_evaluation(-1)],
    # )
    # print(evals_result)
    # print(type(gb_model))
    predictions = gb_model.predict(
        train_data_df[[col for col in train_data_df.columns if col != pred_col]]
    )
    valid_predictions = gb_model.predict(
        valid_data_df[[col for col in train_data_df.columns if col != pred_col]]
    )
    # print(predictions)
    train_data_df = train_data_df.reset_index()
    valid_data_df = valid_data_df.reset_index()
    train_data_df["preds"] = Series(predictions).swifter.apply(round)
    valid_data_df["preds"] = Series(valid_predictions).swifter.apply(round)
    # print( DataFrame(valid_predictions).columns)
    valid_results = dict()
    valid_results["accuracy"] = len(
        valid_data_df.loc[valid_data_df[pred_col] == valid_data_df["preds"]]
    ) / len(valid_data_df)
    if len(valid_data_df.loc[valid_data_df["preds"] == 1]) == 0:
        valid_results["precision"] = 0
    else:
        valid_results["precision"] = len(
            valid_data_df.loc[
                (valid_data_df[pred_col] == valid_data_df["preds"]) & (valid_data_df[pred_col] == 1)
            ]
        ) / len(valid_data_df.loc[valid_data_df["preds"] == 1])
    valid_results["recall"] = len(
        valid_data_df.loc[
            (valid_data_df[pred_col] == valid_data_df["preds"]) & (valid_data_df[pred_col] == 1)
        ]
    ) / len(valid_data_df.loc[valid_data_df[pred_col] == 1])
    valid_results["true_prevalence"] = len(valid_data_df.loc[valid_data_df[pred_col] == 1]) / len(
        valid_data_df
    )
    if valid_results["precision"] + valid_results["recall"] == 0:
        valid_results["F1"] = 0
    else:
        valid_results["F1"] = (2 * valid_results["precision"] * valid_results["recall"]) / (
            valid_results["precision"] + valid_results["recall"]
        )

    if plot:
        import matplotlib.pyplot as plt

        lgb.plot_tree(gb_model)
        lgb.plot_importance(gb_model)
        plt.show()
    if writeout:
        train_data_df.to_csv("res.csv", index=False)
        valid_data_df.to_csv("vres.csv", index=False)
    return valid_results


if __name__ == "__main__":
    train_data_df = read_csv("ingredient_train.csv")
    test_data_df = read_csv("ingredient_test.csv")

    # (train_data_df, valid_data_df) = split_data(0.2, train_data_df)
    pred_col_base = "pop_ingred_"
    total_results = {"accuracy": [], "precision": [], "recall": [], "true_prevalence": [], "F1": []}

    for i in range(0, 100):
        valid_results = train_and_assess(
            train_data_df, pred_col_base + str(i), test_data_df=test_data_df
        )
        for key, res in valid_results.items():
            total_results[key].append(res)
        print(f"Round {i} results: {valid_results}")
    print(total_results)

    plotable_res = []
    total_results_keys = list(total_results.keys())
    done_reformatting = False
    index = 0
    while not done_reformatting:
        plotable_res.append([])
        for key in total_results_keys:
            new_elem = None
            try:
                new_elem = total_results[key].pop()
            except IndexError:
                plotable_res.pop()
                done_reformatting = True
                break
            plotable_res[index].append(new_elem)
        print(total_results)
        index += 1
    for i in range(len(plotable_res)):
        # print([x for x in range(len(plotable_res[0]))])
        # print(plotable_res[i])
        pyplot.bar(
            [x - 0.9 * i / len(plotable_res) for x in range(len(plotable_res[0]))],
            plotable_res[i],
            width=0.9 / len(plotable_res),
            tick_label=total_results_keys,
        )
    pyplot.show()
