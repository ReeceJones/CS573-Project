import warnings
import modin.pandas as pd
import pandas
import swifter
import ast

from typing import List, Tuple, Mapping
from functools import partial

from split import split_data

warnings.filterwarnings(action="ignore", category=UserWarning, module=r".*modin")


def filter_and_one_hot(row: pandas.Series, *, top100_map: Mapping[int, int]):
    len_row = len(row[2])
    new_row = [row[0], row[1], len_row]
    offset = len(new_row)
    for i in range(len(top100_map)):
        new_row.append(0)
    for i in range(len_row):
        ingred = row[2][i]
        if ingred in top100_map.keys():
            new_row[offset + top100_map[ingred]] = 1
    return pandas.Series(
        new_row,
        index=["id", "calorie_level", "num_ingred"]
        + ["pop_ingred_" + str(i) for i in range(len(top100_map))],
    )


if __name__ == "__main__":

    df = pd.read_csv("PP_recipes.csv")
    df.ingredient_ids = df.ingredient_ids.swifter.apply(ast.literal_eval)
    ingredient_max = df.ingredient_ids.swifter.apply(max).max()
    print(ingredient_max)

    # exploded = df[["id", "calorie_level", "ingredient_ids"]].apply(explode_to_cols, axis = 1)

    exploded = df[["id", "ingredient_ids"]].explode("ingredient_ids")
    print(exploded)
    counts = exploded.groupby("ingredient_ids").agg({"id": "count"})
    print(counts.reset_index().loc[counts.reset_index()["ingredient_ids"] == 4096])
    top100 = enumerate(list(counts.sort_values("id", ascending=False).head(100).index))
    top100_mapping = dict()
    for ingred in top100:
        top100_mapping[ingred[1]] = ingred[0]
    print(top100_mapping)

    exploded2 = df[["id", "calorie_level", "ingredient_ids"]].swifter.apply(
        filter_and_one_hot, top100_map=top100_mapping, axis=1
    )
    print(exploded2)

    raw_recipes = pd.read_csv("RAW_recipes.csv")

    joint = exploded2.join(
        raw_recipes[["id", "minutes", "n_steps"]].copy().set_index("id"), on="id"
    )

    (train, test) = split_data(0.2, joint)

    # keep_cols = [*[str(i) for i in range(5)], "calorie_level", "num_ingred", "minutes", "n_steps"]
    # train_out = train[keep_cols].copy()
    # test_out = test[keep_cols].copy()

    train.to_csv("ingredient_train.csv", index=False)
    test.to_csv("ingredient_test.csv", index=False)

    # for i in range(len(df[["ingredient_ids"]]) // 10000):
    #    exploded_part = (df[["ingredient_ids"]].iloc[1000 * i: 1000 * (i+1)].explode("ingredient_ids").reset_index().pivot(columns="ingredient_ids").fillna(-1) + 1).astype(bool)
    #    exploded = exploded.append(exploded_part)
