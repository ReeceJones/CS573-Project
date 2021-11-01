import pandas
import ast

from split import split_data

_PRED_VAL_CAP = 8023  #789 * 4 

df = pandas.read_csv("/Users/nmerz/Downloads/archive/PP_recipes.csv")
df.ingredient_ids = df.ingredient_ids.apply(ast.literal_eval)
ingredient_max = df.ingredient_ids.apply(max).max()
print(ingredient_max)

def explode_to_cols(row):
    len_row = len(row[2])
    new_row = [row[0], row[1], len_row]
    for i in range(20):
        if len_row > i:
            if row[2][i] == 0:
                new_row.append(8023)
            else:
                new_row.append(row[2][i])
        else:
            new_row.append(0)
#    for col in range(2, len(new_row)):
#        if col == 5 + 2:
#            new_row[col] = int(new_row[col] == 0)
#        new_row[col] = min(new_row[col], _PRED_VAL_CAP)
    return pandas.Series(new_row, index = ["id", "calorie_level", "num_ingred"]+[str(i) for i in range(20)])

exploded = df[["id", "calorie_level", "ingredient_ids"]].apply(explode_to_cols, axis = 1)

raw_recipes = pandas.read_csv("/Users/nmerz/Downloads/archive/RAW_recipes.csv")

joint = exploded.join(raw_recipes[["id", "minutes", "n_steps"]].copy().set_index("id"), on="id")


(train, test) = split_data(.2, joint)


keep_cols = [*[str(i) for i in range(5)], "calorie_level", "num_ingred", "minutes", "n_steps"]
train_out = train[keep_cols].copy()
test_out = test[keep_cols].copy()


train_out.to_csv("ingredient_first4_train.csv", index = False)
test_out.to_csv("ingredient_first4_test.csv", index = False)

#for i in range(len(df[["ingredient_ids"]]) // 10000):
#    exploded_part = (df[["ingredient_ids"]].iloc[1000 * i: 1000 * (i+1)].explode("ingredient_ids").reset_index().pivot(columns="ingredient_ids").fillna(-1) + 1).astype(bool)
#    exploded = exploded.append(exploded_part)

