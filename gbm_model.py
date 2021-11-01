import lightgbm as lgb

from math import isnan
from numpy import transpose
from pandas import DataFrame, read_csv, Series

from split import split_data


def get_pred(comp_vals):
    max_val = 0
    pred = 0
    for i, val in enumerate(comp_vals):
        if val > max_val:
            pred = i
            max_val = val
    if isnan(pred):
        print(comp_vals)    
#    if pred > 20:
#        print(len(comp_vals))
    return pred


num_round = 10
train_data_df = read_csv("ingredient_first4_train.csv")
(train_data_df, valid_data_df) = split_data(.2, train_data_df)
params = {
    "objective": "multiclass",
    "lambda_l2": .01,
    "force_col_wise": True,
    "min_data_in_leaf": 10,
    "is_unbalanced": True,
    "num_classes": 21,
    "metric": "multi_logloss"
}
train_data = lgb.Dataset(train_data_df[[col for col in train_data_df.columns if col != "num_ingred"]], train_data_df["num_ingred"])
valid_data = lgb.Dataset(valid_data_df[[col for col in train_data_df.columns if col != "num_ingred"]], valid_data_df["num_ingred"])
print(train_data_df)
print(train_data)
evals_result = dict()
gb_model = lgb.train(params, train_data, num_round, valid_sets = valid_data, valid_names="valid_set", evals_result=evals_result)
print(evals_result)
print(type(gb_model))
predictions = gb_model.predict(train_data_df[[col for col in train_data_df.columns if col != "num_ingred"]])
valid_predictions = gb_model.predict(valid_data_df[[col for col in train_data_df.columns if col != "num_ingred"]])
print(predictions)
train_data_df = train_data_df.reset_index()
valid_data_df = valid_data_df.reset_index()
train_data_df["preds"] = DataFrame(predictions).apply(get_pred, axis=1)
valid_data_df["preds"] = DataFrame(valid_predictions).apply(get_pred, axis=1)
print( DataFrame(valid_predictions).columns)

train_data_df.to_csv("res.csv", index = False)
valid_data_df.to_csv("vres.csv", index = False)
#for i in range(len(predictions)):
#    if get_pred(predictions[i]) == 5:
#        print(i)

#train_data_df["preds_1"] = Series(predictions)
#train_data_df["preds"] = train_data_df["preds_1"] > .5
#    print(get_pred(predictions[i]))
#print(17)
#print(get_pred(predictions[17]))
"""
for index, pred in enumerate(predictions):
    diff = False
    for p1, p2 in zip(pred, predictions[0]):
        if p1 != p2:
            diff = True
            break
    if diff:
        print(index)
        print(pred)
        break
"""
import matplotlib.pyplot as plt

lgb.plot_tree(gb_model)
lgb.plot_importance(gb_model)
plt.show()
print(gb_model)
