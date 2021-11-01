import pandas

def split_data(split_size: float, in_data: pandas.DataFrame, random_state: int = None) -> (pandas.DataFrame, pandas.DataFrame):
    test_data = in_data.sample(frac=split_size, random_state=random_state)

    test_data["test"] = True
    in_and_test = in_data.join(test_data[["test"]])
    train_data = in_and_test.loc[in_and_test["test"] != True].copy()[[col for col in in_and_test.columns if col != "test"]]

    test_data = test_data[[col for col in test_data.columns if col != "test"]]
    return (train_data, test_data)
