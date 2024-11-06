import pandas as pd

from data.labels import NEW_NAMED_COLS, FILE, TARGET


class DataSets:
    df = pd.read_csv(FILE)
    df.columns = NEW_NAMED_COLS

    train_df = df[df[NEW_NAMED_COLS[0]] == "train"]
    train_df = train_df.drop(columns="Origin")

    test_df = df[df[NEW_NAMED_COLS[0]] == "test"]

    x_train = train_df.drop(columns=NEW_NAMED_COLS[-1])
    y_train = train_df[NEW_NAMED_COLS[-1]]

    x_test = test_df.drop(columns=[NEW_NAMED_COLS[-1], NEW_NAMED_COLS[0]])
    y_test = test_df[TARGET]
