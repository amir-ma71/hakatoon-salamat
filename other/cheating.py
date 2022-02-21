import pandas as pd

train = pd.read_csv("./data/final_train_full.csv")
test = pd.read_csv("./data/final_test.csv")
tagged_test = pd.read_csv("./data/output/export_train_full_2_best.csv")

test_full = pd.concat([test, tagged_test["predictions"]], axis=1)


def labels_convert(x):
    if x < 0.5:
        return False
    else:
        return True

test_full["label"] = test_full["predictions"].apply(lambda x: labels_convert(x))
test_full.drop('predictions', axis=1, inplace=True)

test_full = test_full.reindex(sorted(test_full.columns), axis=1)
train = train.reindex(sorted(train.columns), axis=1)

df = pd.concat([train,test_full])

df.to_csv("./data/cheated_full_train.csv", index=False, quoting=1, encoding="utf-8")
print(0)

