import math
from collections import Counter

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def entropy(labels):
    total = len(labels)
    counts = Counter(labels)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def information_gain(df, feature, target):
    base = entropy(df[target])
    weighted = 0
    for value, subset in df.groupby(feature):
        weighted += (len(subset) / len(df)) * entropy(subset[target])
    return base - weighted


def build_tree(df, features, target):
    labels = df[target]

    # Stop 1: all rows have the same class
    if labels.nunique() == 1:
        return labels.iloc[0]

    # Stop 2: no features left, use the most common class
    if not features:
        return labels.mode()[0]

    # Pick the feature with the highest information gain
    best = max(features, key=lambda f: information_gain(df, f, target))
    tree = {best: {}}
    remaining = [f for f in features if f != best]

    for value, subset in df.groupby(best):
        tree[best][value] = build_tree(subset, remaining, target)

    # Fallback class for values never seen in training
    tree[best]["_default"] = labels.mode()[0]
    return tree


def predict_one(tree, row):
    if not isinstance(tree, dict):
        return tree
    feature = next(iter(tree))
    value = row[feature]
    branch = tree[feature].get(value, tree[feature]["_default"])
    return predict_one(branch, row)


def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(indent + "-> " + str(tree))
        return
    feature = next(iter(tree))
    for value, subtree in tree[feature].items():
        if value == "_default":
            continue
        print(f"{indent}{feature} = {value}")
        print_tree(subtree, indent + "    ")


def run_id3():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=["sepal_len", "sepal_wid", "petal_len", "petal_wid"])
    df["species"] = [iris.target_names[i] for i in iris.target]

    # Turn numbers into 3 categories (ID3 needs categories)
    features = ["sepal_len", "sepal_wid", "petal_len", "petal_wid"]
    for f in features:
        df[f] = pd.cut(df[f], bins=3, labels=["low", "medium", "high"]).astype(str)

    train, test = train_test_split(df, test_size=0.3, random_state=42)

    tree = build_tree(train, features, "species")

    predictions = test.apply(lambda row: predict_one(tree, row), axis=1)
    accuracy = (predictions == test["species"]).mean()

    print("=== ID3 ===")
    print("Accuracy:", round(accuracy, 4))
    print("\nTree:")
    print_tree(tree)
    return accuracy


if __name__ == "__main__":
    run_id3()