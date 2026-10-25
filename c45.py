import math
from collections import Counter

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def entropy(labels):
    total = len(labels)
    counts = Counter(labels)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def best_split(df, features, target):
    """Find the feature and threshold with the highest gain ratio."""
    base = entropy(df[target])
    best = (None, None, 0.0)  # feature, threshold, gain ratio

    for f in features:
        values = sorted(df[f].unique())
        # candidate thresholds: midpoints between neighbouring values
        for a, b in zip(values[:-1], values[1:]):
            t = (a + b) / 2
            left = df[df[f] <= t]
            right = df[df[f] > t]
            p_left = len(left) / len(df)
            p_right = 1 - p_left

            gain = base - (p_left * entropy(left[target]) + p_right * entropy(right[target]))
            split_info = -(p_left * math.log2(p_left) + p_right * math.log2(p_right))
            if split_info == 0:
                continue

            ratio = gain / split_info
            if ratio > best[2]:
                best = (f, t, ratio)
    return best


def build_tree(df, features, target, depth=0, max_depth=5):
    labels = df[target]

    if labels.nunique() == 1 or depth >= max_depth or len(df) < 2:
        return labels.mode()[0]

    feature, threshold, ratio = best_split(df, features, target)
    if feature is None:
        return labels.mode()[0]

    left = df[df[feature] <= threshold]
    right = df[df[feature] > threshold]
    return {
        "feature": feature,
        "threshold": threshold,
        "left": build_tree(left, features, target, depth + 1, max_depth),
        "right": build_tree(right, features, target, depth + 1, max_depth),
    }


def predict_one(tree, row):
    if not isinstance(tree, dict):
        return tree
    branch = "left" if row[tree["feature"]] <= tree["threshold"] else "right"
    return predict_one(tree[branch], row)


def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(indent + "-> " + str(tree))
        return
    print(f"{indent}{tree['feature']} <= {tree['threshold']:.2f}")
    print_tree(tree["left"], indent + "    ")
    print(f"{indent}{tree['feature']} > {tree['threshold']:.2f}")
    print_tree(tree["right"], indent + "    ")


def run_c45():
    iris = load_iris()
    features = ["sepal_len", "sepal_wid", "petal_len", "petal_wid"]
    df = pd.DataFrame(iris.data, columns=features)
    df["species"] = [iris.target_names[i] for i in iris.target]

    train, test = train_test_split(df, test_size=0.3, random_state=42)

    tree = build_tree(train, features, "species")

    predictions = test.apply(lambda row: predict_one(tree, row), axis=1)
    accuracy = (predictions == test["species"]).mean()

    print("=== C4.5 ===")
    print("Accuracy:", round(accuracy, 4))
    print("\nTree:")
    print_tree(tree)