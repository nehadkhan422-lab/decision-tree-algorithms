from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
iris = load_iris()

# Put it into a table (DataFrame)
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = [iris.target_names[i] for i in iris.target]

# Look at the data
print("First 5 rows:")
print(df.head())

print("\nShape (rows, columns):", df.shape)

print("\nHow many flowers of each species:")
print(df["species"].value_counts())