from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score


def run_cart():
    iris = load_iris()
    X, y = iris.data, iris.target

    # 70% for training, 30% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # criterion="gini" is what makes this CART
    model = DecisionTreeClassifier(criterion="gini", random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("=== CART ===")
    print("Accuracy:", round(accuracy, 4))
    print("\nTree rules:")
    print(export_text(model, feature_names=iris.feature_names))

    return accuracy


if __name__ == "__main__":
    run_cart()
