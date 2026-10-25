# decision-tree-algorithms
ID3, C4.5 and CART decision trees on a dataset
# Decision Tree Algorithms: ID3, C4.5 and CART

This project implements and compares three decision tree algorithms on the Iris dataset.

## Dataset

The Iris dataset has 150 flowers (50 each of setosa, versicolor and virginica).
There are 4 numeric features: sepal length, sepal width, petal length and petal width.
The goal is to predict the species. The data is split 70% training and 30% testing (random_state=42).

## Algorithms

| Algorithm | Split criterion | Numeric features | Implementation |
|-----------|-----------------|------------------|----------------|
| ID3 | Information gain (entropy) | Converted into 3 groups (low, medium, high) | Written from scratch (id3.py) |
| C4.5 | Gain ratio | Handled directly with thresholds | Written from scratch (c45.py) |
| CART | Gini impurity | Handled directly, binary splits | scikit-learn (cart.py) |

## Results

| Algorithm | Test accuracy |
|-----------|---------------|
| ID3 | 0.9778 |
| C4.5 | 1.0000 |
| CART | 1.0000 |

## Observations

- ID3 scored slightly lower because it needs categories. Cutting each numeric column into only 3 groups loses detail.
- C4.5 and CART both start with the same question (petal length <= 2.45), which separates setosa immediately.
- Iris is small and clean, so results can change with a different random split.

## How to run

1. Install the libraries:
   `python -m pip install -r requirements.txt`
2. Run everything:
   `python main.py`
3. Or run one algorithm: `python id3.py`, `python c45.py` or `python cart.py`

## Files

- `id3.py`: ID3 algorithm
- `c45.py`: C4.5 algorithm
- `cart.py`: CART algorithm
- `main.py`: runs all three and compares accuracy
- `requirements.txt`: required libraries
