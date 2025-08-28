from dataclasses import dataclass
from typing import Literal, Any, Optional
import pandas as pd
import numpy as np
from utils import opt_split_attribute, split_data, check_ifreal

class Node:
    def __init__(self,
                 feature: Optional[str] = None,
                 threshold: Optional[float] = None,
                 children: Optional[dict] = None,
                 value: Optional[Any] = None):
        self.feature = feature
        self.threshold = threshold
        self.children = children  # dict: split_label -> Node
        self.value = value        # leaf prediction

@dataclass
class DecisionTree:
    criterion: Literal["entropy", "gini"]  # For classification only
    max_depth: int = 5

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None
        self.is_regression = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.is_regression = check_ifreal(y)
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int) -> Node:
        # Stop if pure or max depth reached
        if len(y.unique()) == 1 or depth >= self.max_depth or X.empty:
            return Node(value=self._leaf_value(y))

        # Find best split
        features = X.columns
        best_feature, best_threshold, best_gain = opt_split_attribute(X, y, self.criterion)

        if best_feature is None or best_gain <= 0:
            return Node(value=self._leaf_value(y))

        # Split and recurse
        splits = split_data(X, y, best_feature, best_threshold)
        children = {}
        for split_label, (X_sub, y_sub) in splits.items():
            children[split_label] = self._build_tree(X_sub, y_sub, depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, children=children)

    def _leaf_value(self, y: pd.Series) -> Any:
        if self.is_regression:
            return y.mean()
        else:
            return y.value_counts().idxmax()

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X.apply(lambda row: self._predict_one(row, self.root), axis=1)

    def _predict_one(self, row: pd.Series, node: Node):
        if node.value is not None:
            return node.value
        if node.threshold is not None:
            # Continuous feature
            if row[node.feature] <= node.threshold:
                return self._predict_one(row, node.children[f"<= {node.threshold}"])
            else:
                return self._predict_one(row, node.children[f"> {node.threshold}"])
        else:
            # Discrete feature
            val = row[node.feature]
            if val in node.children:
                return self._predict_one(row, node.children[val])
            else:
                return None  # unseen category

    def plot(self) -> None:
        self._plot_node(self.root, depth=0)

    def _plot_node(self, node: Node, depth: int):
        indent = "    " * depth
        if node.value is not None:
            print(f"{indent}Predict -> {node.value}")
        elif node.threshold is not None:
            print(f"{indent}?( {node.feature} <= {node.threshold} )")
            print(f"{indent}Y:")
            self._plot_node(node.children[f"<= {node.threshold}"], depth + 1)
            print(f"{indent}N:")
            self._plot_node(node.children[f"> {node.threshold}"], depth + 1)
        else:
            print(f"{indent}Split on {node.feature}")
            for val, child in node.children.items():
                print(f"{indent}{val}:")
                self._plot_node(child, depth + 1)