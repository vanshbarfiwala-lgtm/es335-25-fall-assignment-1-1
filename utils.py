"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    encoded_df = pd.DataFrame()

    for col in X.columns:
        unique_values = sorted(X[col].unique())

        for val in unique_values:
            encoded_df[str(col) + "_" + str(val)] = (X[col] == val).astype(int)

    return encoded_df

""""
def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:

    Convert all categorical columns in X to one-hot encoded columns.
    Preserves the original DataFrame index.

    Parameters:
        X : pd.DataFrame - input features (categorical)

    Returns:
        pd.DataFrame - one-hot encoded DataFrame

    encoded_df = pd.DataFrame(index=X.index)  # preserve index

    FINALCODEFROMHERE
    
    for col in X.columns:
        unique_values = sorted(X[col].unique())
        for val in unique_values:
            encoded_df[f"{col}_{val}"] = (X[col] == val).astype(int)

    return encoded_df
    TILLHEREIS
    """

def check_ifreal(y: pd.Series) -> bool:
    if(y.dtype == "float64" or y.dtype =="float32"):
        return True
    else:
        return False



def entropy(Y: pd.Series) -> float:
    values = Y.unique()
    entropy_val = 0.0
    for v in values:
        p = (Y == v).sum() / len(Y)
        if p > 0:
            entropy_val -= p * np.log2(p)
    return entropy_val


def gini_index(Y: pd.Series) -> float:
    if len(Y) == 0:
        return 0.0
    probs = Y.value_counts(normalize=True)
    return 1 - np.sum(probs**2)

def variance(y: pd.Series) -> float:
    if len(y) == 0:
        return 0.0
    return float(np.var(y))

def variance_reduction(y: pd.Series, attr: pd.Series) -> float:
    total_var = variance(y)
    total = len(y)
    weighted_child = 0.0

    for val, idx in attr.groupby(attr, observed=False).groups.items():

        y_subset = y.loc[idx]
        weighted_child += (len(y_subset) / total) * variance(y_subset)

    return total_var - weighted_child



def information_gain(Y: pd.Series, attr: pd.Series, criterion: str, threshold: Optional[float] = None
) -> float:
    """
    Compute the information gain (for classification) or variance reduction (for regression)
    for a given attribute. Handles discrete or continuous input attributes.

    Parameters:
        Y        : pd.Series, target/output
        attr     : pd.Series, input attribute
        criterion: "entropy" or "gini" (only for classification)
        threshold: float or None. If given, splits continuous attribute at this threshold

    Returns:
        float: information gain (classification) or variance reduction (regression)
    """
    # Determine if output is real (regression) or discrete (classification)
    is_regression = check_ifreal(Y)
    
    if threshold is not None:
        # Continuous input: split using threshold
        left_mask = attr <= threshold
        right_mask = attr > threshold
        Y_left, Y_right = Y[left_mask], Y[right_mask]
        
        if is_regression:
            total_var = variance(Y)
            weighted_var = (len(Y_left)/len(Y))*variance(Y_left) + (len(Y_right)/len(Y))*variance(Y_right)
            gain = total_var - weighted_var
        else:
            total_impurity = entropy(Y) if criterion == "entropy" else gini_index(Y)
            weighted_impurity = (len(Y_left)/len(Y))*(entropy(Y_left) if criterion=="entropy" else gini_index(Y_left)) + \
                                (len(Y_right)/len(Y))*(entropy(Y_right) if criterion=="entropy" else gini_index(Y_right))
            gain = total_impurity - weighted_impurity
        return gain
    
    else:
        # Discrete input: split by unique values
        if is_regression:
            # variance reduction for regression
            total_var = variance(Y)
            total = len(Y)
            weighted_child = 0.0
            for val, idx in attr.groupby(attr, observed=False).groups.items():
                Y_subset = Y.loc[idx]
                weighted_child += (len(Y_subset)/total)*variance(Y_subset)
            gain = total_var - weighted_child
        else:
            # information gain for classification
            if criterion not in ["entropy", "gini"]:
                raise ValueError("Invalid criterion")
            total = len(Y)
            parent_impurity = entropy(Y) if criterion=="entropy" else gini_index(Y)
            weighted_child = 0
            for val, idx in attr.groupby(attr, observed=False).groups.items():
                Y_subset = Y.loc[idx]
                weighted_child += (len(Y_subset)/total)*(entropy(Y_subset) if criterion=="entropy" else gini_index(Y_subset))
            gain = parent_impurity - weighted_child
        
        return gain



def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion: str):
    """
    Find the best attribute (and threshold if real) for splitting.
    Assumes either all features are discrete or all are real.
    
    Returns:
        best_feature   : str, column name to split
        best_threshold : float or None, threshold for real inputs
        best_gain      : float, information gain or variance reduction
    """
    best_feature, best_threshold, best_gain = None, None, -1

    # Check if inputs are real
    is_real = check_ifreal(X.iloc[:, 0])

    for col in X.columns:
        x_col = X[col]

        if is_real:
            # Continuous attribute → try candidate thresholds
            unique_vals = np.sort(x_col.unique())
            if len(unique_vals) <= 1:
                continue
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2

            for t in thresholds:
                gain = information_gain(y, x_col, criterion, threshold=t)
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, col, t

        else:
            # Discrete attribute
            gain = information_gain(y, x_col, criterion)
            if gain > best_gain:
                best_gain, best_feature, best_threshold = gain, col, None

    return best_feature, best_threshold, best_gain





def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    attribute: str,
    threshold: Optional[float] = None
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Split data according to an attribute.

    For discrete attributes:
        Returns a dict where keys are unique values and values are (X_subset, y_subset)
    For continuous attributes:
        Returns a dict with keys '<=threshold' and '>threshold'

    Parameters:
        X         : pd.DataFrame, input features
        y         : pd.Series, target
        attribute : str, column to split
        threshold : float or None, threshold for continuous attribute

    Returns:
        splits : dict mapping split label to (X_subset, y_subset)
    """
    splits = {}

    if threshold is None:
        # Discrete attribute
        for val in X[attribute].unique():
            mask = X[attribute] == val
            X_sub = X[mask].drop(columns=[attribute])
            y_sub = y[mask]
            splits[val] = (X_sub, y_sub)
    else:
        # Continuous attribute
        left_mask = X[attribute] <= threshold
        right_mask = X[attribute] > threshold

        splits[f"<= {threshold}"] = (X[left_mask].drop(columns=[attribute]), y[left_mask])
        splits[f"> {threshold}"] = (X[right_mask].drop(columns=[attribute]), y[right_mask])

    return splits 