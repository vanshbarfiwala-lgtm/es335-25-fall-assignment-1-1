from typing import Union
import pandas as pd
import numpy as np

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    correct_values = (y_hat == y)
    accuracy = correct_values.sum()/len(y_hat)
    return accuracy




def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    true_positive = ((y_hat == cls) & (y == cls)).sum()
    false_positive = ((y_hat == cls) & (y != cls)).sum()
    if( true_positive + false_positive )== 0:
        return 0.0
    precision = true_positive/(true_positive+false_positive)
    return precision


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    true_positive = ((y_hat == cls) & (y == cls)).sum()
    false_negative = ((y_hat != cls) & (y == cls)).sum()
    if( true_positive + false_negative )== 0:
        return 0.0
    recall = true_positive/(true_positive+false_negative)
    return recall


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """

    rmserror = np.sqrt(((y_hat-y)**2).mean())
    return rmserror


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    maerror = abs(y_hat-y).mean()
    return maerror