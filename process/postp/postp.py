from pandas import DataFrame as pd_DataFrame
from pandas import to_numeric as pd_to_numeric
from numpy import nan as np_nan
from numpy import where as np_where
from pandas import Series as pd_Series
from numpy import inf as np_inf


def average(df1: pd_DataFrame, df2: pd_DataFrame, target_col: str) -> pd_DataFrame:
    """
    Averages the values of two DataFrames based on a common column.
    Args:
        df1 (pd.DataFrame): The first DataFrame.
        df2 (pd.DataFrame): The second DataFrame.
        target_col (str): The column name to join on.
    Returns:
        pd.DataFrame: A new DataFrame with averaged values.
    """
    df1[target_col] = pd_to_numeric(df1[target_col]).combine(
        pd_to_numeric(df2[target_col]), lambda s1, s2: (s1 + s2) / 2
    )

    return df1


def between(df1: pd_DataFrame, df2: pd_DataFrame, target_col: str) -> pd_DataFrame:
    """
    Sets the values of the first DataFrame to be between the values of the second DataFrame.
    Args:
        df1 (pd.DataFrame): The first DataFrame.
        df2 (pd.DataFrame): The second DataFrame.
        target_col (str): The column name to join on.
    Returns:
        pd.DataFrame: A new DataFrame with adjusted values.
    """

    def _sort_string(s):
        items = [x.strip() for x in s.split("-")]
        items = sorted(items)
        return "-".join(items)

    df1[target_col] = df1[target_col].astype(str) + "-" + df2[target_col].astype(str)

    df1[target_col] = df1[target_col].apply(_sort_string)

    df1[target_col] = df1[target_col].replace("nan-nan", np_nan)

    return df1
