from process.data.query import obtain_stats_data
from process.data.utils import stats_data_proc
from pandas import DataFrame as pdDataFrame
from sklearn.preprocessing import LabelEncoder


def obtain_data(cfg: dict, api_key: str):
    """
    Obtains and processes population statistics data based on the provided configuration and API key.

    The function performs the following steps:
    1. Fetches raw statistics data using the API configuration and key.
    2. Processes the raw data according to the configuration.
    3. Applies inclusion and exclusion filters to the data based on specified criteria.
    4. Maps the filtered data to the desired columns.
    5. Expands the DataFrame by repeating rows according to the 'value' column.
    6. Returns the final DataFrame with the 'value' column removed and index reset.

    Args:
        cfg (dict): Configuration dictionary containing API settings, mapping of column names,
            inclusion and exclusion criteria.
        api_key (str): API key for accessing the statistics data.

    Returns:
        pandas.DataFrame: Processed and filtered DataFrame with population statistics.
    """

    def _obtain_qc_key(cfg, proc_qc_type):
        if proc_qc_type in cfg["map"]:
            return cfg["map"][proc_qc_type]
        return proc_qc_type

    data_pop = obtain_stats_data(cfg["api"], api_key=api_key)
    data_pop = stats_data_proc(data_pop, cfg)

    try:
        for proc_qc_type in cfg["inclusion"]:
            proc_qc_key = _obtain_qc_key(cfg, proc_qc_type)
            data_pop = data_pop[
                data_pop[proc_qc_key].isin(cfg["inclusion"][proc_qc_type])
            ]
    except TypeError:
        pass

    try:
        for proc_qc_type in cfg["exclusion"]:
            proc_qc_key = _obtain_qc_key(cfg, proc_qc_type)
            data_pop = data_pop[
                ~data_pop[proc_qc_key].isin(cfg["exclusion"][proc_qc_type])
            ]
    except TypeError:
        pass

    df = data_pop[list(cfg["map"].values())]
    df = df.loc[df.index.repeat(df["value"])].copy()

    return df.reset_index(drop=True).drop(columns=["value"])


def prepare_model_data(
    df: pdDataFrame, deps_cols: list, target_cols: list or None = None
):
    """
    Prepares data for modeling by encoding target columns and converting dependent columns to categorical dtype.
    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing the data.
    deps_cols : list
        List of column names to be used as dependent (feature) variables.
    target_cols : list or None, optional
        List of column names to be used as target variables. If None, target encoding is skipped.
    Returns
    -------
    dict
        A dictionary containing:
            - 'X': pd.DataFrame of dependent columns with categorical dtype.
            - 'y': pd.Series of encoded target values (if target_cols is provided), otherwise None.
            - 'target_encoder': LabelEncoder instance used for encoding targets (if target_cols is provided), otherwise None.
    Notes
    -----
    If multiple target columns are provided, their string representations are concatenated with underscores before encoding.
    Dependent columns are converted to categorical dtype for modeling purposes.
    """
    y = None
    target_encoder = None
    if target_cols is not None:
        df["target"] = df[target_cols].astype(str).agg("_".join, axis=1)
        target_encoder = LabelEncoder()
        df["target_encoded"] = target_encoder.fit_transform(df["target"])
        y = df["target_encoded"]

    # --- Convert dependent columns to categorical dtype ---
    for col in deps_cols:
        df[col] = df[col].astype("category")

    X = df[deps_cols]

    return {"X": X, "y": y, "target_encoder": target_encoder}
