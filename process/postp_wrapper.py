from pandas import DataFrame as pd_DataFrame
from process.postp.postp import average as postp_average
from process.postp.postp import between as postp_between


def run_postp_wrapper(
    target_cols: list,
    deps_cols: list,
    data: dict,
    input_df: pd_DataFrame,
    postp_algorithm: str or None,
):
    """
    Combines the combined DataFrame with the population data using a specified
    post-processing algorithm if the target columns already exist in the population data.
    """

    postp_algorithm_map = {"average": postp_average, "between": postp_between}

    for col in target_cols:

        proc_data = data["pop"].combine_first(input_df[deps_cols + [col]])

        if col in data["pop"].columns:
            proc_data = postp_algorithm_map[postp_algorithm[col]](
                proc_data, input_df, target_col=col
            )

        data["pop"] = proc_data

    return data


# df_combined = proc_data.combine_first(input_df[deps_cols + [col]])

"""
df_combined = df.combine_first(df2)

# For column X: take mean if both exist, else whichever is available
df_combined["X"] = df["X"].combine(df2["X"], lambda s1, s2: (s1 + s2) / 2)
"""
