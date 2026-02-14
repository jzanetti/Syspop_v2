from pandas import to_numeric
from pandas import DataFrame
from etc.api_keys import STATS_API


def stats_data_proc(data: DataFrame, cfg: dict):

    data = data.rename(columns=cfg["map"])

    data["value"] = to_numeric(data["value"], errors="coerce")
    data = data.dropna()
    data["value"] = data["value"].astype(int)

    return data


def obtain_api_key(api_key: str or None):
    if api_key is None:
        api_key = STATS_API

    return api_key


def create_data_info(data_dict: dict, output_dir: str or None = None):
    rows = []

    for key, df in data_dict.items():
        for col in df.columns:
            unique_list = df[col].unique()
            unique_str = ", ".join(map(str, unique_list))
            unique_count = len(unique_list)
            rows.append(
                {
                    "data_key": key,
                    "cols": col,
                    "unique_count": unique_count,
                    "unique_value": unique_str,
                }
            )

    summary_df = DataFrame(rows)
    if output_dir is not None:
        summary_df.to_csv(f"{output_dir}/data_info.csv", index=False)
