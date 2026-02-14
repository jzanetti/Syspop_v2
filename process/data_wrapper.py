from process.data.utils import obtain_api_key
from process.data.data import obtain_data
from logging import info as log_info


def obtain_data_wrapper(
    cfg_data, api_key: str or None = None, data_types: list = ["pop"]
):
    """
    Wrapper function to retrieve data for specified types using an API key.

        cfg_data (dict): Dictionary containing configuration for each data type,
            where keys are data type strings and values are configuration dicts.
        api_key (str or None, optional): API key to use for data retrieval. If None,
            the key will be obtained via `obtain_api_key`. Defaults to None.
        data_types (list, optional): List of data type strings to retrieve.
            Defaults to ["pop"].

        dict: A dictionary mapping each data type (from `data_types`) to its
            corresponding data retrieved using the API.

    Raises:
        KeyError: If a specified data type is not present in `cfg_data`.
        Exception: Propagates exceptions raised during API key retrieval or data fetching.

    Example:
        >>> cfg_data = {"pop": {...}, "income": {...}}
        >>> result = obtain_data_wrapper(cfg_data, api_key="my_key", data_types=["pop", "income"])
        >>> print(result)
        {'pop': <pop_data>, 'income': <income_data>}
    """
    api_key = obtain_api_key(api_key)

    data_dict = {}
    for data_type in data_types:

        log_info(f"Obtaining data for type: {data_type}")

        data_dict[data_type] = obtain_data(cfg_data[data_type], api_key)

    return data_dict
