from process.model.model import model_train, model_predict, model_sample
from pandas import DataFrame
from pickle import dump as pickle_dump
from pickle import load as pickle_load
from process import TMP_DIR
from process.model import MODEL_TRAINING_TEST_RATIO
from process.model.utils import create_input_data
from process.data.data import prepare_model_data
from pandas import concat as pd_concat
from logging import info as log_info
from process.postp_wrapper import run_postp_wrapper


def run_model_train_wrapper(data: dict, cfg: dict, data_types: list = []):
    for data_type in data_types:
        log_info(f"Training model for data type: {data_type}")
        model = model_train(
            data[data_type],
            deps_cols=cfg[data_type]["features"],
            target_cols=cfg[data_type]["targets"],
            test_size=MODEL_TRAINING_TEST_RATIO,
        )
        pickle_dump(model, open(f"{TMP_DIR}/model_{data_type}.p", "wb"))


def run_model_pred_wrapper(
    data: dict,
    data_types: list = [],
    postp_algorithm: str or None = None,
    output_dir: str = "",
):
    """
    Runs prediction models for specified data types using provided population data.
    This function loads pre-trained models for each data type, prepares the input data,
    performs predictions, samples the predictions, and combines the results with the original data.
    The final combined DataFrame for each data type is saved as a pickle file.
    Args:
        data (dict): Dictionary containing input data. Must include a "pop" key for base population data.
        data_types (list, optional): List of data type strings for which predictions are to be made.
        postp_algorithm (str or None, optional): Post-processing algorithm to combine results if target columns already exist. Defaults to None.
        output_dir (str, optional): Directory to save the final results parquet file. Defaults to "".
    Raises:
        ValueError: If "pop" key is not present in the input data dictionary.
    Side Effects:
        Saves combined prediction results for each data type as pickle files in TMP_DIR.
    """

    if "pop" not in data:
        raise ValueError("Base population data is required for prediction")

    for data_type in data_types:

        log_info(f"Running prediction for data type: {data_type}")

        model = pickle_load(open(f"{TMP_DIR}/model_{data_type}.p", "rb"))

        proc_data = create_input_data(
            data["pop"], model["deps_cols"], model["data_range"]
        )

        proc_data_input = prepare_model_data(
            proc_data["data_in_range"], model["deps_cols"]
        )

        pred = model_predict(
            model["model"], model["target_encoder"], proc_data_input["X"]
        )

        pred = model_sample(pred, model["deps_cols"], model["target_cols"])

        df_combined = pd_concat(
            [
                proc_data["data_in_range"].combine_first(pred),
                proc_data["data_out_range"],
            ],
            ignore_index=False,
        )

        data = run_postp_wrapper(
            model["target_cols"],
            model["deps_cols"],
            data,
            df_combined,
            postp_algorithm,
        )

        pickle_dump(data[data_type], open(f"{TMP_DIR}/results_{data_type}.p", "wb"))

    data["pop"].astype("category").to_parquet(f"{output_dir}/results.parquet")
