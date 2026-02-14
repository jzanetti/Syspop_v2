import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np
from process.data.data import prepare_model_data


def model_train(
    df: DataFrame,
    deps_cols: list,
    target_cols: list,
    test_size: float = 0.2,
    random_state: None or int = None,
):
    """
    Train an XGBoost model with native categorical support to predict the probability
    of each (target_cols) combination given categorical features (deps_cols).

    Parameters
    ----------
    df : DataFrame
        Input data containing both predictor and target columns.
    deps_cols : list
        List of predictor (feature) column names.
    target_cols : list
        List of columns to combine into the target variable.
    test_size : float
        Fraction of data to use for testing.
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    probs_df : DataFrame
        Predicted probabilities for each target combination, with most likely label and probability.
    model : XGBClassifier
        Trained XGBoost model.
    target_encoder : LabelEncoder
        Encoder for decoding target labels.
    """

    model_data = prepare_model_data(df, deps_cols, target_cols=target_cols)

    # --- Split data ---
    X_train, X_test, y_train, y_test = train_test_split(
        model_data["X"], model_data["y"], test_size=test_size, random_state=random_state
    )

    # --- Train model with native categorical support ---
    model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",  # required for categorical support
        enable_categorical=True,  # enables native handling of categories
        random_state=random_state,
        use_label_encoder=False,
    )

    model.fit(X_train[deps_cols], y_train)

    # --- Get data range for training ---
    data_range = {}
    for col in deps_cols:
        data_range[col] = list(model_data["X"][col].cat.categories)

    return {
        "model": model,
        "data_range": data_range,
        "target_encoder": model_data["target_encoder"],
        "test_data": {"x": X_test, "y": y_test},
        "deps_cols": deps_cols,
        "target_cols": target_cols,
    }


def model_predict(model, target_encoder, x_data):
    # --- 1. Predict probabilities ---
    probs = model.predict_proba(x_data)
    class_labels = target_encoder.classes_

    # --- 2. Build output dataframe ---
    probs_df = pd.DataFrame(probs, columns=class_labels, index=x_data.index)

    probs_df = pd.concat([x_data, probs_df], axis=1)
    # probs_df = pd.concat(
    #    [x_data.reset_index(drop=True), probs_df.reset_index(drop=True)], axis=1
    # )
    return probs_df


def model_sample(df_probs, deps_cols, target_cols, seed=None):
    """
    Vectorized sampling of industry/work_status per row based on probabilities.

    Parameters
    ----------
    df_probs : pd.DataFrame
        Wide format: feature columns + probability columns like A_1, A_2, etc.
        Must also contain 'most_likely' and 'max_prob' columns (ignored here).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: age, gender, location, industry, work_status
    """
    if seed is not None:
        np.random.seed(seed)

    prob_cols = [c for c in df_probs.columns if c not in deps_cols]

    # Extract probabilities as NumPy array
    probs_array = df_probs[prob_cols].values.astype(float)
    probs_array = probs_array / probs_array.sum(axis=1, keepdims=True)  # normalize

    # Number of rows
    n_rows = df_probs.shape[0]

    # Flatten probabilities for sampling
    sampled_cols = []
    for i in range(n_rows):
        sampled = np.random.choice(prob_cols, size=1, p=probs_array[i])
        sampled_cols.extend(sampled)

    sampled_cols = [s.split("_") for s in sampled_cols]
    sampled_cols = np.array(sampled_cols)

    # Combine features and sampled
    df_long = pd.DataFrame(
        np.hstack([df_probs[deps_cols].values, sampled_cols]),
        columns=deps_cols + target_cols,
        index=df_probs.index,
    )
    return df_long
