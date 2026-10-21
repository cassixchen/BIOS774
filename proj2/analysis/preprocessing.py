import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess_data(
    train,
    eval_data=None,
    method="standard",
    target_sum=10000.0,
):
    if method == "none":
        train_processed = train.copy()

        if eval_data is not None:
            eval_processed = eval_data.copy()
        else:
            eval_processed = None

    elif method == "standard":
        scaler = StandardScaler()

        train_processed = scaler.fit_transform(train)

        if eval_data is not None:
            eval_processed = scaler.transform(eval_data)
        else:
            eval_processed = None

    elif method == "log_normalize":
        train_processed = log_normalize(
            train,
            target_sum=target_sum,
        )

        if eval_data is not None:
            eval_processed = log_normalize(
                eval_data,
                target_sum=target_sum,
            )
        else:
            eval_processed = None

    else:
        raise ValueError(
            "Preprocessing method must be "
            "'none', 'standard', or 'log_normalize'."
        )

    return train_processed, eval_processed


def log_normalize(data, target_sum=10000.0):

    library_sizes = data.sum(axis=1)

    normalized = (
        data
        / library_sizes[:, np.newaxis]
        * target_sum
    )

    return np.log1p(normalized)