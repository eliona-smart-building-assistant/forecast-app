import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from sklearn.model_selection import train_test_split
from api.models import AssetModel, TrainingStatus
from app.get_data.fetch_and_format_data import prepare_data
from app.get_data.api_calls import (
    save_scaler,
    save_parameters,
    set_training_status
)
from kerastuner.tuners import BayesianOptimization
from app.train_and_retrain.train.lstmHyperModel import LSTMHyperModel
from .build_standard_lstm import build_lstm_model
from .callbacks import (
    CustomCallback,
    HyperModelCheckpointCallback,
    CustomBayesianOptimization,
)
import logging

logger = logging.getLogger(__name__)


def train_lstm_model(
    asset: AssetModel,
    asset_id,
    data,
    tz,
    model_save_path,
):
    batch_size = 1

    project_name = f"hyperparameters_model_{asset_id}_{asset.target_attribute}_{asset.forecast_length}"

    logger.info("Training Parameters:")
    logger.info(f"  Epochs: {asset.trainingparameters.epochs}")
    logger.info(f"  Patience: {asset.trainingparameters.patience}")
    logger.info(f"  Validation Split: {asset.trainingparameters.validation_split}")

    X, y, scaler, last_timestamp = prepare_data(asset, data)

    save_scaler(scaler, asset)

    logger.info(f"X tail training: {X[-3:]}")
    logger.info(f"y tail training: {y[-3:]}")
    logger.info(f"X shape: {X.shape}")
    logger.info(f"y shape: {y.shape}")
    num_features = X.shape[2]

    # Commented out: hyperparameter search branch
    """
    if asset.hyperparameters or ((not asset.hyperparameters) and (not asset.parameters)):
        set_training_status(asset, TrainingStatus.START_HYPER_TRAINING)
        data_length = int(len(X) * asset.hyperparameters.percent_data)
        X_hyper = X[:data_length]
        y_hyper = y[:data_length]
        validation_samples = int(len(X_hyper) * asset.trainingparameters.validation_split)
        if validation_samples == 0:
            logger.info("Validation split results in 0 validation samples. Skipping training.")
            return
        X_train_hyper, X_val_hyper, y_train_hyper, y_val_hyper = train_test_split(
            X_hyper, y_hyper, test_size=asset.trainingparameters.validation_split, shuffle=False
        )
        # Hyperparameter optimization
        hypermodel = LSTMHyperModel(
            asset.context_length,
            num_features,
            parameters=asset.parameters,
            hyperparameters=asset.hyperparameters,
        )
        tuner = CustomBayesianOptimization(
            hypermodel,
            objective=asset.trainingparameters.objective,
            max_trials=asset.hyperparameters.max_trials,
            directory="/tmp/hyperparameter_search",
            project_name=project_name,
            max_retries_per_trial=0,  # Prevent retries
            max_consecutive_failed_trials=100,
            model_save_path=model_save_path,
            asset_details=asset,
            tz=tz,
            latest_timestamp=last_timestamp,
        )

        tuner.search_space_summary()
        hypermodel_checkpoint_callback = HyperModelCheckpointCallback(
            stateful=asset.parameters.stateful
        )

        tuner.search(
            X_train_hyper,
            y_train_hyper,
            epochs=asset.trainingparameters.epochs,
            batch_size=batch_size,
            validation_data=(X_val_hyper, y_val_hyper),
            callbacks=[
                EarlyStopping(
                    monitor=asset.trainingparameters.objective,
                    patience=asset.trainingparameters.patience,
                    restore_best_weights=True,
                ),
                TerminateOnNaN(),
                hypermodel_checkpoint_callback,
            ],
        )

        tuner.results_summary()

        # Calculate the number of top models (5% of total trials)
        total_trials = len(tuner.oracle.trials)
        best_trails_percent = asset.hyperparameters.get("best_trails_percent", 0.05)
        top_n = max(1, int(total_trials * best_trails_percent))

        logger.info(f"Selecting the top {top_n} models out of {total_trials} total trials.")

        # Get the top N hyperparameters and models
        best_hyperparameters_list = tuner.get_best_hyperparameters(num_trials=top_n)
        best_models = tuner.get_best_models(num_models=top_n)

        # Prepare new data to test on (next data_length of data)
        data_length = int(len(X) * asset.hyperparameters.percent_data)
        test_data_start = data_length
        test_data_end = data_length * 2

        if len(X) >= test_data_end:
            X_test_hyper = X[test_data_start:test_data_end]
            y_test_hyper = y[test_data_start:test_data_end]
        else:
            # If not enough data, use the remaining data
            X_test_hyper = X[test_data_start:]
            y_test_hyper = y[test_data_start:]

        logger.info(
            f"Testing top models on new data from index {test_data_start} to {test_data_end}"
        )

        # Evaluate each model on the new data
        best_score = None
        best_hyperparameters = None

        for i, (model, hyperparams) in enumerate(
            zip(best_models, best_hyperparameters_list)
        ):
            # Evaluate the model
            evaluation = model.evaluate(
                X_test_hyper, y_test_hyper, verbose=0, batch_size=batch_size
            )
            # Assuming the first element is the loss
            loss = evaluation[0] if isinstance(evaluation, (list, tuple)) else evaluation

            logger.info(f"Model {i+1} evaluation on new data: Loss = {loss}")

            # Select the model with the best performance
            if best_score is None or loss < best_score:
                best_score = loss
                best_hyperparameters = hyperparams

        logger.info("Best hyperparameters after testing on new data:")
        logger.info(f"{best_hyperparameters}")
        logger.info("Best hyperparameters values:")
        logger.info(f"{best_hyperparameters.values}")
        validation_samples = int(len(X) * asset.trainingparameters.validation_split)
        if validation_samples == 0:
            logger.info("Validation split results in 0 validation samples. Skipping training.")
            return
        # Now, retrain the best model with the full dataset
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=asset.trainingparameters.validation_split, shuffle=False
        )

        # Build the model with the best hyperparameters
        model = build_lstm_model(
            asset.context_length,
            num_features,
            best_hyperparameters.values,
        )

        # Callbacks
        early_stopping = EarlyStopping(
            monitor=asset.trainingparameters.objective,
            patience=asset.trainingparameters.patience,
            restore_best_weights=True
        )
        custom_callback = CustomCallback(
            model_save_path=model_save_path,
            asset_details=asset,
            tz=tz,
            latest_timestamp=last_timestamp,
        )
        save_parameters(best_hyperparameters.values, asset)

        # Train model
        set_training_status(asset, TrainingStatus.START_HYPER_TRAINING)
        model.fit(
            X_train,
            y_train,
            epochs=asset.trainingparameters.epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, custom_callback],
            shuffle=False,
        )

        return model
    """

    # Always perform normal training (hyperparameter search is commented out)
    validation_samples = int(len(X) * asset.trainingparameters.validation_split)
    if validation_samples == 0:
        logger.info("Validation split results in 0 validation samples. Skipping training.")
        return
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=asset.trainingparameters.validation_split, shuffle=False
    )
    model = build_lstm_model(
        asset.context_length,
        num_features,
        asset.parameters,
    )

    # Callbacks
    early_stopping = EarlyStopping(
        monitor=asset.trainingparameters.objective,
        patience=asset.trainingparameters.patience,
        restore_best_weights=True
    )
    custom_callback = CustomCallback(
        model_save_path=model_save_path,
        asset_details=asset,
        tz=tz,
        latest_timestamp=last_timestamp,
    )

    model.fit(
        X_train,
        y_train,
        epochs=asset.trainingparameters.epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, custom_callback, TerminateOnNaN()],
        shuffle=False,
    )

    # Return the trained model
    return model