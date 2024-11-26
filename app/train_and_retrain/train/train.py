import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from app.get_data.fetch_and_format_data import prepare_data
from app.get_data.api_calls import (
    save_scaler,
    save_parameters,
    set_processing_status,
)
from kerastuner.tuners import BayesianOptimization
from app.train_and_retrain.train.lstmHyperModel import LSTMHyperModel
from .build_standard_lstm import build_lstm_model
from .callbacks import (
    CustomCallback,
    HyperModelCheckpointCallback,
    CustomBayesianOptimization,
)


def train_lstm_model(
    asset_details,
    asset_id,
    data,
    SessionLocal,
    Asset,
    tz,
    context_length,
    forecast_length,
    model_save_path,
):
    batch_size = 1
    parameters = asset_details["parameters"] or {}

    trainingparameters = asset_details["trainingparameters"] or {}
    epochs = trainingparameters.get("epochs", 50)
    patience = trainingparameters.get("patience", 5)
    validation_split = trainingparameters.get("validation_split", 0.2)
    objective = trainingparameters.get("objective", "val_loss")

    hyperparameters = asset_details["hyperparameters"] or {}
    hyperparameters_percent_data = hyperparameters.get("percent_data", 0.01)
    max_trials = hyperparameters.get("max_trials", 100)
    project_name = f"hyperparameters_model_{asset_id}_{asset_details['target_attribute']}_{forecast_length}"

    # Prepare data
    print(f"Training Parameters:")
    print(f"  Epochs: {epochs}")
    print(f"  Patience: {patience}")
    print(f"  Validation Split: {validation_split}")

    X, y, scaler, last_timestamp = prepare_data(
        data,
        context_length,
        forecast_length,
        asset_details["target_attribute"],
        asset_details["feature_attributes"],
    )

    save_scaler(SessionLocal, Asset, scaler, asset_details)

    print("X tail training", X[-3:])
    print("y tail training", y[-3:])
    print("X shape", X.shape)
    print("y shape", y.shape)
    # Split the data
    num_features = X.shape[2]
    if hyperparameters or ((not hyperparameters) and (not parameters)):
        set_processing_status(
            SessionLocal, Asset, asset_details, "start_hyperparameter_search_training"
        )
        data_length = int(len(X) * hyperparameters_percent_data)
        X_hyper = X[:data_length]
        y_hyper = y[:data_length]
        validation_samples = int(len(X_hyper) * validation_split)
        if validation_samples == 0:
            print(
                "Validation split results in 0 validation samples. Skipping training."
            )
            return
        X_train_hyper, X_val_hyper, y_train_hyper, y_val_hyper = train_test_split(
            X_hyper, y_hyper, test_size=validation_split, shuffle=False
        )
        # Hyperparameter optimization
        hypermodel = LSTMHyperModel(
            context_length,
            num_features,
            parameters=parameters,
            hyperparameters=hyperparameters,
        )
        tuner = CustomBayesianOptimization(
            hypermodel,
            objective=objective,
            max_trials=max_trials,
            directory="hyperparameter_search",
            project_name=project_name,
            max_retries_per_trial=0,  # Prevent retries
            max_consecutive_failed_trials=50,
            model_save_path=model_save_path,
            SessionLocal=SessionLocal,
            Asset=Asset,
            asset_details=asset_details,
            tz=tz,
            latest_timestamp=last_timestamp,
        )

        tuner.search_space_summary()
        hypermodel_checkpoint_callback = HyperModelCheckpointCallback()

        tuner.search(
            X_train_hyper,
            y_train_hyper,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_hyper, y_val_hyper),
            callbacks=[
                EarlyStopping(
                    monitor=objective,
                    patience=patience,
                    restore_best_weights=True,
                ),
                hypermodel_checkpoint_callback,
            ],
        )

        tuner.results_summary()

        # Calculate the number of top models (5% of total trials)
        total_trials = len(tuner.oracle.trials)
        best_trails_percent = hyperparameters.get("best_trails_percent", 0.05)
        top_n = max(1, int(total_trials * best_trails_percent))

        print(f"Selecting the top {top_n} models out of {total_trials} total trials.")

        # Get the top N hyperparameters and models
        best_hyperparameters_list = tuner.get_best_hyperparameters(num_trials=top_n)
        best_models = tuner.get_best_models(num_models=top_n)

        # Prepare new data to test on (next data_length of data)
        data_length = int(len(X) * hyperparameters_percent_data)
        test_data_start = data_length
        test_data_end = data_length * 2

        if len(X) >= test_data_end:
            X_test_hyper = X[test_data_start:test_data_end]
            y_test_hyper = y[test_data_start:test_data_end]
        else:
            # If not enough data, use the remaining data
            X_test_hyper = X[test_data_start:]
            y_test_hyper = y[test_data_start:]

        print(
            f"Testing top models on new data from index {test_data_start} to {test_data_end}"
        )

        # Evaluate each model on the new data
        best_score = None
        best_hyperparameters = None

        for i, (model, hyperparameters) in enumerate(
            zip(best_models, best_hyperparameters_list)
        ):
            # Evaluate the model
            evaluation = model.evaluate(
                X_test_hyper, y_test_hyper, verbose=0, batch_size=batch_size
            )
            # Assuming the first element is the loss
            loss = (
                evaluation[0] if isinstance(evaluation, (list, tuple)) else evaluation
            )

            print(f"Model {i+1} evaluation on new data: Loss = {loss}")

            # Select the model with the best performance
            if best_score is None or loss < best_score:
                best_score = loss
                best_hyperparameters = hyperparameters

        print("Best hyperparameters after testing on new data:")
        print(best_hyperparameters)
        print("Best hyperparameters values:")
        print(best_hyperparameters.values)
        validation_samples = int(len(X) * validation_split)
        if validation_samples == 0:
            print(
                "Validation split results in 0 validation samples. Skipping training."
            )
            return
        # Now, retrain the best model with the full dataset
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, shuffle=False
        )

        # Build the model with the best hyperparameters
        model = build_lstm_model(
            context_length,
            num_features,
            best_hyperparameters.values,
        )

        # Callbacks
        early_stopping = EarlyStopping(
            monitor=objective, patience=patience, restore_best_weights=True
        )
        custom_callback = CustomCallback(
            model_save_path=model_save_path,
            SessionLocal=SessionLocal,
            Asset=Asset,
            asset_details=asset_details,
            tz=tz,
            latest_timestamp=last_timestamp,
        )
        save_parameters(SessionLocal, Asset, best_hyperparameters.values, asset_details)

        # Train model
        set_processing_status(
            SessionLocal, Asset, asset_details, "start_hyperparameter_training"
        )
        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, custom_callback],
            shuffle=False,
        )

        return model

    else:
        validation_samples = int(len(X) * validation_split)
        if validation_samples == 0:
            print(
                "Validation split results in 0 validation samples. Skipping training."
            )
            return
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, shuffle=False
        )
        model = build_lstm_model(
            context_length,
            num_features,
            parameters,
        )

        # Callbacks
        early_stopping = EarlyStopping(
            monitor=objective, patience=patience, restore_best_weights=True
        )
        custom_callback = CustomCallback(
            model_save_path=model_save_path,
            SessionLocal=SessionLocal,
            Asset=Asset,
            asset_details=asset_details,
            tz=tz,
            latest_timestamp=last_timestamp,
        )

        # Train model
        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, custom_callback],
            shuffle=False,
        )

        # Return the best model and other details
        return model
