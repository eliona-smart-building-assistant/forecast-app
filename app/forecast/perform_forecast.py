from datetime import datetime, timedelta
import pytz
import os

import numpy as np

from api.models import AssetModel, ForecastStatus
from app.get_data.api_calls import (
    get_training_status,
    load_latest_timestamp,
    save_latest_timestamp,
    load_scaler,
    loadState,
    saveState,
    set_forecast_status,
)
from app.get_data.fetch_and_format_data import (
    fetch_pandas_data,
    prepare_data_for_forecast,
)
from app.data_to_eliona.write_into_eliona import write_into_eliona
from tensorflow.keras.models import load_model # type: ignore
import logging
from filelock import FileLock, Timeout
from dateutil import parser

logger = logging.getLogger(__name__)
timestamp_diff_buffer = timedelta(days=5)

def perform_forecast(asset: AssetModel, asset_id):
    tz = pytz.timezone("Europe/Berlin")
    model_filename = (
        f"/tmp/LSTM_model_{asset_id}_{asset.target_attribute}_{asset.forecast_length}.keras"
    )
    batch_size = 1  # For stateful prediction
    global timestamp_diff_buffer

    if os.path.exists(model_filename):
        lock_file = model_filename + ".lock"
        lock = FileLock(lock_file, timeout=1e6)
        try:
            with lock:
                # Load the trained model
                model = load_model(model_filename)
                loadState(model, asset)
                scaler = load_scaler(asset)

                timestep_in_file = load_latest_timestamp(asset)
                timestep_in_file = parser.parse(timestep_in_file)

                new_end_date = datetime.now(tz)
                new_start_date = (
                    timestep_in_file - timestamp_diff_buffer * 10
                ).astimezone(tz)
                set_forecast_status(asset, ForecastStatus.FETCHING)
                df = fetch_pandas_data(
                    asset_id,
                    new_start_date,
                    new_end_date,
                    asset.target_attribute,
                    asset.feature_attributes,
                )

                if df.empty:
                    logger.info(f"No data fetched for {asset_id}, skipping iteration.")
                    return
                set_forecast_status(asset, ForecastStatus.PREPARING)
                X_update, X_last, new_next_timestamp, last_y_timestamp = (
                    prepare_data_for_forecast(
                        asset,
                        df,
                        scaler,
                        timestep_in_file,
                    )
                )
                if new_next_timestamp is None:
                    logger.info(f"new_next_timestamp: {new_next_timestamp} last_y_timestamp: {last_y_timestamp}")
                    return
                timestamp_diff_buffer = (
                    new_next_timestamp - last_y_timestamp
                ) * asset.context_length

                if X_update is None and X_last is None:
                    logger.info(f"No new X sequences to process for {asset_id}. Skipping...")
                    return

                set_forecast_status(asset, ForecastStatus.PREDICTING)

                # Predict sequentially without resetting states
                if len(X_update) > 0:
                    model.summary()
                    for i in range(len(X_update)):
                        x = X_update[i].reshape(
                            (1, asset.context_length, X_update.shape[2])
                        )
                        _ = model.predict(x, batch_size=batch_size)

                if X_last is not None:
                    next_prediction_scaled = model.predict(
                        X_last, batch_size=batch_size
                    )
                    next_prediction_scaled = np.clip(next_prediction_scaled, 0, 1)
                    if asset.parameters.binary_encoding:
                        predicted_class = next_prediction_scaled
                        predicted_class_label = (
                            1 if predicted_class[0][0] > 0.5 else 0
                        )
                        formatted_prediction = predicted_class_label
                    if asset.parameters.num_classes:
                        predicted_class = np.argmax(
                            next_prediction_scaled, axis=1
                        )
                        formatted_prediction = predicted_class[0]
                    else:
                        next_prediction = scaler[asset.target_attribute].inverse_transform(
                            next_prediction_scaled
                        )
                        formatted_prediction = next_prediction[0][0]
                    set_forecast_status(asset, ForecastStatus.WRITING)
                    write_into_eliona(
                        asset_id,
                        new_next_timestamp,
                        formatted_prediction,
                        asset.target_attribute,
                        asset.forecast_length,
                    )

                else:
                    logger.info(f"X_last is None for {asset_id}. Skipping forecasting.")

                training_status = get_training_status(asset)
                if "Saving" in training_status:
                    logger.info(f"Model {model_filename} is currently being saved in training. dont change status")
                else:
                    set_forecast_status(asset, ForecastStatus.SAVING)
                    save_latest_timestamp(last_y_timestamp, tz, asset)
                    model.save(model_filename)
                    saveState(model, asset)
        except Timeout:
            logger.error("Timeout occurred while trying to acquire the file lock.")
            return
    else:
        logger.info(f"Model {model_filename} does not exist. Skipping iteration.")