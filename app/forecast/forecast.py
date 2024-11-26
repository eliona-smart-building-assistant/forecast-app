from datetime import datetime, timedelta
import pytz
import os
import time

from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker
from app.get_data.api_calls import (
    load_latest_timestamp,
    save_latest_timestamp,
    load_contextlength,
    load_scaler,
    loadState,
    saveState,
    get_processing_status,
    set_processing_status,
)
from app.get_data.fetch_and_format_data import (
    fetch_pandas_data,
    prepare_data_for_forecast,
)
from app.data_to_eliona.write_into_eliona import write_into_eliona
from app.data_to_eliona.create_asset_to_save_models import (
    load_model_from_eliona,
    save_model_to_eliona,
    model_exists,
)
import websocket
import ssl
import threading
import sys
from api.api_calls import get_asset_by_id

last_processed_time = 0  # Initialize the last processed time


def forecast(asset_details, asset_id):

    process_lock = threading.Lock()
    # Initialize database connection and ORM models here
    db_url = os.getenv("CONNECTION_STRING")
    db_url_sql = db_url.replace("postgres", "postgresql")
    DATABASE_URL = db_url_sql
    engine = create_engine(DATABASE_URL)

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    metadata = MetaData()
    Asset = Table(
        "assets_to_forecast", metadata, autoload_with=engine, schema="forecast"
    )

    forecast_length = asset_details["forecast_length"]
    target_column = asset_details["target_attribute"]
    feature_columns = asset_details["feature_attributes"]
    tz = pytz.timezone("Europe/Berlin")
    model_filename = f"LSTM_model_{asset_id}_{target_column}_{forecast_length}.keras"
    batch_size = 1  # Setting batch size to 1 for stateful LSTM

    # Define the perform_forecast function
    def perform_forecast():
        timestamp_diff_buffer = timedelta(days=5)
        if model_exists(model_filename):
            processing_status = get_processing_status(
                SessionLocal, Asset, asset_details
            )
            if "Saving" in processing_status:
                print("Model is currently being saved in training. dont change status")

            elif (
                "training" in processing_status and processing_status != "done_training"
            ):
                set_processing_status(
                    SessionLocal, Asset, asset_details, "forecasting_and_training"
                )
            else:
                set_processing_status(SessionLocal, Asset, asset_details, "forecasting")
            print(f"Loading existing model from {model_filename}")
            model = load_model_from_eliona(model_filename)
            loadState(SessionLocal, Asset, model, asset_details)
            # Load the scaler
            scaler = load_scaler(SessionLocal, Asset, asset_details)

            timestep_in_file = load_latest_timestamp(SessionLocal, Asset, asset_details)
            timestep_in_file = datetime.fromisoformat(timestep_in_file)
            context_length = load_contextlength(SessionLocal, Asset, asset_details)

            new_end_date = datetime.now(tz)
            print("timestep_in_file", timestep_in_file)
            new_start_date = (timestep_in_file - timestamp_diff_buffer * 10).astimezone(
                tz
            )

            print("new_start_date", new_start_date)
            print("timestamp_diff_buffer", timestamp_diff_buffer)

            df = fetch_pandas_data(
                asset_id, new_start_date, new_end_date, target_column, feature_columns
            )

            if df.empty:
                print("No data fetched, skipping iteration.")
                return

            X_update, X_last, new_next_timestamp, last_y_timestamp = (
                prepare_data_for_forecast(
                    df,
                    context_length,
                    forecast_length,
                    scaler,
                    timestep_in_file,
                    asset_details["target_attribute"],
                    asset_details["feature_attributes"],
                )
            )
            timestamp_diff_buffer = (
                new_next_timestamp - last_y_timestamp
            ) * context_length
            print("Latest timestamp updated.")
            print("Prepared data")
            if X_update is None and X_last is None:
                print("No new X sequences to process. Skipping...")
                return

            if len(X_update) > 0:
                print(f"Updating model's state with {len(X_update)} new X sequences.")
                print("First sequences from forecasting data:", X_update[:3])
                for i in range(len(X_update)):
                    x = X_update[i].reshape(
                        (1, context_length, X_update.shape[2])
                    )  # Shape: (1, context_length, features)
                    _ = model.predict(
                        x, batch_size=batch_size
                    )  # Perform prediction to update state
                print("Model's state updated with new X sequences.")

            # Forecast the next y using X_last
            if X_last is not None:
                print("last x sequence:", X_last)
                print("Forecasting the next value using the latest X sequence.")
                next_prediction_scaled = model.predict(X_last, batch_size=batch_size)
                next_prediction = scaler[target_column].inverse_transform(
                    next_prediction_scaled
                )
                print("Next prediction:", next_prediction[0][0])
                print("Predicted next timestamp:", new_next_timestamp)

                # Write prediction into Eliona
                write_into_eliona(
                    asset_id,
                    new_next_timestamp,
                    next_prediction[0][0],
                    target_column,
                    forecast_length,
                )

            else:
                print("X_last is None. Skipping forecasting.")

            # Save the updated model after processing
            processing_status = get_processing_status(
                SessionLocal, Asset, asset_details
            )
            if "Saving" in processing_status:
                print("Model is currently being saved in training. Skipping saving.")
            else:
                save_latest_timestamp(
                    SessionLocal, Asset, last_y_timestamp, tz, asset_details
                )
                save_model_to_eliona(model, model_filename)
                saveState(SessionLocal, Asset, model, asset_details)
                print(f"Model saved to {model_filename}.")
        else:
            print(f"Model {model_filename} does not exist. Skipping iteration.")

    # Now set up the WebSocket listener
    ELIONA_API_KEY = os.getenv("API_TOKEN")
    ELIONA_HOST = os.getenv("API_ENDPOINT")

    if not ELIONA_API_KEY or not ELIONA_HOST:
        print("Error: API_TOKEN or API_ENDPOINT environment variables not set.")
        return

    # Ensure ELIONA_HOST uses wss:// and includes /api/v2
    base_websocket_url = (
        ELIONA_HOST.replace("https://", "wss://").rstrip("/") + "/data-listener"
    )

    # Build query parameters
    query_params = []
    if asset_id is not None:
        query_params.append(f"assetId={asset_id}")
        query_params.append('data_subtype="input"')

    if query_params:
        base_websocket_url += "?" + "&".join(query_params)

    # Build the headers for authentication using X-API-Key
    headers = [f"X-API-Key: {ELIONA_API_KEY}"]

    # Reconnection logic variables
    reconnect_delay = 1  # Initial delay in seconds

    while True:

        print("Connecting to WebSocket...")
        websocket_url = base_websocket_url  # Reassign in case it changes
        print("WebSocket URL:", websocket_url)
        print("Headers:", headers)

        def on_message(ws, message):

            global last_processed_time
            current_time = time.time()
            with process_lock:
                if (current_time - last_processed_time) >= 5:
                    last_processed_time = current_time
                    print("Received message:", message)
                    try:
                        if (
                            get_asset_by_id(SessionLocal, Asset, id=asset_details["id"])
                            is None
                        ):
                            print("Asset does not exist")
                            sys.exit()
                        perform_forecast()
                    except Exception as e:
                        print("Error processing message:", e)
                else:
                    print(
                        "Received message within 5 seconds of the last one. Ignoring."
                    )

        def on_error(ws, error):
            print("WebSocket error:", error)

        def on_close(ws, close_status_code, close_msg):
            print(
                f"WebSocket connection closed. Code: {close_status_code}, Message: {close_msg}"
            )

        def on_open(ws):
            print("WebSocket connection opened")
            # Reset reconnection delay upon successful connection
            nonlocal reconnect_delay
            reconnect_delay = 1  # Reset to initial delay

        ws = websocket.WebSocketApp(
            websocket_url,
            header=headers,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )

        # SSL options to disable certificate verification if needed
        sslopt = {"cert_reqs": ssl.CERT_NONE}

        try:
            ws.run_forever(sslopt=sslopt, ping_interval=10, ping_timeout=8)
        except KeyboardInterrupt:
            print("WebSocket connection closed by user.")
            break
        except Exception as e:
            print("Exception occurred: ", e)

        # Reconnection logic
        print(f"Reconnecting in {reconnect_delay} seconds...")
        time.sleep(reconnect_delay)
        # Exponential backoff
