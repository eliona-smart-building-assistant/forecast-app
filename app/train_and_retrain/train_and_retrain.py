import time
import pytz
import os
from datetime import datetime
from app.get_data.fetch_and_format_data import fetch_pandas_data
from app.train_and_retrain.train.train import train_lstm_model
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import (
    MetaData,
    Table,
    create_engine,
)
import sys
from app.get_data.api_calls import (
    save_datalength,
    load_contextlength,
    load_datalength,
    set_processing_status,
)
from app.data_to_eliona.create_asset_to_save_models import model_exists

from api.api_calls import get_asset_by_id


def train_and_retrain(
    asset_details,
    asset_id,
):
    trainingparameters = asset_details["trainingparameters"] or {}
    sleep_time = trainingparameters.get("sleep_time", 3600) or 3600
    print("sleep_time", sleep_time)
    forecast_length = asset_details["forecast_length"]
    target_column = asset_details["target_attribute"]
    tz = pytz.timezone("Europe/Berlin")
    start_date_str = asset_details["start_date"] or "2024-11-6"
    print("start_date_str", start_date_str)
    start_date = tz.localize(datetime.strptime(start_date_str, "%Y-%m-%d"))
    print("start_date", start_date)
    model_filename = f"LSTM_model_{asset_id}_{target_column}_{forecast_length}.keras"
    percentage_data_when_to_retrain = (
        asset_details["trainingparameters"].get("percentage_data_when_to_retrain", 1.15)
        or 1.15
    )
    print("percentage_data_when_to_retrain", percentage_data_when_to_retrain)
    db_url = os.getenv("CONNECTION_STRING")
    db_url_sql = db_url.replace("postgres", "postgresql")
    DATABASE_URL = db_url_sql
    engine = create_engine(DATABASE_URL)

    # Use MetaData to reflect the 'assets_to_forecast' table from the 'forecast' schema
    metadata = MetaData()
    Asset = Table(
        "assets_to_forecast", metadata, autoload_with=engine, schema="forecast"
    )
    # Create a new session for database interactions
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    context_length = load_contextlength(SessionLocal, Asset, asset_details)

    def train_and_handle():

        train_lstm_model(
            asset_details,
            asset_id,
            df,
            SessionLocal=SessionLocal,
            Asset=Asset,
            tz=tz,
            context_length=context_length,
            forecast_length=forecast_length,
            model_save_path=model_filename,
        )
        set_processing_status(SessionLocal, Asset, asset_details, "done_training")
        print("Length X", len(df))
        save_datalength(SessionLocal, Asset, len(df), asset_details)

    while True:
        if get_asset_by_id(SessionLocal, Asset, id=asset_details["id"]) == None:
            print("Asset does not exist")
            sys.exit()
        end_date = tz.localize(datetime.now())

        if model_exists(model_filename):
            print(f"Model {model_filename} exists")
            data_length = load_datalength(SessionLocal, Asset, asset_details) or 0
            print("Data length", data_length)
            df = fetch_pandas_data(
                asset_id,
                start_date,
                end_date,
                target_column,
                asset_details["feature_attributes"],
            )
            required_min_length = context_length + forecast_length
            if len(df) < required_min_length:
                print("Not enough data to forecast.")
                print("min length", required_min_length)
                set_processing_status(
                    SessionLocal, Asset, asset_details, "not enough data for training"
                )
                print(f"Skipping retraining due to insufficient data.")
                time.sleep(sleep_time)
                continue
            if len(df) > data_length * percentage_data_when_to_retrain:
                print("Retraining model")
                set_processing_status(
                    SessionLocal, Asset, asset_details, "start_re_training"
                )
                train_and_handle()
        else:
            print("Model does not exist")

            df = fetch_pandas_data(
                asset_id,
                start_date,
                end_date,
                target_column,
                asset_details["feature_attributes"],
            )
            required_min_length = context_length + forecast_length
            if len(df) < required_min_length:
                print("Not enough data to forecast.")
                print("min length", required_min_length)
                set_processing_status(
                    SessionLocal, Asset, asset_details, "not enough data for training"
                )
                print(f"Skipping retraining due to insufficient data.")
                time.sleep(sleep_time)
                continue

            print(df.tail(20))
            set_processing_status(SessionLocal, Asset, asset_details, "start_training")
            train_and_handle()

        # Wait for the specified sleep time before running again
        print(f"Sleeping for {sleep_time} seconds before next retraining cycle...")
        time.sleep(sleep_time)
