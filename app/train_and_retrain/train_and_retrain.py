import time
import pytz
import os
from datetime import datetime
import sys
import logging
from app.get_data.fetch_and_format_data import fetch_pandas_data
from app.train_and_retrain.train.train import train_lstm_model
from app.get_data.api_calls import save_datalength,  load_datalength, set_training_status
from api.api_calls import get_asset_by_id
from api.models import AssetModel, TrainingStatus
from app.data_to_eliona.add_forecast_attributes import add_forecast_attributes
logger = logging.getLogger(__name__)

def train_and_retrain(asset: AssetModel):
    tz = pytz.timezone("Europe/Berlin")
    start_date = tz.localize(datetime.strptime(asset.start_date, "%Y-%m-%d"))
    logger.info(f"start_date: {start_date}")
    
    forecast_name_suffix = f"_forecast_{asset.forecast_length}"

    try:
        asset_id = add_forecast_attributes(asset.gai, asset.target_attribute, forecast_name_suffix)
    except Exception as e:
        logger.error(f"Error computing asset_id: {e}")
        sys.exit(1)
    model_filename = f"/tmp/LSTM_model_{asset_id}_{asset.target_attribute}_{asset.forecast_length}.keras"
    
    def train_and_handle(df):
        train_lstm_model(
            asset,
            asset_id,
            df,
            tz=tz,
            model_save_path=model_filename,
        )
        set_training_status(asset, TrainingStatus.COMPLETED)
        save_datalength(len(df), asset)

    while True:
        if get_asset_by_id(id=asset.id) is None:
            logger.info("Asset does not exist. Exiting training loop.")
            sys.exit()
        end_date = tz.localize(datetime.now())
        
        if os.path.exists(model_filename):
            logger.info(f"Model {model_filename} exists.")
            data_length = load_datalength(asset) or 0
            logger.info(f"Data length: {data_length}")
            df = fetch_pandas_data(
                asset_id,
                start_date,
                end_date,
                asset.target_attribute,
                asset.feature_attributes,
            )
            required_min_length = asset.context_length + asset.forecast_length
            if len(df) < required_min_length:
                logger.info("Not enough data to forecast.")
                logger.info(f"Required min length: {required_min_length}")
                set_training_status(asset, TrainingStatus.NOT_ENOUGH_DATA)
                logger.info("Skipping retraining due to insufficient data.")
                logger.info(f"Sleeping for {asset.trainingparameters.sleep_time} seconds before next retraining cycle...")
                time.sleep(asset.trainingparameters.sleep_time)
                continue
            if len(df) > data_length * asset.trainingparameters.percentage_data_when_to_retrain:
                logger.info("Retraining model.")
                set_training_status(asset, TrainingStatus.START_RE_TRAINING)
                train_and_handle(df)
        else:
            logger.info("Model does not exist. Starting training.")
            df = fetch_pandas_data(
                asset_id,
                start_date,
                end_date,
                asset.target_attribute,
                asset.feature_attributes,
            )
            required_min_length = asset.context_length + asset.forecast_length
            if len(df) < required_min_length:
                logger.info("Not enough data to forecast.")
                logger.info(f"Required min length: {required_min_length}")
                set_training_status(asset, TrainingStatus.NOT_ENOUGH_DATA)
                logger.info("Skipping training due to insufficient data.")
                logger.info(f"Sleeping for {asset.trainingparameters.sleep_time} seconds before next training cycle...")
                time.sleep(asset.trainingparameters.sleep_time)
                continue
            set_training_status(asset, TrainingStatus.START_TRAINING)
            train_and_handle(df)

        logger.info(f"Sleeping for {asset.trainingparameters.sleep_time} seconds before next retraining cycle...")
        time.sleep(asset.trainingparameters.sleep_time)