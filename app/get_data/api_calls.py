from api.api_calls import (
    update_asset,
    get_asset_by_id,
)
from datetime import datetime
import numpy as np
import pandas as pd
from tensorflow.keras.layers import LSTM
import pickle
import base64 
import logging

from api.models import AssetModel, ForecastStatus, TrainingStatus

logger = logging.getLogger(__name__)


def saveState(model, asset:AssetModel):
    states = {}
    for layer in model.layers:
        if isinstance(layer, LSTM) and layer.stateful:
            if layer.states is not None:
                h_state, c_state = layer.states
                states[layer.name] = [h_state.numpy(), c_state.numpy()]
            else:
                logger.info(f"Warning: Layer '{layer.name}' has no initialized states.")

    serialized_states = pickle.dumps(states)
    update_asset(
        id=asset.id,
        state=serialized_states,
    )


def loadState(model, asset:AssetModel):
    asset = get_asset_by_id(asset.id)
    if asset.state: 

        raw_state = base64.b64decode(asset.state)
        states = pickle.loads(raw_state)
    else:
        logger.info("No saved states found for the given asset.")
        return

    for layer in model.layers:
        if isinstance(layer, LSTM) and layer.stateful:
            if layer.name in states:
                h_state_value, c_state_value = states[layer.name]
                h_state, c_state = layer.states
                h_state.assign(h_state_value)
                c_state.assign(c_state_value)
                logger.info(f"States loaded into layer '{layer.name}'")
            else:
                logger.info(f"No saved state for layer '{layer.name}'")
    return model




def save_latest_timestamp(timestamp, tz, asset: AssetModel):

    if isinstance(timestamp, datetime) and timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=tz)
    elif isinstance(timestamp, np.datetime64):
        timestamp = pd.to_datetime(timestamp).tz_localize(tz).to_pydatetime()

    update_asset(id=asset.id, latest_timestamp=timestamp)


def load_latest_timestamp(
    asset:AssetModel,
):
    asset = get_asset_by_id(asset.id)
    return asset.latest_timestamp





def load_datalength(asset:  AssetModel):
    logger.info(asset)
    asset = get_asset_by_id(asset.id)
    return asset.datalength


def save_datalength(datalength, asset:  AssetModel):
    update_asset(
        id=asset.id,
        datalength=datalength,
    )


def save_scaler(scaler, asset:  AssetModel):
    serialized_scaler = pickle.dumps(scaler)
    update_asset(
        id=asset.id,
        scaler=serialized_scaler,  
    )


def load_scaler(asset: AssetModel):
    asset = get_asset_by_id(asset.id)
    if asset.scaler:
        raw_scaler = base64.b64decode(asset.scaler)  
        return pickle.loads(raw_scaler)
    else:
        logger.info("No scaler found for the given asset.")
        return None


def save_parameters(parameters, asset: AssetModel):

    existing_parameters = asset.parameters or {}

    if parameters:
        existing_parameters.update(parameters)
    update_asset(id=asset.id, parameters=existing_parameters)

    asset.parameters = existing_parameters


def set_forecast_status(asset: AssetModel, status: ForecastStatus):
    update_asset(id=asset.id, forecast_status=status.value)


def get_forecast_status(asset: AssetModel):
    updated_asset = get_asset_by_id(asset.id)
    return updated_asset.forecast_status.value


def set_training_status(asset: AssetModel, status: TrainingStatus):
    update_asset(id=asset.id, train_status=status.value)


def get_training_status(asset: AssetModel):
    updated_asset = get_asset_by_id(asset.id)
    return updated_asset.train_status.value

def set_forecast_bool(asset: AssetModel, bool: bool):
    update_asset(id=asset.id, forecast=bool)

def get_forecast_bool(asset: AssetModel):
    updated_asset = get_asset_by_id(asset.id)
    return updated_asset.forecast

def set_train_bool(asset: AssetModel, bool: bool):
    update_asset(id=asset.id, train=bool)

def get_train_bool(asset: AssetModel):
    updated_asset = get_asset_by_id(asset.id)
    return updated_asset.train