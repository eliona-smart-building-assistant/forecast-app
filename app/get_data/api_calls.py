from api.api_calls import (
    update_asset,
    get_asset_by_id,
)
from datetime import datetime
import numpy as np
import pandas as pd
from tensorflow.keras.layers import LSTM
import pickle

import logging

# Initialize the logger
logger = logging.getLogger(__name__)


def saveState(model, asset_details):
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
        id=asset_details["id"],
        state=serialized_states,
    )


def loadState(model, asset_details):
    asset = get_asset_by_id(asset_details["id"])
    if asset.state:
        states = pickle.loads(asset.state)
    else:
        logger.info("No saved states found for the given asset.")
        return

    # Set the states in the model's LSTM layers
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


def printState(model):
    """
    Prints the hidden and cell states of all stateful LSTM layers in the model.

    :param model: The Keras model containing stateful LSTM layers
    """
    for layer in model.layers:
        if isinstance(layer, LSTM) and layer.stateful:
            h_state, c_state = layer.states
            logger.info(f"States for layer '{layer.name}':")
            logger.info(f"Hidden state (h): {h_state.numpy()}")
            logger.info(f"Cell state (c): {c_state.numpy()}")


def save_latest_timestamp(timestamp, tz, asset_details):
    logger.info("Updating latest timestamp")
    logger.info(f"{timestamp}")
    if isinstance(timestamp, datetime) and timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=tz)
    elif isinstance(timestamp, np.datetime64):
        timestamp = pd.to_datetime(timestamp).tz_localize(tz).to_pydatetime()

    update_asset(id=asset_details["id"], latest_timestamp=timestamp)


def load_latest_timestamp(
    asset_details,
):
    asset = get_asset_by_id(asset_details["id"])
    return asset.latest_timestamp


def load_contextlength(asset_details):
    asset = get_asset_by_id(asset_details["id"])
    context_length = asset.context_length

    if not context_length:
        context_length = asset.forecast_length * 3

    return context_length


def load_datalength(asset_details):
    asset = get_asset_by_id(asset_details["id"])
    return asset.datalength


def save_datalength(datalength, asset_details):
    update_asset(
        id=asset_details["id"],
        datalength=datalength,
    )


def save_scaler(scaler, asset_details):
    """
    Serializes and saves the scaler to the database for the given asset.


    :param scaler: The scaler object to be serialized and saved
    :param asset_details: Dictionary containing asset details
    """
    # Serialize the scaler using pickle
    logger.info("Saving scaler")
    logger.info(f"{scaler}")
    serialized_scaler = pickle.dumps(scaler)
    logger.info(f"{serialized_scaler}")
    update_asset(
        id=asset_details["id"],
        scaler=serialized_scaler,  # Save serialized bytes
    )


def load_scaler(asset_details):
    """
    Loads and deserializes the scaler from the database for the given asset.

    :param SessionLocal: The database session
    :param Asset: The Asset model
    :param asset_details: Dictionary containing asset details
    :return: The deserialized scaler object
    """
    asset = get_asset_by_id(asset_details["id"])
    if asset.scaler:
        return pickle.loads(asset.scaler)  # Deserialize the scaler
    else:
        logger.info("No scaler found for the given asset.")
        return None


def save_parameters(parameters, asset_details):
    """
    Updates parameters in asset_details while preserving unspecified ones.

    Args:
        SessionLocal: Database session
        Asset: Asset model
        parameters: New parameters to update
        asset_details: Current asset details
    """
    # Get existing parameters or initialize empty dict
    existing_parameters = asset_details.get("parameters", {}) or {}

    # Update only specified parameters
    if parameters:
        existing_parameters.update(parameters)

    # Save updated parameters
    update_asset(id=asset_details["id"], parameters=existing_parameters)

    # Update asset_details with new parameters
    asset_details["parameters"] = existing_parameters


def set_processing_status(asset_details, status):
    """
    Updates the processing status of the asset.

    Args:
        SessionLocal: Database session
        Asset: Asset model
        asset_details: Current asset details
        status: New processing status
    """
    update_asset(id=asset_details["id"], processing_status=status)


def get_processing_status(asset_details):
    """
    Retrieves the processing status of the asset.

    Args:
        SessionLocal: Database session
        Asset: Asset model
        asset_details: Current asset details
    """
    asset = get_asset_by_id(asset_details["id"])
    return asset.processing_status
