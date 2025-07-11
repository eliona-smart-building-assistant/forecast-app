import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from api.models import AssetModel, TrainingStatus
from app.get_data.fetch_and_format_data import prepare_data
from app.get_data.api_calls import (
    save_scaler,
    set_training_status,
)
from .build_standard_lstm import build_lstm_model
from .callbacks import CustomCallback
import logging

logger = logging.getLogger(__name__)

def train_lstm_model(
    asset: AssetModel,
    asset_id,
    data,
    tz,
    model_save_path,
):
    batch_size = 16

    set_training_status(asset, TrainingStatus.PREPARING)
    X, y, scaler, last_timestamp = prepare_data(asset, data)
    save_scaler(scaler, asset)
    num_features = X.shape[2]

    validation_samples = int(len(X) * asset.trainingparameters.validation_split)
    if validation_samples == 0:
        logger.info(f"Validation split results in 0 validation samples. Skipping training for {asset.id}.")
        return

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=asset.trainingparameters.validation_split, shuffle=False
    )

    # Build the simple LSTM model
    model = build_lstm_model(asset.context_length, num_features, parameters={})

    # Define callbacks
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

    set_training_status(asset, TrainingStatus.START_TRAINING)
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