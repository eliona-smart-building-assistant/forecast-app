import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model
import json
import logging

# Initialize the logger
logger = logging.getLogger(__name__)


def build_lstm_model(
    context_length, num_features, parameters  # Catch-all for additional parameters
):
    """
    Builds a customizable multi-layer stateful LSTM model for time series forecasting.

    :param context_length: Number of timesteps used for context (input window)
    :param num_features: Number of input features
    :param parameters: Dictionary of additional hyperparameters
    :return: Compiled LSTM model
    """

    from tensorflow.keras.layers import Dropout, BatchNormalization

    if isinstance(parameters, str):
        parameters = json.loads(parameters)
    elif not isinstance(parameters, dict):
        parameters = dict(parameters)

    # Set default values
    logger.info(f"parameters {parameters}")
    num_lstm_layers = parameters.get("num_lstm_layers", 2)
    lstm_units = parameters.get("lstm_units", 50)

    activation = parameters.get("activation", "tanh")
    learning_rate = parameters.get("learning_rate", 0.001)
    optimizer_type = parameters.get("optimizer_type", "adam")
    clipnorm = parameters.get("clipnorm", None)
    loss = parameters.get("loss", "mean_squared_error")
    dropout_rate = parameters.get("dropout_rate", 0.0)
    recurrent_dropout_rate = parameters.get("recurrent_dropout_rate", 0.0)
    num_dense_layers = parameters.get("num_dense_layers", 0)
    dense_units = parameters.get("dense_units", 50)
    dense_activation = parameters.get("dense_activation", "relu")
    use_batch_norm = parameters.get("use_batch_norm", False)
    metrics = parameters.get("metrics", ["mse"])
    stateful = parameters.get("stateful", True)
    batch_size = parameters.get("batch_size", 1)
    logger.info("all parameters:")
    logger.info(f"num_lstm_layers {num_lstm_layers}")
    logger.info(f"lstm_units {lstm_units}")
    logger.info(f"activation {activation}")
    logger.info(f"learning_rate {learning_rate}")
    logger.info(f"optimizer_type {optimizer_type}")
    logger.info(f"clipnorm {clipnorm}")
    logger.info(f"loss {loss}")
    logger.info(f"dropout_rate {dropout_rate}")
    logger.info(f"recurrent_dropout_rate {recurrent_dropout_rate}")
    logger.info(f"num_dense_layers {num_dense_layers}")
    logger.info(f"dense_units {dense_units}")
    logger.info(f"dense_activation {dense_activation}")
    logger.info(f"use_batch_norm {use_batch_norm}")
    logger.info(f"metrics {metrics}")
    logger.info(f"batch_size {batch_size}")

    inputs = Input(batch_shape=(batch_size, context_length, num_features))
    x = inputs

    for layer_num in range(num_lstm_layers):
        return_seq = True if layer_num < num_lstm_layers - 1 else False
        lstm_layer = LSTM(
            lstm_units,
            activation=activation,
            stateful=stateful,
            return_sequences=return_seq,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout_rate,
        )

        x = lstm_layer(x)

        if use_batch_norm:
            x = BatchNormalization()(x)

        # Add Dense layers

        x = Dense(
            dense_units,
            activation=dense_activation,
        )(x)
        if use_batch_norm:
            x = BatchNormalization()(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)

    # Output layer
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)

    # Select optimizer
    optimizer_mapping = {
        "adam": tf.keras.optimizers.Adam,
        "sgd": tf.keras.optimizers.SGD,
        "rmsprop": tf.keras.optimizers.RMSprop,
        "adagrad": tf.keras.optimizers.Adagrad,
        "adadelta": tf.keras.optimizers.Adadelta,
        "adamax": tf.keras.optimizers.Adamax,
        "nadam": tf.keras.optimizers.Nadam,
    }

    optimizer_class = optimizer_mapping.get(
        optimizer_type.lower(), tf.keras.optimizers.Adam
    )
    optimizer_kwargs = {"learning_rate": learning_rate}
    if clipnorm is not None:
        optimizer_kwargs["clipnorm"] = clipnorm

    optimizer = optimizer_class(**optimizer_kwargs)
    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics if metrics else [])

    return model
