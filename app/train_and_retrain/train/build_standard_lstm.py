import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.models import Model # type: ignore
import json

def build_lstm_model(
    context_length, num_features, parameters  
):
    if isinstance(parameters, str):
        parameters = json.loads(parameters)
    elif not isinstance(parameters, dict):
        parameters = dict(parameters)
        
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
    binary_encoding = parameters.get("binary_encoding", False)
    num_classes = parameters.get("num_classes", None) 

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

        for _ in range(num_dense_layers):
            x = Dense(dense_units, activation=dense_activation)(x)
            if use_batch_norm:
                x = BatchNormalization()(x)
            if dropout_rate > 0:
                x = Dropout(dropout_rate)(x)

    if binary_encoding:
        outputs = Dense(1, activation="sigmoid")(x)
        loss = parameters.get("loss", "binary_crossentropy")
        metrics = parameters.get("metrics", ["accuracy"])
    elif num_classes is not None and num_classes > 1:
        outputs = Dense(num_classes, activation="softmax")(x)
        loss = parameters.get(
            "loss", "sparse_categorical_crossentropy"
        ) 
        metrics = parameters.get("metrics", ["accuracy"])
    else:
        outputs = Dense(1)(x)

    model = Model(inputs, outputs)
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

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics if metrics else [])

    return model
