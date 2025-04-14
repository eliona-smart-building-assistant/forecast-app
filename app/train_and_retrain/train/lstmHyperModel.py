from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Input,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.optimizers import ( # type: ignore
    Adam,
    SGD,
    RMSprop,
    Adagrad,
    Adadelta,
    Adamax,
    Nadam,
)
from kerastuner import HyperModel
import logging

from api.models import HyperparametersModel, ParametersModel

logger = logging.getLogger(__name__)


class LSTMHyperModel(HyperModel):
    def __init__(
        self, context_length, num_features, parameters=None, hyperparameters=None
    ):
        self.context_length = context_length
        self.num_features = num_features
        self.parameters : ParametersModel = parameters or {}
        self.hyperparameters: HyperparametersModel = self.validate_hyperparameters(hyperparameters)

    def validate_hyperparameters(self, hyperparameters):
        if not hyperparameters:
            return {}
        if not isinstance(hyperparameters, dict):
            hyperparameters = hyperparameters.dict()
        validated_hyperparameters = {}
        for key, params in hyperparameters.items():
            if params is None:
                continue
            if isinstance(params, dict):
                required_keys = {"min_value", "max_value", "step"}
                if required_keys.issubset(params.keys()):
                    validated_hyperparameters[key] = params
            elif isinstance(params, list):
                validated_hyperparameters[key] = params
            elif isinstance(params, bool):
                validated_hyperparameters[key] = params

        return validated_hyperparameters

    def build(self, hp):
        inputs = Input(batch_shape=(1, self.context_length, self.num_features))
        x = inputs
        activation_choices = self.hyperparameters.get("activation") or ["tanh", "relu", "sigmoid", "linear"]
        activation = hp.Choice("activation", values=activation_choices)

        dense_activation_choices = self.hyperparameters.get("dense_activation") or ["tanh", "relu", "sigmoid", "linear"]
        dense_activation = hp.Choice("dense_activation", values=dense_activation_choices)

        num_lstm_layers_params = self.hyperparameters.get("num_lstm_layers", {})
        num_lstm_layers = hp.Int(
            "num_lstm_layers",
            min_value=num_lstm_layers_params.get("min_value", 1),
            max_value=num_lstm_layers_params.get("max_value", 3),
            step=num_lstm_layers_params.get("step", 1),
        )

        lstm_units_params = self.hyperparameters.get("lstm_units", {})
        lstm_units = hp.Int(
            "lstm_units",
            min_value=lstm_units_params.get("min_value", 32),
            max_value=lstm_units_params.get("max_value", 256),
            step=lstm_units_params.get("step", 32),
        )

        dropout_rate_params = self.hyperparameters.get("dropout_rate", {})
        dropout_rate = hp.Float(
            "dropout_rate",
            min_value=dropout_rate_params.get("min_value", 0.0),
            max_value=dropout_rate_params.get("max_value", 0.7),
            step=dropout_rate_params.get("step", 0.1),
        )

        recurrent_dropout_rate_params = self.hyperparameters.get("recurrent_dropout_rate", {})
        recurrent_dropout_rate = hp.Float(
            "recurrent_dropout_rate",
            min_value=recurrent_dropout_rate_params.get("min_value", 0.0),
            max_value=recurrent_dropout_rate_params.get("max_value", 0.7),
            step=recurrent_dropout_rate_params.get("step", 0.1),
        )

        num_dense_layers_params = self.hyperparameters.get("num_dense_layers", {})
        num_dense_layers = hp.Int(
            "num_dense_layers",
            min_value=num_dense_layers_params.get("min_value", 0),
            max_value=num_dense_layers_params.get("max_value", 3),
            step=num_dense_layers_params.get("step", 1),
        )

        dense_units_params = self.hyperparameters.get("dense_units", {})
        dense_units = hp.Int(
            "dense_units",
            min_value=dense_units_params.get("min_value", 32),
            max_value=dense_units_params.get("max_value", 256),
            step=dense_units_params.get("step", 32),
        )

        learning_rate_params = self.hyperparameters.get("learning_rate", {})
        learning_rate = hp.Float(
            "learning_rate",
            min_value=learning_rate_params.get("min_value", 1e-5),
            max_value=learning_rate_params.get("max_value", 1e-2),
            sampling="log",
        )

        optimizer_type_choices = self.hyperparameters.get("optimizer_type") or ["adam", "sgd", "rmsprop", "adagrad", "adadelta", "adamax", "nadam"]
        optimizer_type = hp.Choice("optimizer_type", values=optimizer_type_choices)

        use_batch_norm = hp.Boolean("use_batch_norm", default=self.hyperparameters.get("use_batch_norm", False))

        binary_encoding = self.parameters.binary_encoding
        num_classes = self.parameters.num_classes

        for layer_num in range(num_lstm_layers):
            return_seq = layer_num < num_lstm_layers - 1
            lstm_layer = LSTM(
                lstm_units,
                activation=activation,
                stateful=True,
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
            loss = "binary_crossentropy"
        elif num_classes is not None and num_classes > 1:
            outputs = Dense(num_classes, activation="softmax")(x)
            loss = "sparse_categorical_crossentropy"
        else:
            outputs = Dense(1)(x)
            loss = self.parameters.loss if self.parameters.loss is not None else "mean_squared_error"

        model = Model(inputs, outputs)

        optimizer_mapping = {
            "adam": Adam,
            "sgd": SGD,
            "rmsprop": RMSprop,
            "adagrad": Adagrad,
            "adadelta": Adadelta,
            "adamax": Adamax,
            "nadam": Nadam,
        }
        optimizer_class = optimizer_mapping[optimizer_type]
        optimizer = optimizer_class(learning_rate=learning_rate)

        model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
        return model