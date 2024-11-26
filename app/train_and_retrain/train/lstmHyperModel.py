
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Input,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import (
    Adam,
    SGD,
    RMSprop,
    Adagrad,
    Adadelta,
    Adamax,
    Nadam,
)
from kerastuner import HyperModel


class LSTMHyperModel(HyperModel):
    def __init__(
        self, context_length, num_features, parameters=None, hyperparameters=None
    ):
        self.context_length = context_length
        self.num_features = num_features
        self.parameters = parameters or {}
        self.hyperparameters = self.validate_hyperparameters(hyperparameters)

    def validate_hyperparameters(self, hyperparameters):
        if not hyperparameters:
            return {}

        validated_hyperparameters = {}

        for key, params in hyperparameters.items():
            if isinstance(params, dict):
                # For hp.Int and hp.Float
                required_keys = {"min_value", "max_value", "step"}
                if required_keys.issubset(params.keys()):
                    validated_hyperparameters[key] = params
                else:
                    print(
                        f"Warning: Hyperparameter '{key}' is missing required keys. Using standard values."
                    )
            elif isinstance(params, list):
                # For hp.Choice
                validated_hyperparameters[key] = params
            elif isinstance(params, bool):
                # For hp.Boolean
                validated_hyperparameters[key] = params
            else:
                print(
                    f"Warning: Hyperparameter '{key}' has an unsupported format. Using standard values."
                )

        return validated_hyperparameters

    def build(self, hp):
        inputs = Input(batch_shape=(1, self.context_length, self.num_features))
        x = inputs

        # Hyperparameters to tune
        activation = hp.Choice(
            "activation",
            values=self.hyperparameters.get("activation", ["tanh", "relu", "sigmoid", "linear"]),
        )
        dense_activation = hp.Choice(
            "dense_activation",
            values=self.hyperparameters.get("dense_activation", ["tanh", "relu", "sigmoid", "linear"]),
        )
        
        # For num_lstm_layers
        num_lstm_layers_params = self.hyperparameters.get("num_lstm_layers", {})
        num_lstm_layers = hp.Int(
            "num_lstm_layers",
            min_value=num_lstm_layers_params.get("min_value", 1),
            max_value=num_lstm_layers_params.get("max_value", 4),
            step=num_lstm_layers_params.get("step", 1),
        )
        
        # For lstm_units
        lstm_units_params = self.hyperparameters.get("lstm_units", {})
        lstm_units = hp.Int(
            "lstm_units",
            min_value=lstm_units_params.get("min_value", 32),
            max_value=lstm_units_params.get("max_value", 256),
            step=lstm_units_params.get("step", 32),
        )
        
        # Similarly handle other hyperparameters
        # For dropout_rate
        dropout_rate_params = self.hyperparameters.get("dropout_rate", {})
        dropout_rate = hp.Float(
            "dropout_rate",
            min_value=dropout_rate_params.get("min_value", 0.0),
            max_value=dropout_rate_params.get("max_value", 0.5),
            step=dropout_rate_params.get("step", 0.1),
        )
        
        # For recurrent_dropout_rate
        recurrent_dropout_rate_params = self.hyperparameters.get("recurrent_dropout_rate", {})
        recurrent_dropout_rate = hp.Float(
            "recurrent_dropout_rate",
            min_value=recurrent_dropout_rate_params.get("min_value", 0.0),
            max_value=recurrent_dropout_rate_params.get("max_value", 0.5),
            step=recurrent_dropout_rate_params.get("step", 0.1),
        )
        
        # For num_dense_layers
        num_dense_layers_params = self.hyperparameters.get("num_dense_layers", {})
        num_dense_layers = hp.Int(
            "num_dense_layers",
            min_value=num_dense_layers_params.get("min_value", 0),
            max_value=num_dense_layers_params.get("max_value", 3),
            step=num_dense_layers_params.get("step", 1),
        )
        
        # For dense_units
        dense_units_params = self.hyperparameters.get("dense_units", {})
        dense_units = hp.Int(
            "dense_units",
            min_value=dense_units_params.get("min_value", 32),
            max_value=dense_units_params.get("max_value", 256),
            step=dense_units_params.get("step", 32),
        )
        
        # For learning_rate
        learning_rate_params = self.hyperparameters.get("learning_rate", {})
        learning_rate = hp.Float(
            "learning_rate",
            min_value=learning_rate_params.get("min_value", 1e-4),
            max_value=learning_rate_params.get("max_value", 1e-2),
            sampling="log",
        )
        
        # For optimizer_type
        optimizer_type = hp.Choice(
            "optimizer_type",
            values=self.hyperparameters.get("optimizer_type", ["adam", "sgd", "rmsprop", "adagrad", "adadelta", "adamax", "nadam"]),
        )
        
        # For use_batch_norm
        use_batch_norm = hp.Boolean(
            "use_batch_norm",
            default=self.hyperparameters.get("use_batch_norm", False)
        )

        # Build the model using these hyperparameters
        for layer_num in range(num_lstm_layers):
            return_seq = True if layer_num < num_lstm_layers - 1 else False
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

        outputs = Dense(1)(x)

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

        model.compile(
            optimizer=optimizer,
            loss=self.parameters.get("loss", "mean_squared_error"),
        )

        return model