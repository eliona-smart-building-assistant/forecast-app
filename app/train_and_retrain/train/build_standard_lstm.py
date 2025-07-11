from tensorflow.keras.layers import InputLayer, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError

def build_lstm_model(context_length, num_features, parameters):
    lookback = context_length  # Assuming context_length corresponds to lookback
   

    model = Sequential()
    model.add(InputLayer(input_shape=(lookback, num_features)))  # Explicit input layer
    model.add(LSTM(64))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=MeanSquaredError(),
        metrics=[RootMeanSquaredError()]
    )
    model.summary()


    return model