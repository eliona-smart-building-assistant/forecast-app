from app.get_data.api_calls import saveState, save_latest_timestamp
import tensorflow as tf
import numpy as np


from app.data_to_eliona.create_asset_to_save_models import save_model_to_eliona
from kerastuner.tuners import BayesianOptimization
from app.get_data.api_calls import set_processing_status


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(
        self, model_save_path, SessionLocal, Asset, asset_details, tz, latest_timestamp
    ):
        super(CustomCallback, self).__init__()
        self.model_save_path = model_save_path
        self.SessionLocal = SessionLocal
        self.Asset = Asset
        self.asset_details = asset_details
        self.best_val_loss = np.inf
        self.best_weights = None
        self.latest_timestamp = latest_timestamp
        self.tz = tz

    def on_epoch_end(self, epoch, logs=None):
        # Reset states after each epoch

        # Check if 'val_loss' improved
        current_val_loss = logs.get("val_loss")
        if current_val_loss is not None and current_val_loss < self.best_val_loss:
            print(
                f"Validation loss improved from {self.best_val_loss} to {current_val_loss}. Saving model."
            )
            self.best_val_loss = current_val_loss
            # Save the model
            set_processing_status(
                self.SessionLocal,
                self.Asset,
                self.asset_details,
                "Saving best model on epoch end",
            )
            save_model_to_eliona(self.model, self.model_save_path)
            # Call saveState function
            save_latest_timestamp(
                self.SessionLocal,
                self.Asset,
                self.latest_timestamp,
                self.tz,
                self.asset_details,
            )

            saveState(self.SessionLocal, self.Asset, self.model, self.asset_details)
            set_processing_status(
                self.SessionLocal,
                self.Asset,
                self.asset_details,
                "continue training next epoch",
            )
        else:
            print(f"Validation loss did not improve from {self.best_val_loss}.")
            if self.best_weights:
                self.model.set_weights(self.best_weights)
        stateful = self.asset_details["parameters"].get("stateful", True)
        if stateful:
            for layer in self.model.layers:
                if hasattr(layer, "reset_states") and callable(layer.reset_states):
                    layer.reset_states()


class HyperModelCheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(HyperModelCheckpointCallback, self).__init__()
        self.best_val_loss = np.inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        # Reset states of stateful layers
        current_val_loss = logs.get("val_loss")
        if current_val_loss is not None and current_val_loss < self.best_val_loss:
            print(
                f"Validation loss improved from {self.best_val_loss} to {current_val_loss}. Saving model."
            )
            self.best_val_loss = current_val_loss
            self.best_weights = self.model.get_weights()
        else:
            print(f"Validation loss did not improve from {self.best_val_loss}.")
            if self.best_weights:
                self.model.set_weights(self.best_weights)
        stateful = self.asset_details["parameters"].get("stateful", True)
        if stateful:
            for layer in self.model.layers:
                if hasattr(layer, "reset_states") and callable(layer.reset_states):
                    layer.reset_states()


class CustomBayesianOptimization(BayesianOptimization):
    def __init__(
        self,
        *args,
        model_save_path,
        SessionLocal,
        Asset,
        asset_details,
        tz,
        latest_timestamp,
        **kwargs,
    ):
        super(CustomBayesianOptimization, self).__init__(*args, **kwargs)
        self.model_save_path = model_save_path
        self.SessionLocal = SessionLocal
        self.Asset = Asset
        self.asset_details = asset_details
        self.latest_timestamp = latest_timestamp
        self.tz = tz

    def on_trial_end(self, trial):
        super(CustomBayesianOptimization, self).on_trial_end(trial)
        # Custom logic after each trial
        print(f"Trial {trial.trial_id} ended with score: {trial.score}")
        # Save the best model
        set_processing_status(
            self.SessionLocal,
            self.Asset,
            self.asset_details,
            "Saving best model on trial end",
        )
        best_model = self.get_best_models(num_models=1)[0]
        save_model_to_eliona(best_model, self.model_save_path)
        # Call saveState function
        save_latest_timestamp(
            self.SessionLocal,
            self.Asset,
            self.latest_timestamp,
            self.tz,
            self.asset_details,
        )
        saveState(self.SessionLocal, self.Asset, best_model, self.asset_details)
        set_processing_status(
            self.SessionLocal,
            self.Asset,
            self.asset_details,
            "continue training next trail",
        )
