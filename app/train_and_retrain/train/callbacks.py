from api.models import  AssetModel, TrainingStatus
from app.get_data.api_calls import get_train_bool, saveState, save_latest_timestamp
import tensorflow as tf
import numpy as np


from kerastuner.tuners import BayesianOptimization
from app.get_data.api_calls import set_training_status
import logging
from filelock import FileLock, Timeout

logger = logging.getLogger(__name__)


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_save_path, asset_details, tz, latest_timestamp):
        super(CustomCallback, self).__init__()
        self.model_save_path = model_save_path
        self.asset_details: AssetModel = asset_details
        self.best_val_loss = np.inf
        self.best_weights = None
        self.latest_timestamp = latest_timestamp
        self.tz = tz
        self.lock_file = self.model_save_path + ".lock"

    def on_epoch_end(self, epoch, logs=None):
        if not get_train_bool(self.asset_details):
            set_training_status(self.asset_details, TrainingStatus.INACTIVE)
            logger.info(f"Training bool is false. Stopping training from callback. for asset: {self.asset_details.id}")
            self.model.stop_training = True
            return
        current_val_loss = logs.get("val_loss")
        if current_val_loss is not None and current_val_loss < self.best_val_loss:
            logger.info(f"Validation loss for {self.asset_details.id} improved from {self.best_val_loss} to {current_val_loss}. Saving model." )
            self.best_weights = self.model.get_weights()
            self.best_val_loss = current_val_loss
            lock = FileLock(self.lock_file, timeout=1e6)
            try:
                with lock:
                    set_training_status(self.asset_details, TrainingStatus.SAVE_ON_EPOCH_END)
                    self.model.save(self.model_save_path)
                    save_latest_timestamp(self.latest_timestamp, self.tz, self.asset_details)
                    saveState(self.model, self.asset_details)

            except Timeout:
                logger.error(f"Timeout occurred while trying to acquire the file lock {self.asset_details.id}")
        else:
            logger.info(f"Validation loss did not improve for {self.asset_details.id} from {self.best_val_loss}.")
            if self.best_weights:
                self.model.set_weights(self.best_weights)
        stateful = self.asset_details.parameters.stateful if self.asset_details.parameters is not None else True
        if stateful:
            for layer in self.model.layers:
                if hasattr(layer, "reset_states") and callable(layer.reset_states):
                    layer.reset_states()
        epoch_status = f"Training Epoch {epoch + 1}/{self.asset_details.trainingparameters.epochs} best val_loss: {self.best_val_loss} last val_loss: {current_val_loss}."
        set_training_status(self.asset_details, epoch_status)
                    
                    
class HyperModelCheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, asset, stateful):
        super(HyperModelCheckpointCallback, self).__init__()
        self.asset = asset
        self.best_val_loss = np.inf
        self.best_weights = None
        self.stateful = stateful

    def on_epoch_end(self, epoch, logs=None):
        if not get_train_bool(self.asset):
            set_training_status(self.asset, TrainingStatus.INACTIVE)
            logger.info(f"Training bool is false. Stopping training from callback.")
            self.model.stop_training = True
            return
        
        current_val_loss = logs.get("val_loss")
        if current_val_loss is not None and current_val_loss < self.best_val_loss:
            logger.info(
                f"Validation loss improved from {self.best_val_loss} to {current_val_loss}. Saving weights internally. "
            )
            self.best_val_loss = current_val_loss
            self.best_weights = self.model.get_weights()
        else:
            logger.info(f"Validation loss did not improve from {self.best_val_loss}.")
            if self.best_weights:
                self.model.set_weights(self.best_weights)

        if self.stateful:
            for layer in self.model.layers:
                if hasattr(layer, "reset_states") and callable(layer.reset_states):
                    layer.reset_states()


class CustomBayesianOptimization(BayesianOptimization):
    def __init__(self, *args, model_save_path, asset_details, tz, latest_timestamp, **kwargs):
        super(CustomBayesianOptimization, self).__init__(*args, **kwargs)
        self.model_save_path = model_save_path
        self.asset_details : AssetModel = asset_details
        self.latest_timestamp = latest_timestamp
        self.tz = tz
        self.lock_file = self.model_save_path + ".lock"
        self.best_val_loss = np.inf

    def run_trial(self, trial, *args, **kwargs):
        if not get_train_bool(self.asset_details):
            set_training_status(self.asset_details, TrainingStatus.INACTIVE)
            logger.info(f"Training bool is false; aborting hyperparameter search {self.asset_details.id}")
            raise KeyboardInterrupt(f"Training bool switched off; aborting hyperparameter searchfor {self.asset_details.id}")
        return super(CustomBayesianOptimization, self).run_trial(trial, *args, **kwargs)

    def on_trial_end(self, trial):
        super(CustomBayesianOptimization, self).on_trial_end(trial)
        logger.info(f"Trial {trial.trial_id} ended with score: {trial.score}")
        if trial.score < self.best_val_loss:
            self.best_val_loss = trial.score
        from filelock import FileLock, Timeout
        lock = FileLock(self.lock_file, timeout=1e6)
        try:
            with lock:
                set_training_status(self.asset_details, TrainingStatus.SAVE_ON_TRAIL_END)
                best_model = self.get_best_models(num_models=1)[0]
                best_model.save(self.model_save_path)
                save_latest_timestamp(self.latest_timestamp, self.tz, self.asset_details)
                saveState(best_model, self.asset_details)
                trail_end_status = (f"Trial {trial.trial_id}/{self.asset_details.hyperparameters.max_trails} ended with score: {trial.score} best val_loss: {self.best_val_loss} continue training next trail")
                set_training_status(self.asset_details, trail_end_status)
        except Timeout:
            logger.error(f"Timeout occurred while trying to acquire the file lockfor {self.asset_details.id}")
