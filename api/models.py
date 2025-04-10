from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Union
import base64
import re
from sqlalchemy import Row
import enum

class ForecastStatus(str, enum.Enum):
    NEW = "new"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    ERROR = "error"
    RUNNING = "running"

class TrainingStatus(str, enum.Enum):
    NEW = "new"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    ERROR = "error"
    NoData = "not enough data for training"
    START_TRAINING = "start_training"
    COMPLETED = "done_training"
    NOT_ENOUGH_DATA = "not enough data for training"
    START_RE_TRAINING = "start_re_training"
    SAVE_ON_EPOCH_END = "Saving best model on epoch end"
    CONTINUE_TRAINING = "continue training next epoch"
    SAVE_ON_TRAIL_END = "Saving best model on trial end"
    TRAIN_NEXT_TRIAL = "continue training next trial"
    START_HYPER_TRAINING = "start_hyperparameter_search_training"

class ParametersModel(BaseModel):
    was_empty: bool = Field(default=True, description="True if no custom parameters were provided")
    num_classes: Optional[int] = None
    binary_encoding: Optional[bool] = None
    num_lstm_layers: Optional[int] = 2
    lstm_units: Optional[int] = 50
    activation: Optional[str] = "tanh"
    learning_rate: Optional[float] = 0.001 
    optimizer_type: Optional[str] =  "adam"
    clipnorm: Optional[float] = 1.0 
    loss: Optional[str] = "mse"
    dropout_rate: Optional[float] = 0.2
    recurrent_dropout_rate: Optional[float] = 0.2
    num_dense_layers: Optional[int] = 1
    dense_units: Optional[int] = 8
    dense_activation: Optional[str] = "relu"
    use_batch_norm: Optional[bool] = False
    metrics: Optional[List[str]] = ["mae"]
    stateful: Optional[bool] = True
    batch_size: Optional[int] = 1

class TrainingParametersModel(BaseModel):
    was_empty: bool = Field(default=True, description="True if no custom training parameters were provided")
    epochs: int = 200
    patience: int = 10
    sleep_time: int = 3600
    percentage_data_when_to_retrain: float = 1.15
    validation_split: float = 0.2
    objective: str = "val_loss"

class HyperparametersModel(BaseModel):
    was_empty: bool = Field(default=True, description="True if no custom hyperparameters were provided")
    activation: List[str] = Field(default_factory=list)
    dense_activation: List[str] = Field(default_factory=list)
    num_lstm_layers: Dict[str, Union[int, float]] = Field(default_factory=dict)
    lstm_units: Dict[str, Union[int, float]] = Field(default_factory=dict)
    dropout_rate: Dict[str, Union[int, float]] = Field(default_factory=dict)
    recurrent_dropout_rate: Dict[str, Union[int, float]] = Field(default_factory=dict)
    num_dense_layers: Dict[str, Union[int, float]] = Field(default_factory=dict)
    dense_units: Dict[str, Union[int, float]] = Field(default_factory=dict)
    learning_rate: Dict[str, Union[float, str]] = Field(default_factory=dict)
    optimizer_type: List[str] = Field(default_factory=list)
    use_batch_norm: Optional[bool] = None
    percent_data: float = 0.1
    max_trials: int = 100
    best_trails_percent: Optional[float] = None

class AssetModel(BaseModel):
    id: Optional[int] = None
    gai: str
    target_attribute: str
    forecast_length: int
    context_length: int
    feature_attributes: Optional[List[str]] = []
    start_date: Optional[str] = "2025-1-1"
    parameters: ParametersModel = Field(default_factory=ParametersModel)
    trainingparameters: TrainingParametersModel = Field(default_factory=TrainingParametersModel)
    hyperparameters: HyperparametersModel = Field(default_factory=HyperparametersModel)
    train: Optional[bool] = False
    forecast: Optional[bool] = False
    datalength: Optional[int] = 0
    latest_timestamp: Optional[str] = None
    scaler: Optional[str] = ""
    state: Optional[str] = None
    forecast_status: ForecastStatus = ForecastStatus.NEW
    train_status: TrainingStatus = TrainingStatus.NEW

    @field_validator("parameters", mode="before", check_fields=True)
    def set_default_parameters(cls, v):
        return v or ParametersModel()

    @classmethod
    def from_orm(cls, asset):
        asset_dict = asset._asdict() if isinstance(asset, Row) else dict(asset)
        if asset_dict.get("scaler"):
            asset_dict["scaler"] = base64.b64encode(asset_dict["scaler"]).decode("utf-8")
        if asset_dict.get("state"):
            asset_dict["state"] = base64.b64encode(asset_dict["state"]).decode("utf-8")
        return cls(**asset_dict)

    def to_db_model(self):
        asset_dict = self.model_dump()
        asset_dict.pop("train_status", None)
        asset_dict.pop("forecast_status", None)
        for field in ["scaler", "state"]:
            value = asset_dict.get(field)
            if value:
                if isinstance(value, str):
                    try:
                        asset_dict[field] = base64.b64decode(value.encode("utf-8"))
                    except Exception as e:
                        asset_dict[field] = None
                # If already bytes, leave them as is.
            else:
                asset_dict[field] = None
        return asset_dict

    class Config:
        # This tells Pydantic to use the enum value (e.g., "new") when serializing enum fields.
        use_enum_values = True

class ModelModel(BaseModel):
    filename: str
    asset_id: Optional[int] = None
    target_column: Optional[str] = None
    forecast_length: Optional[int] = None
    exists: bool

    @classmethod
    def from_filename(cls, filename: str):
        pattern = r"LSTM_model_(\d+)_(\w+)_(\d+)\.keras"
        match = re.match(pattern, filename)
        if match:
            asset_id, target_column, forecast_length = match.groups()
            return cls(
                filename=filename,
                asset_id=int(asset_id),
                target_column=target_column,
                forecast_length=int(forecast_length),
                exists=True,
            )
        return cls(filename=filename, exists=True)

class ThreadInfo(BaseModel):
    """Represents forecast and training thread details for a single asset."""
    forecast_running: bool = False
    forecast_thread_id: Optional[int] = None
    train_running: bool = False
    train_thread_id: Optional[int] = None