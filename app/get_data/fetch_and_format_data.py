import pandas as pd
import numpy as np
from datetime import timedelta
import eliona.api_client2
from eliona.api_client2.rest import ApiException
from eliona.api_client2.api.data_api import DataApi
import os
from sklearn.preprocessing import MinMaxScaler
import logging

from api.models import AssetModel

logger = logging.getLogger(__name__)

configuration = eliona.api_client2.Configuration(host=os.getenv("API_ENDPOINT"))
configuration.api_key["ApiKeyAuth"] = os.getenv("API_TOKEN")
api_client = eliona.api_client2.ApiClient(configuration)
data_api = DataApi(api_client)


def get_trend_data(asset_id, start_date, end_date):
    asset_id = int(asset_id)
    from_date = start_date.isoformat()
    to_date = end_date.isoformat()
    try:
        result = data_api.get_data_trends(
            from_date=from_date,
            to_date=to_date,
            asset_id=asset_id,
            data_subtype="input",
        )
        return result
    except ApiException as e:
        logger.info(f"Exception when calling DataApi->get_data_trends {asset_id}: {e}")
        return None


def fetch_data_in_chunks(asset_id, start_date, end_date):
    all_data = []
    current_start = start_date
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=5), end_date)
        data_chunk = get_trend_data(asset_id, current_start, current_end)
        if data_chunk:
            all_data.extend(data_chunk)
        current_start = current_end + timedelta(seconds=1)
    return all_data


def convert_to_pandas(data):
    formatted_data = {}

    for entry in data:
        timestamp = entry.timestamp
        data_dict = entry.data

        if timestamp in formatted_data:
            formatted_data[timestamp].update(data_dict)
        else:
            formatted_data[timestamp] = data_dict

    df = pd.DataFrame.from_dict(formatted_data, orient="index")
    df.index = pd.to_datetime(df.index, utc=True)
    df.index = df.index.tz_convert("Europe/Berlin")
    df.sort_index(inplace=True)
    df.reset_index(inplace=True)
    df.rename(columns={"index": "timestamp"}, inplace=True)

    return df


def fetch_pandas_data(
    asset_id,
    start_date,
    end_date,
    target_attribute,
    feature_attributes,
):
    data = fetch_data_in_chunks(asset_id, start_date, end_date)
    df = convert_to_pandas(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    if feature_attributes is None:
        feature_attributes = []

    time_features = [
        "second_of_minute_sin",
        "second_of_minute_cos",
        "minute_of_hour_sin",
        "minute_of_hour_cos",
        "hour_of_day_sin",
        "hour_of_day_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "day_of_month_sin",
        "day_of_month_cos",
        "month_of_year_sin",
        "month_of_year_cos",
        "day_of_year_sin",
        "day_of_year_cos",
    ]

    requested_time_features = [
        feat for feat in feature_attributes if feat in time_features
    ]

    feature_attributes = [
        feat for feat in feature_attributes if feat not in time_features
    ]

    attributes = [target_attribute] + feature_attributes

    computed_time_components = set()
    base_feature_periods = {
        "second_of_minute": 60,
        "minute_of_hour": 60,
        "hour_of_day": 24,
        "day_of_week": 7,
        "day_of_month": 31,
        "month_of_year": 12,
        "day_of_year": 366,
    }

    for feat in requested_time_features:
        if feat.endswith("_sin"):
            base_feat = feat[:-4]  
            transformation = "sin"
        elif feat.endswith("_cos"):
            base_feat = feat[:-4]  
            transformation = "cos"
        else:
            continue

        if base_feat not in computed_time_components:
            if base_feat == "second_of_minute":
                df[base_feat] = df["timestamp"].dt.second
            elif base_feat == "minute_of_hour":
                df[base_feat] = df["timestamp"].dt.minute
            elif base_feat == "hour_of_day":
                df[base_feat] = df["timestamp"].dt.hour
            elif base_feat == "day_of_week":
                df[base_feat] = df["timestamp"].dt.weekday  
            elif base_feat == "day_of_month":
                df[base_feat] = df["timestamp"].dt.day
            elif base_feat == "month_of_year":
                df[base_feat] = df["timestamp"].dt.month
            elif base_feat == "day_of_year":
                df[base_feat] = df["timestamp"].dt.dayofyear
            else:
                continue 
            computed_time_components.add(base_feat)

        period = base_feature_periods.get(base_feat)
        if period is None:
            continue  

        if transformation == "sin":
            df[feat] = np.sin(2 * np.pi * df[base_feat] / period)
        elif transformation == "cos":
            df[feat] = np.cos(2 * np.pi * df[base_feat] / period)

        attributes.append(feat)

        other_transformation = "cos" if transformation == "sin" else "sin"
        other_feat = base_feat + "_" + other_transformation
        if (other_feat in requested_time_features and other_feat in df.columns) or (
            other_feat not in requested_time_features
        ):
            df.drop(columns=[base_feat], inplace=True)

    missing_attributes = [col for col in attributes if col not in df.columns]
    for col in missing_attributes:
        df[col] = 0

    df = df[["timestamp"] + attributes]

    if feature_attributes:
        df[feature_attributes] = df[feature_attributes].ffill()

    df = df[df[target_attribute].notna()]
    df.dropna(inplace=True)

    return df


def prepare_data(
    asset: AssetModel,
    data,
):
    data = data.sort_values("timestamp").reset_index(drop=True)
    all_attributes = [asset.target_attribute] + asset.feature_attributes

    scalers = {}
    scaled_data = pd.DataFrame()
    for attr in all_attributes:
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        values = data[attr].values.reshape(-1, 1)

        if attr == asset.target_attribute and asset.parameters and (asset.parameters.binary_encoding or asset.parameters.num_classes):
            scaled_values = values.flatten() 
        else:
            scaled_values = scaler.fit_transform(values).flatten()

        scaled_data[attr] = scaled_values
        scalers[attr] = scaler  

    X = []
    Y = []

    total_samples = len(scaled_data) - asset.context_length - asset.forecast_length + 1

    for i in range(total_samples):
        x = scaled_data[all_attributes].iloc[i : i + asset.context_length].values
        y_index = i + asset.context_length + asset.forecast_length - 1
        if y_index < len(scaled_data):
            y = scaled_data[asset.target_attribute].iloc[y_index]
            X.append(x)
            Y.append(y)
        else:
            break  

    X = np.array(X)  
    Y = np.array(Y)  

    last_timestamp = data["timestamp"].iloc[len(data) - asset.forecast_length]
    last_timestamp = pd.to_datetime(last_timestamp)
    return X, Y, scalers, last_timestamp


def prepare_data_for_forecast(
    asset: AssetModel,
    data,
    scalers,
    last_timestamp,
):
 
    data = data.sort_values("timestamp").reset_index(drop=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    if asset.feature_attributes is None:
        asset.feature_attributes = []
    all_attributes = [asset.target_attribute] + asset.feature_attributes
    scaled_data = pd.DataFrame()

    for attr in all_attributes:
        if attr in data.columns:
            values = data[attr].values.reshape(-1, 1)
            if attr == asset.target_attribute and (asset.parameters.binary_encoding or asset.parameters.num_classes):
                scaled_values = values.flatten()
            elif attr in scalers:
                scaler = scalers[attr]
                try:
                    scaled_values = scaler.transform(values).flatten()
                except Exception as e:
                    logger.warning(f"Scaler issue for {asset.id} '{attr}': {e}. Using raw values.")
                    scaled_values = values.flatten()
            else:
                logger.warning(f"No scaler found for {asset.id} '{attr}'. Using raw values.")
                scaled_values = values.flatten()

            scaled_data[attr] = scaled_values
        else:
            logger.warning(f"Attribute '{attr}' not found in data {asset.id}. Filling with zeros.")
            scaled_data[attr] = 0.0 
            
    indices = data[data["timestamp"] > last_timestamp].index
    if len(indices) == 0:
        logger.warn(f"No new data available after the last timestamp for {asset.id}")
        return None, None, None, None

    last_index = indices[0]
    start_index = last_index - asset.context_length
    if start_index < 0:
        start_index = 0

    X_new = []
    for i in range(start_index, len(scaled_data) - asset.context_length + 1):
        x = scaled_data[all_attributes].iloc[i : i + asset.context_length].values
        X_new.append(x)

    if not X_new:                         # nothing new → bail out early
        logger.info(f"No valid input sequences found after filtering for {asset.id}")
        return None, None, None, None

    X_new = np.array(X_new)

    # --- split into “state-update” and “real forecast” ------------------------
    if len(X_new) > 1:
        X_update = X_new[:-1]             # keep state warm
    else:
        X_update = np.empty((0, asset.context_length, len(all_attributes)))

    X_last = X_new[-1].reshape((1, asset.context_length, len(all_attributes)))

    # timestamp of the last observed target value
    last_y_timestamp_new = data["timestamp"].iloc[-1]

    if len(data) >= 2:
        timestamp_diffs = data["timestamp"].diff().dropna()
        timestamp_diff = timestamp_diffs.mean()
        new_next_timestamp = (
            data["timestamp"].iloc[-1] + timestamp_diff * asset.forecast_length
        )
    else:
        new_next_timestamp = None
        logger.info(f"Insufficient data for timestamp calculation for {asset.id}")

    return X_update, X_last, new_next_timestamp, last_y_timestamp_new
