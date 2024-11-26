import pandas as pd
import numpy as np
from datetime import timedelta
import eliona.api_client2
from eliona.api_client2.rest import ApiException
from eliona.api_client2.api.data_api import DataApi
import os
from sklearn.preprocessing import MinMaxScaler

# Set up configuration for the Eliona API
configuration = eliona.api_client2.Configuration(host=os.getenv("API_ENDPOINT"))
configuration.api_key["ApiKeyAuth"] = os.getenv("API_TOKEN")

# Create an instance of the API client
api_client = eliona.api_client2.ApiClient(configuration)
data_api = DataApi(api_client)


def get_trend_data(asset_id, start_date, end_date):
    asset_id = int(asset_id)
    from_date = start_date.isoformat()
    to_date = end_date.isoformat()
    try:
        print(f"Fetching data for asset {asset_id} from {from_date} to {to_date}")
        result = data_api.get_data_trends(
            from_date=from_date,
            to_date=to_date,
            asset_id=asset_id,
            data_subtype="input",
        )
        print(f"Received {len(result)} data points")
        return result
    except ApiException as e:
        print(f"Exception when calling DataApi->get_data_trends: {e}")
        return None


def fetch_data_in_chunks(asset_id, start_date, end_date):
    all_data = []
    current_start = start_date
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=15), end_date)
        data_chunk = get_trend_data(asset_id, current_start, current_end)
        if data_chunk:
            all_data.extend(data_chunk)
        current_start = current_end + timedelta(seconds=1)
    return all_data


def convert_to_pandas(data):
    # Dictionary to hold the rows, using the timestamp as the key
    formatted_data = {}

    for entry in data:
        # Extract timestamp and data
        timestamp = entry.timestamp
        data_dict = entry.data

        # If this timestamp already exists, update the existing row
        if timestamp in formatted_data:
            formatted_data[timestamp].update(data_dict)
        else:
            # Create a new row for this timestamp
            formatted_data[timestamp] = data_dict

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame.from_dict(formatted_data, orient="index")

    # Set the index (timestamp) as a proper datetime index
    df.index = pd.to_datetime(df.index, utc=True)

    # Convert the index to the desired timezone (e.g., Europe/Berlin)
    df.index = df.index.tz_convert("Europe/Berlin")

    # **Optional: Sort the DataFrame by index (timestamp)**
    df.sort_index(inplace=True)

    # Reset index to have 'timestamp' as a column
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
    # Fetch all data without filtering by attributes
    data = fetch_data_in_chunks(asset_id, start_date, end_date)

    # Convert data to pandas DataFrame
    df = convert_to_pandas(data)

    # Ensure 'timestamp' is datetime and sorted
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Initialize feature_attributes if None
    if feature_attributes is None:
        feature_attributes = []

    # List of possible time-based features with sine and cosine
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

    # Identify which time-based features are requested
    requested_time_features = [
        feat for feat in feature_attributes if feat in time_features
    ]

    # Remove time-based features from feature_attributes (since we'll process them separately)
    feature_attributes = [
        feat for feat in feature_attributes if feat not in time_features
    ]

    # Select only the target and remaining feature attributes
    attributes = [target_attribute] + feature_attributes

    # Initialize sets and dictionaries to keep track of computations
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

    # Process time-based features
    for feat in requested_time_features:
        # Check if the feature ends with '_sin' or '_cos'
        if feat.endswith("_sin"):
            base_feat = feat[:-4]  # Remove '_sin'
            transformation = "sin"
        elif feat.endswith("_cos"):
            base_feat = feat[:-4]  # Remove '_cos'
            transformation = "cos"
        else:
            # Not a recognized time feature with '_sin' or '_cos'
            continue

        # If the base time component has not been computed yet, compute it
        if base_feat not in computed_time_components:
            # Compute the time component
            if base_feat == "second_of_minute":
                df[base_feat] = df["timestamp"].dt.second
            elif base_feat == "minute_of_hour":
                df[base_feat] = df["timestamp"].dt.minute
            elif base_feat == "hour_of_day":
                df[base_feat] = df["timestamp"].dt.hour
            elif base_feat == "day_of_week":
                df[base_feat] = df["timestamp"].dt.weekday  # Monday=0, Sunday=6
            elif base_feat == "day_of_month":
                df[base_feat] = df["timestamp"].dt.day
            elif base_feat == "month_of_year":
                df[base_feat] = df["timestamp"].dt.month
            elif base_feat == "day_of_year":
                df[base_feat] = df["timestamp"].dt.dayofyear
            else:
                continue  # Unrecognized base feature
            computed_time_components.add(base_feat)

        # Get the period associated with the base feature
        period = base_feature_periods.get(base_feat)
        if period is None:
            continue  # Unrecognized base feature, skip

        # Apply the sine or cosine transformation
        if transformation == "sin":
            df[feat] = np.sin(2 * np.pi * df[base_feat] / period)
        elif transformation == "cos":
            df[feat] = np.cos(2 * np.pi * df[base_feat] / period)

        # Add the transformed feature to the attributes list
        attributes.append(feat)

        # Optionally, drop the base time component column if both transformations are computed or not needed
        # Check if both '_sin' and '_cos' variants are requested
        other_transformation = "cos" if transformation == "sin" else "sin"
        other_feat = base_feat + "_" + other_transformation
        if (other_feat in requested_time_features and other_feat in df.columns) or (
            other_feat not in requested_time_features
        ):
            # Both transformations have been computed or the other is not requested; drop base feature
            df.drop(columns=[base_feat], inplace=True)

    # Handle missing attributes by filling with zeros
    missing_attributes = [col for col in attributes if col not in df.columns]
    for col in missing_attributes:
        df[col] = 0

    # Keep only the relevant columns
    df = df[["timestamp"] + attributes]

    # Forward fill missing values for feature attributes (if any)
    if feature_attributes:
        df[feature_attributes] = df[feature_attributes].ffill()

    # Keep only the rows where the target attribute is not missing
    df = df[df[target_attribute].notna()]

    # Drop any remaining NaN values
    df.dropna(inplace=True)

    print("df.head()", df.head())
    return df


def prepare_data(
    data, context_length, forecast_length, target_attribute, feature_attributes
):
    """
    Prepares data for training a TensorFlow LSTM model, including scaling.
    Includes sequences of context_length for each attribute.

    :param data: Pandas DataFrame containing 'timestamp', target_attribute, and feature_attributes
    :param context_length: The number of timesteps used for context (input window)
    :param forecast_length: The number of timesteps ahead to predict
    :param target_attribute: The name of the target attribute (string)
    :param feature_attributes: List of feature attribute names (list of strings)
    :return: X, Y, scalers dictionary, last_timestamp
    """
    # Ensure the data is sorted by timestamp
    data = data.sort_values("timestamp").reset_index(drop=True)

    # Combine target and feature attributes
    if feature_attributes is None:
        feature_attributes = []
    all_attributes = [target_attribute] + feature_attributes

    # Initialize a scaler for each attribute and scale the data
    scalers = {}
    scaled_data = pd.DataFrame()
    for attr in all_attributes:
        scaler = MinMaxScaler(feature_range=(0, 1))
        values = data[attr].values.reshape(-1, 1)
        scaled_values = scaler.fit_transform(values).flatten()
        scaled_data[attr] = scaled_values
        scalers[attr] = scaler  # Store the scaler for each attribute

    X = []
    Y = []

    # Calculate the number of samples that can be generated
    total_samples = len(scaled_data) - context_length - forecast_length + 1

    # Loop over the dataset to create sequences
    for i in range(total_samples):
        # Extract input sequences for all attributes
        x = scaled_data[all_attributes].iloc[i : i + context_length].values
        # Extract the target value that is 'forecast_length' steps ahead
        y_index = i + context_length + forecast_length - 1
        if y_index < len(scaled_data):
            y = scaled_data[target_attribute].iloc[y_index]
            X.append(x)
            Y.append(y)
        else:
            break  # Break if the target index is out of bounds

    # Convert lists to numpy arrays
    X = np.array(X)  # Shape: [samples, context_length, num_features]
    Y = np.array(Y)  # Shape: [samples, ]

    # Extract the last Y's timestamp
    last_timestamp = data["timestamp"].iloc[len(data) - forecast_length]
    last_timestamp = pd.to_datetime(last_timestamp)
    print("X_shape", X.shape)
    print("Y_shape", Y.shape)
    return X, Y, scalers, last_timestamp


def prepare_data_for_forecast(
    data,
    context_length,
    forecast_length,
    scalers,
    last_timestamp,
    target_attribute,
    feature_attributes,
):
    """
    Prepares data for forecasting using a trained LSTM model.
    Includes sequences of context_length for each attribute.

    :param data: Pandas DataFrame containing 'timestamp', target_attribute, and feature_attributes
    :param context_length: The number of timesteps used for context (input window)
    :param forecast_length: The number of timesteps ahead to predict
    :param scalers: Dictionary of scalers for each attribute
    :param last_timestamp: The last timestamp from the training data
    :param target_attribute: The name of the target attribute (string)
    :param feature_attributes: List of feature attribute names (list of strings)
    :return: X_update, X_last, new_next_timestamp, last_y_timestamp_new
    """
    # Ensure the data is sorted by timestamp
    data = data.sort_values("timestamp").reset_index(drop=True)

    # Convert 'timestamp' to datetime if not already
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    # Combine target and feature attributes
    if feature_attributes is None:
        feature_attributes = []
    all_attributes = [target_attribute] + feature_attributes
    scaled_data = pd.DataFrame()

    # Use the loaded scalers to transform values
    for attr in all_attributes:
        if attr in data.columns:
            values = data[attr].values.reshape(-1, 1)
            scaler = scalers[attr]
            scaled_values = scaler.transform(values).flatten()
            scaled_data[attr] = scaled_values
        else:
            print(f"Attribute '{attr}' not found in data. Filling with zeros.")
            scaled_data[attr] = 0.0  # Or handle appropriately

    # Find the index corresponding to last_timestamp

    # If exact match not found, find the closest timestamp after last_timestamp
    indices = data[data["timestamp"] > last_timestamp].index

    if len(indices) == 0:
        print("No new data available after the last timestamp.")
        return None, None, None, None
    print("iding indices[0]", indices[:5])
    last_index = indices[0]
    print("last_timestamp", last_timestamp)
    print("last_index", last_index)
    print("timesamp at last_index", data["timestamp"].iloc[last_index])
    # Start index to include context_length steps before last_index
    start_index = last_index - context_length
    if start_index < 0:
        start_index = 0

    # Prepare sequences
    X_new = []
    target_timestamps = []

    for i in range(start_index, len(scaled_data) - context_length):
        x = scaled_data[all_attributes].iloc[i : i + context_length].values
        X_new.append(x)
        timestamp = data["timestamp"].iloc[i + context_length]
        target_timestamps.append(timestamp)

    if not X_new:
        print("No valid input sequences found after filtering.")
        return None, None, None, None

    X_new = np.array(X_new)  # Shape: [samples, context_length, num_features]

    # Separate X_new into X_update and X_last
    if len(X_new) > 1:
        X_update = X_new[:-1]
        X_last = X_new[-1].reshape((1, context_length, len(all_attributes)))
        last_y_timestamp_new = target_timestamps[-1]
    else:
        X_update = np.empty((0, context_length, len(all_attributes)))
        X_last = X_new[-1].reshape((1, context_length, len(all_attributes)))
        last_y_timestamp_new = target_timestamps[-1]

    # Calculate `new_next_timestamp`
    if len(data) >= 2:
        timestamp_diff = data["timestamp"].iloc[-1] - data["timestamp"].iloc[-2]
        new_next_timestamp = (
            data["timestamp"].iloc[-1] + timestamp_diff * forecast_length
        )
    else:
        new_next_timestamp = None
        print("Insufficient data for timestamp calculation.")
    print("X_update_shape", X_update.shape)
    print("X_last_shape", X_last.shape)
    return X_update, X_last, new_next_timestamp, last_y_timestamp_new
