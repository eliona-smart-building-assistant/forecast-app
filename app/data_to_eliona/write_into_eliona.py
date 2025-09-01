import datetime
from eliona.api_client2 import ApiClient, Configuration, DataApi
from eliona.api_client2.models import Data
import pytz
import os
import pandas as pd

ELIONA_API_KEY = os.getenv("API_TOKEN")
ELIONA_HOST = os.getenv("API_ENDPOINT")

def write_into_eliona(asset_id, timestamp, data, name, prediction_length):
    configuration = Configuration(
        host=ELIONA_HOST, api_key={"ApiKeyAuth": ELIONA_API_KEY}
    )
    with ApiClient(configuration) as api_client:
        data_api = DataApi(api_client)

        # Prepare forecast name suffix
        forecast_name_suffix = f"{name}_forecast_{prediction_length}"
        ts_attr_name = f"{forecast_name_suffix}_ts"

        # Ensure timestamp is timezone-aware and in ISO format
        if isinstance(timestamp, pd.Timestamp):
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            timestamp_iso = timestamp.isoformat()
        elif isinstance(timestamp, datetime.datetime):
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=pytz.utc)
            timestamp_iso = timestamp.isoformat()
        else:
            raise ValueError("Unsupported timestamp type")

        # 1. Write predicted value to trend with future timestamp
        trend_data_dict = {forecast_name_suffix: float(data)}
        trend_data = Data(
            asset_id=asset_id,
            subtype="input",
            timestamp=timestamp_iso,
            data=trend_data_dict
        )
        data_api.put_data_trend(trend_data)

        # 2. Write last calculated value to heap with current timestamp
        heap_data_dict = {forecast_name_suffix: float(data)}
        heap_data = Data(
            asset_id=asset_id,
            subtype="input",
            data=heap_data_dict
        )
        data_api.put_data(heap_data)

        # 3. Write the predicted timestamp as a string attribute to heap
        ts_string_data_dict = {ts_attr_name: timestamp_iso}
        ts_string_data = Data(
            asset_id=asset_id,
            subtype="input",
            data=ts_string_data_dict
        )
        data_api.put_data(ts_string_data)