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

        forecast_name_suffix = f"{name}_forecast_{prediction_length}"
        data_dict = {f"{forecast_name_suffix}": float(data)}
        if isinstance(timestamp, pd.Timestamp):
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            timestamp = timestamp.isoformat()
        elif isinstance(timestamp, datetime):
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=pytz.utc)
            timestamp = timestamp.isoformat()

        data = Data(
            asset_id=asset_id, subtype="output", timestamp=timestamp, data=data_dict
        )
        data_api.put_data(data)
