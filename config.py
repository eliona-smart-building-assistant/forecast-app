import os
from eliona.api_client2 import (
    AppsApi,
    AssetsApi,
    ApiClient,
    Configuration,
    AssetTypesApi,
    ApiException,
    AssetTypeAttribute,
)


host = os.getenv("API_ENDPOINT")
api_key = os.getenv("API_TOKEN")
db_url = os.getenv("CONNECTION_STRING")
db_url_sql = db_url.replace("postgres", "postgresql")
port = os.getenv("API_SERVER_PORT")

configuration = Configuration(host=host)
configuration.api_key["ApiKeyAuth"] = api_key
api_client = ApiClient(configuration)


apps_api = AppsApi(api_client)
assets_api = AssetsApi(api_client)
asset_types_api = AssetTypesApi(api_client)
