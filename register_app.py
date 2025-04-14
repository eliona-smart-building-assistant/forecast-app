import os
import logging

from eliona.api_client2 import (
    AppsApi,
    ApiClient,
    Configuration,
)


host = os.getenv("API_ENDPOINT")
api_key = os.getenv("API_TOKEN")
db_url = os.getenv("CONNECTION_STRING")
db_url_sql = db_url.replace("postgres", "postgresql")

print(f"db_url_sql: {db_url}")

configuration = Configuration(host=host)
configuration.api_key["ApiKeyAuth"] = api_key
api_client = ApiClient(configuration)


apps_api = AppsApi(api_client)

logger = logging.getLogger(__name__)

def Initialize():

    app = apps_api.get_app_by_name("forecast")

    if not app.registered:
        apps_api.patch_app_by_name("forecast", True)
        logger.info("App 'forecast' registered.")
    else:
        logger.info("App 'forecast' already active.")
