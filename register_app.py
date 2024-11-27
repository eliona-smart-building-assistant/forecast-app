from config import apps_api

import logging

# Initialize the logger
logger = logging.getLogger(__name__)


def Initialize():

    app = apps_api.get_app_by_name("forecast")

    if not app.registered:
        apps_api.patch_app_by_name("forecast", True)
        logger.info("App 'forecast' registered.")

    # else:
    #     logger.info("App 'forecast' already active.")
