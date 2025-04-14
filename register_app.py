from config import apps_api
import logging

logger = logging.getLogger(__name__)

def Initialize():

    app = apps_api.get_app_by_name("forecast")

    if not app.registered:
        apps_api.patch_app_by_name("forecast", True)
    else:
        logger.info("App 'forecast' already active.")
