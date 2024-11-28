# filepath: /c:/Users/sti/eliona/pythonscriptesting/forecast-app/main.py
import db
import register_app
import app.app as app
import uvicorn
import logging
from concurrent.futures import ThreadPoolExecutor
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def start_api():
    port = int(os.getenv("API_SERVER_PORT", 3000))
    uvicorn.run("api.openapi:app", host="0.0.0.0", port=port)


def start_background_tasks():
    logger.info("Starting background tasks...")
    try:
        # Ensure that any multiprocessing setup is done properly here
        db.create_schema_and_table()

        #
        register_app.Initialize()

        logger.info("API started")
        SessionLocal, Asset = db.setup_database()

        # Now, call the function to start the forecast and training processes
        app.app_background_worker(SessionLocal, Asset)
    except Exception as e:
        logger.error(f"Error in background tasks: {e}")


if __name__ == "__main__":
    with ThreadPoolExecutor() as executor:
        future_api = executor.submit(start_api)
        future_background = executor.submit(start_background_tasks)
        logger.info("API and background tasks submitted to executor")
        logger.info(f"API task running: {future_api.running()}")
        logger.info(f"Background task running: {future_background.running()}")
