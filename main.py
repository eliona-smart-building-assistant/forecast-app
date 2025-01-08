import db
import register_app
import app.app as app
import uvicorn
import logging
from concurrent.futures import ThreadPoolExecutor
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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
        logger.info("Schema and table creation completed.")

        register_app.Initialize()
        logger.info("App initialization completed.")

        logger.info("API started")
        SessionLocal, Asset = db.setup_database()
        logger.info("Database setup completed.")

        # Now, call the function to start the forecast and training processes
        app.app_background_worker(SessionLocal, Asset)
        logger.info("Background worker started.")
    except Exception as e:
        logger.error(f"Error in background tasks: {e}")


def run_with_retry(func):
    while True:
        try:
            func()
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}. Retrying...")


if __name__ == "__main__":
    with ThreadPoolExecutor() as executor:
        executor.submit(run_with_retry, start_api)
        executor.submit(run_with_retry, start_background_tasks)
        logger.info("API and background tasks submitted to executor")
