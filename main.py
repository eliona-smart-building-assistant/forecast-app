import db
import register_app
import uvicorn
import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_api():
    port = int(os.getenv("API_SERVER_PORT", 3000))
    uvicorn.run("api.openapi:app", host="0.0.0.0", port=port)

if __name__ == "__main__":
    db.create_schema_and_table()
    logger.info("Database schema and table created.")
    # register_app.Initialize()
    logger.info("App initialized.")
    start_api()
